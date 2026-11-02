// mc_density — Monte Carlo solid fraction of the ballast from out2.dat
//
// One streaming pass over out2.dat (one frame in memory), several measurement
// boxes per pass, optional exclusion of the tool / container volume, slice
// profiles, and a bound-sensitivity check.  Time is read from the file (istep)
// plus iloop/dt read from in_bc.dat / in_gm.dat next to out2.dat.
//
// Build:  nvcc -O3 -std=c++17 -arch=native -o mc_density mc_density.cu   (GPU)
//         g++  -O3 -std=c++17 -fopenmp -x c++ -o mc_density_cpu mc_density.cu  (CPU)
//
// Usage:  mc_density out2.dat [options]      (see usage() below)
//
// Point classes (per Monte Carlo point, per frame):
//   solid    inside at least one ballast sphere (spheres 1..nballast)
//   excluded inside a tool sphere (mode>=1) or a container/step sphere (mode 2);
//            removed from the volume, i.e. Phi = solid / (N - excluded)
//   void     otherwise
// The same random points are used for every frame (seeded per box), so
// frame-to-frame differences are more precise than the absolute sigma.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #define HD __host__ __device__
  #define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#else
  #define HD
#endif

#ifndef __CUDACC__
struct float4 { float x, y, z, w; };
#endif

enum { T_BALLAST = 0, T_WALL = 1, T_TOOL = 2 };

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
struct Sub { std::string name; double b[6]; };
struct Box {
    std::string name;
    double b[6];          // x0 x1 y0 y1 z0 z1 (mm)
    long   npts;
    int    mode;          // 0 none, 1 exclude tool, 2 exclude tool + container
    double slice;         // slice width (mm), 0 = no profile
    std::string axes;     // subset of "xyz"
    std::vector<Sub> subs;
    bool   is_sens = false;
    // derived
    long off = 0;
    std::vector<int16_t> bin[3];
    std::vector<uint16_t> submask;
    int nbin[3] = {0, 0, 0};
    // per-frame results kept for the summary
    std::vector<double> phi_hist;
    FILE *f_main = nullptr, *f_prof[3] = {nullptr, nullptr, nullptr}, *f_sens = nullptr;
};

static void usage() {
    fprintf(stderr,
"usage: mc_density out2.dat [options]\n"
"  --boxes FILE     box definitions (default: built-in set, written to OUTDIR/mc_boxes_used.txt)\n"
"  --outdir DIR     output directory (default: <dir of out2.dat>/mc)\n"
"  --seed N         random seed (default 42)\n"
"  --npts N         override the number of points of every box\n"
"  --ballast N      ballast spheres are 1..N (default 46711)\n"
"  --tool A:B       tool spheres A..B, 1-based (default 54122:57136)\n"
"  --every K        analyse every K-th frame (default 1)\n"
"  --max_frames N   stop after N analysed frames\n"
"  --dt X           time step (default: from in_gm.dat, else 2e-6)\n"
"  --iloop N        step offset (default: from in_bc.dat, else 0)\n"
"  --avg_n N        frames averaged for the summary (default 10)\n"
"  --cpu            force the CPU path (GPU build)\n"
"Box file lines (# comments):\n"
"  box  NAME x0 x1 y0 y1 z0 z1 NPTS MODE SLICE AXES   MODE 0/1/2, SLICE mm (0 = none), AXES xyz or -\n"
"  sens NAME DELTA NPTS     sensitivity of box NAME: each bound moved by +-DELTA\n");
    exit(1);
}

static const char *DEFAULT_BOXES =
"# name      x0   x1   y0   y1   z0   z1   npts     mode slice axes\n"
"box zone    400  530   70  530   50  250  1000000   0    10   xyz\n"
"box valid   400  500  200  400   50  250  1000000   0    0    -\n"
"box global    0  600    0  600    0  200  1000000   1    0    -\n"
"box profx     0  600   70  530   50  250  2000000   2    10   x\n"
"box profy   400  530    0  600   50  250  2000000   2    10   y\n"
"box profz   400  530   70  530    0  300  2000000   2    10   z\n"
"sens zone 25 4000000\n";

static std::vector<Box> parse_boxes(const std::string &text) {
    std::vector<Box> boxes;
    size_t p = 0;
    int lineno = 0;
    while (p < text.size()) {
        size_t e = text.find('\n', p);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(p, e - p);
        p = e + 1; lineno++;
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        char kw[32], name[64], axes[8] = "-";
        double b[6], slice = 0, delta;
        long npts; int mode = 0;
        if (sscanf(line.c_str(), "%31s", kw) != 1) continue;
        if (!strcmp(kw, "box")) {
            int n = sscanf(line.c_str(), "%*s %63s %lf %lf %lf %lf %lf %lf %ld %d %lf %7s",
                           name, &b[0], &b[1], &b[2], &b[3], &b[4], &b[5], &npts, &mode, &slice, axes);
            if (n < 8) { fprintf(stderr, "box file line %d: need at least name + 6 bounds + npts\n", lineno); exit(1); }
            Box bx; bx.name = name; memcpy(bx.b, b, sizeof b); bx.npts = npts;
            bx.mode = mode; bx.slice = slice; bx.axes = axes;
            for (int k = 0; k < 3; k++)
                if (bx.b[2*k+1] <= bx.b[2*k]) { fprintf(stderr, "box %s: empty range\n", name); exit(1); }
            boxes.push_back(bx);
        } else if (!strcmp(kw, "sens")) {
            if (sscanf(line.c_str(), "%*s %63s %lf %ld", name, &delta, &npts) != 3) {
                fprintf(stderr, "box file line %d: sens NAME DELTA NPTS\n", lineno); exit(1);
            }
            const Box *src = nullptr;
            for (auto &q : boxes) if (q.name == name) src = &q;
            if (!src) { fprintf(stderr, "sens: box %s must be defined before\n", name); exit(1); }
            Box s; s.name = std::string(name) + "_sens"; s.npts = npts; s.mode = src->mode;
            s.is_sens = true;
            for (int k = 0; k < 6; k++) s.b[k] = src->b[k] + ((k % 2) ? delta : -delta);
            Sub base; base.name = "base"; memcpy(base.b, src->b, sizeof base.b);
            s.subs.push_back(base);
            const char *bn[6] = {"x0", "x1", "y0", "y1", "z0", "z1"};
            for (int k = 0; k < 6; k++)
                for (int sg = -1; sg <= 1; sg += 2) {
                    Sub q = base; q.b[k] += sg * delta;
                    char nm[32]; snprintf(nm, sizeof nm, "%s%c%g", bn[k], sg < 0 ? '-' : '+', delta);
                    q.name = nm; s.subs.push_back(q);
                }
            Sub shr = base, grw = base;
            for (int k = 0; k < 6; k++) { shr.b[k] += ((k % 2) ? -delta : delta); grw.b[k] = s.b[k]; }
            shr.name = "all_shrink"; grw.name = "all_grow";
            s.subs.push_back(shr); s.subs.push_back(grw);   // 15 subs
            boxes.push_back(s);
        } else {
            fprintf(stderr, "box file line %d: unknown keyword %s\n", lineno, kw); exit(1);
        }
    }
    return boxes;
}

// ----------------------------------------------------------------------------
// Buffered token reader (out2.dat is several GB; one frame at a time)
// ----------------------------------------------------------------------------
struct Reader {
    FILE *f = nullptr;
    std::vector<char> buf;
    size_t pos = 0, len = 0;
    bool eof = false;
    unsigned long long consumed = 0;
    explicit Reader(FILE *fp) : f(fp), buf((1u << 24) + 1) {}
    bool fill(size_t need) {           // make at least `need` bytes available
        if (len - pos >= need) return true;
        if (eof) return len > pos;
        memmove(buf.data(), buf.data() + pos, len - pos);
        len -= pos; consumed += pos; pos = 0;
        size_t r = fread(buf.data() + len, 1, buf.size() - 1 - len, f);
        len += r;
        if (r == 0) eof = true;
        buf[len] = 0;
        return len - pos >= need || len > pos;
    }
    bool skip_ws() {
        for (;;) {
            if (pos >= len && !fill(1)) return false;
            char c = buf[pos];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') return true;
            pos++;
        }
    }
    bool num(double &v) {
        if (!skip_ws()) return false;
        fill(64);
        char *s = buf.data() + pos, *e;
        v = strtod(s, &e);
        if (e == s) return false;
        pos += e - s;
        return true;
    }
    bool integer(long &v) {
        if (!skip_ws()) return false;
        fill(64);
        char *s = buf.data() + pos, *e;
        v = strtol(s, &e, 10);
        if (e == s) return false;
        pos += e - s;
        return true;
    }
    bool skip_token() {
        if (!skip_ws()) return false;
        for (;;) {
            if (pos >= len && !fill(1)) return true;
            char c = buf[pos];
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') return true;
            pos++;
        }
    }
};

// ----------------------------------------------------------------------------
// Uniform grid of spheres (cell size >= max radius, so a 3x3x3 search suffices)
// ----------------------------------------------------------------------------
struct GridView {
    float ox, oy, oz, inv_h;
    int nx, ny, nz;
    const int *start;      // ncell + 1
    const float4 *sph;     // x, y, z, r^2 (sorted by cell)
    const unsigned char *type;
};

HD inline unsigned char classify_point(const GridView &g, float px, float py, float pz, int mode) {
    int ix = (int)floorf((px - g.ox) * g.inv_h);
    int iy = (int)floorf((py - g.oy) * g.inv_h);
    int iz = (int)floorf((pz - g.oz) * g.inv_h);
    bool solid = false;
    for (int dz = -1; dz <= 1; dz++) {
        int cz = iz + dz; if (cz < 0 || cz >= g.nz) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int cy = iy + dy; if (cy < 0 || cy >= g.ny) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int cx = ix + dx; if (cx < 0 || cx >= g.nx) continue;
                int c = (cz * g.ny + cy) * g.nx + cx;
                for (int s = g.start[c]; s < g.start[c + 1]; s++) {
                    float4 q = g.sph[s];
                    float ddx = px - q.x, ddy = py - q.y, ddz = pz - q.z;
                    if (ddx*ddx + ddy*ddy + ddz*ddz >= q.w) continue;
                    unsigned char t = g.type[s];
                    if (t == T_BALLAST) { solid = true; if (mode == 0) return 1; }
                    else if ((t == T_TOOL && mode >= 1) || (t == T_WALL && mode >= 2)) return 2;
                }
            }
        }
    }
    return solid ? 1 : 0;
}

#ifdef __CUDACC__
__global__ void k_classify(GridView g, const float4 *pts, long n, int mode, unsigned char *out) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = pts[i];
    out[i] = classify_point(g, p.x, p.y, p.z, mode);
}
#endif

// ----------------------------------------------------------------------------
static std::string dirname_of(const std::string &p) {
    size_t s = p.find_last_of('/');
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}
static std::string read_file(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) return "";
    std::string s; char b[65536]; size_t r;
    while ((r = fread(b, 1, sizeof b, f)) > 0) s.append(b, r);
    fclose(f);
    return s;
}
static double fortran_atof(std::string s) {
    for (auto &c : s) if (c == 'D' || c == 'd') c = 'E';
    return atof(s.c_str());
}

int main(int argc, char **argv) {
    if (argc < 2) usage();
    std::string in = argv[1], boxfile, outdir;
    unsigned long long seed = 42;
    long npts_override = 0, nballast = 46711, tool_a = 54122, tool_b = 57136;
    long every = 1, max_frames = -1, iloop = -1, avg_n = 10;
    double dt = -1;
    bool force_cpu = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char * { if (i + 1 >= argc) usage(); return argv[++i]; };
        if      (a == "--boxes")      boxfile = nxt();
        else if (a == "--outdir")     outdir = nxt();
        else if (a == "--seed")       seed = strtoull(nxt(), 0, 10);
        else if (a == "--npts")       npts_override = atol(nxt());
        else if (a == "--ballast")    nballast = atol(nxt());
        else if (a == "--tool")       { if (sscanf(nxt(), "%ld:%ld", &tool_a, &tool_b) != 2) usage(); }
        else if (a == "--every")      every = std::max(1L, atol(nxt()));
        else if (a == "--max_frames") max_frames = atol(nxt());
        else if (a == "--dt")         dt = atof(nxt());
        else if (a == "--iloop")      iloop = atol(nxt());
        else if (a == "--avg_n")      avg_n = std::max(1L, atol(nxt()));
        else if (a == "--cpu")        force_cpu = true;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); }
    }
    std::string dir = dirname_of(in);
    if (outdir.empty()) outdir = dir + "/mc";
    if (system(("mkdir -p '" + outdir + "'").c_str()) != 0) { fprintf(stderr, "cannot create %s\n", outdir.c_str()); return 1; }

    // time axis: t = (iloop + istep) * dt
    if (dt < 0) {
        FILE *f = fopen((dir + "/in_gm.dat").c_str(), "r");
        char tok[64];
        if (f && fscanf(f, "%63s", tok) == 1) dt = fortran_atof(tok);
        if (f) fclose(f);
        if (dt <= 0) { dt = 2e-6; fprintf(stderr, "warning: dt not found in in_gm.dat, using 2e-6\n"); }
    }
    if (iloop < 0) {
        FILE *f = fopen((dir + "/in_bc.dat").c_str(), "r");
        long icld;
        if (!(f && fscanf(f, "%ld %ld", &icld, &iloop) == 2)) { iloop = 0; fprintf(stderr, "warning: iloop not found in in_bc.dat, using 0\n"); }
        if (f) fclose(f);
    }

    // boxes
    std::string btxt = boxfile.empty() ? std::string(DEFAULT_BOXES) : read_file(boxfile);
    if (btxt.empty()) { fprintf(stderr, "cannot read %s\n", boxfile.c_str()); return 1; }
    std::vector<Box> boxes = parse_boxes(btxt);
    if (boxes.empty()) { fprintf(stderr, "no boxes\n"); return 1; }
    { FILE *f = fopen((outdir + "/mc_boxes_used.txt").c_str(), "w"); if (f) { fputs(btxt.c_str(), f); fclose(f); } }

    // ---- open out2.dat, header ----
    FILE *fin = fopen(in.c_str(), "rb");
    if (!fin) { fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }
    Reader rd(fin);
    long ne, nstep;
    if (!rd.integer(ne) || !rd.integer(nstep)) { fprintf(stderr, "bad out2.dat header\n"); return 1; }
    std::vector<float> rad(ne);
    double rmax = 0;
    for (long i = 0; i < ne; i++) {
        long id; double r;
        if (!rd.integer(id) || !rd.num(r) || id != i + 1) { fprintf(stderr, "bad radius line %ld\n", i + 1); return 1; }
        rad[i] = (float)r;
    }
    if (nballast > ne || tool_b > ne || tool_a < 1 || tool_a > tool_b || tool_a <= nballast) {
        fprintf(stderr, "inconsistent ranges: ne=%ld ballast=1..%ld tool=%ld..%ld\n", ne, nballast, tool_a, tool_b);
        return 1;
    }
    std::vector<unsigned char> stype(ne);
    for (long i = 0; i < ne; i++) {
        stype[i] = (i < nballast) ? T_BALLAST : (i >= tool_a - 1 && i <= tool_b - 1) ? T_TOOL : T_WALL;
        rmax = std::max(rmax, (double)rad[i]);
    }
    printf("out2.dat: %ld spheres, nstep %ld | ballast 1..%ld, tool %ld..%ld, rmax %.3f\n",
           ne, nstep, nballast, tool_a, tool_b, rmax);
    printf("time: t = (%ld + istep) * %g\n", iloop, dt);

    // ---- points (same for every frame) and per-point bins ----
    long ntot = 0;
    for (auto &bx : boxes) {
        if (npts_override > 0) bx.npts = npts_override;
        bx.off = ntot; ntot += bx.npts;
        if (bx.subs.size() > 16) { fprintf(stderr, "max 16 sub-boxes\n"); return 1; }
    }
    std::vector<float4> pts(ntot);
    double dom[6] = {1e30, -1e30, 1e30, -1e30, 1e30, -1e30};
    for (size_t ib = 0; ib < boxes.size(); ib++) {
        Box &bx = boxes[ib];
        std::mt19937_64 rng(seed + 1000003ULL * ib);
        std::uniform_real_distribution<double> ux(bx.b[0], bx.b[1]), uy(bx.b[2], bx.b[3]), uz(bx.b[4], bx.b[5]);
        for (int k = 0; k < 3; k++) {
            dom[2*k] = std::min(dom[2*k], bx.b[2*k]); dom[2*k+1] = std::max(dom[2*k+1], bx.b[2*k+1]);
            if (bx.slice > 0 && bx.axes.find("xyz"[k]) != std::string::npos) {
                bx.nbin[k] = (int)ceil((bx.b[2*k+1] - bx.b[2*k]) / bx.slice - 1e-9);
                bx.bin[k].resize(bx.npts);
            }
        }
        if (!bx.subs.empty()) bx.submask.resize(bx.npts);
        for (long i = 0; i < bx.npts; i++) {
            double c[3] = {ux(rng), uy(rng), uz(rng)};
            pts[bx.off + i] = {(float)c[0], (float)c[1], (float)c[2], 0.f};
            for (int k = 0; k < 3; k++)
                if (bx.nbin[k]) bx.bin[k][i] = (int16_t)std::min(bx.nbin[k] - 1, (int)((c[k] - bx.b[2*k]) / bx.slice));
            if (!bx.subs.empty()) {
                uint16_t m = 0;
                for (size_t s = 0; s < bx.subs.size(); s++) {
                    const double *q = bx.subs[s].b;
                    if (c[0] >= q[0] && c[0] < q[1] && c[1] >= q[2] && c[1] < q[3] && c[2] >= q[4] && c[2] < q[5]) m |= (1u << s);
                }
                bx.submask[i] = m;
            }
        }
    }
    printf("%zu boxes, %ld points in total, seed %llu\n", boxes.size(), ntot, seed);

    // grid domain = union of boxes + rmax
    const float h = (float)rmax;
    float g_o[3]; int g_n[3];
    for (int k = 0; k < 3; k++) {
        g_o[k] = (float)(dom[2*k] - rmax);
        g_n[k] = (int)ceil((dom[2*k+1] + rmax - g_o[k]) / h) + 1;
    }
    long ncell = (long)g_n[0] * g_n[1] * g_n[2];

    // ---- output files ----
    for (auto &bx : boxes) {
        std::string base = outdir + "/MC_" + bx.name;
        if (!bx.is_sens) {
            bx.f_main = fopen((base + ".txt").c_str(), "w");
            fprintf(bx.f_main, "# box %s  x %g %g  y %g %g  z %g %g  npts %ld  mode %d  seed %llu\n",
                    bx.name.c_str(), bx.b[0], bx.b[1], bx.b[2], bx.b[3], bx.b[4], bx.b[5], bx.npts, bx.mode, seed);
            fprintf(bx.f_main, "# frame istep time_s phi sigma excluded_frac\n");
            for (int k = 0; k < 3; k++) if (bx.nbin[k]) {
                bx.f_prof[k] = fopen((base + "_prof_" + "xyz"[k] + ".txt").c_str(), "w");
                fprintf(bx.f_prof[k], "# box %s, phi in %g mm slices along %c (NaN = no valid points); row 1 = slice centres\ntime_s",
                        bx.name.c_str(), bx.slice, "xyz"[k]);
                for (int j = 0; j < bx.nbin[k]; j++)
                    fprintf(bx.f_prof[k], " %g", 0.5 * (bx.b[2*k] + j * bx.slice + std::min(bx.b[2*k] + (j + 1) * bx.slice, bx.b[2*k+1])));
                fprintf(bx.f_prof[k], "\n");
            }
        } else {
            bx.f_sens = fopen((base + ".txt").c_str(), "w");
            fprintf(bx.f_sens, "# sensitivity: phi with box bounds moved (parent %s, %ld points)\ntime_s", bx.name.c_str(), bx.npts);
            for (auto &s : bx.subs) fprintf(bx.f_sens, " %s", s.name.c_str());
            fprintf(bx.f_sens, "\n");
        }
    }
    FILE *f_frames = fopen((outdir + "/MC_frames.txt").c_str(), "w");
    fprintf(f_frames, "# frame istep time_s tool_tip_z_mm tool_min_x_mm tool_max_x_mm\n");

    // ---- GPU setup ----
    bool use_gpu = false;
#ifdef __CUDACC__
    int ndev = 0;
    if (!force_cpu && cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) use_gpu = true;
    float4 *d_pts = nullptr, *d_sph = nullptr; int *d_start = nullptr;
    unsigned char *d_type = nullptr, *d_flag = nullptr;
    if (use_gpu) {
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
        printf("GPU: %s\n", pr.name);
        CK(cudaMalloc(&d_pts, ntot * sizeof(float4)));
        CK(cudaMemcpy(d_pts, pts.data(), ntot * sizeof(float4), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_sph, ne * sizeof(float4)));
        CK(cudaMalloc(&d_type, ne));
        CK(cudaMalloc(&d_start, (ncell + 1) * sizeof(int)));
        CK(cudaMalloc(&d_flag, ntot));
    }
#endif
    if (!use_gpu) printf("running on CPU\n");

    std::vector<double> x(ne), y(ne), z(ne);
    std::vector<int> cnt(ncell + 1), cellof(ne);
    std::vector<float4> sph(ne);
    std::vector<unsigned char> styp(ne), flag(ntot);

    auto t_start = std::chrono::steady_clock::now();
    long frame = -1, analysed = 0;
    std::vector<double> times;
    for (;;) {
        long istep;
        if (!rd.integer(istep)) break;              // clean EOF
        frame++;
        bool ok = true;
        for (long i = 0; i < ne && ok; i++) {
            long id;
            ok = rd.integer(id) && id == i + 1 && rd.num(x[i]) && rd.num(y[i]) && rd.num(z[i])
                 && rd.skip_token() && rd.skip_token() && rd.skip_token();
        }
        if (!ok) { fprintf(stderr, "frame %ld (istep %ld) incomplete or malformed -> stopped\n", frame, istep); break; }
        if (frame % every) continue;
        double t = (iloop + istep) * dt;

        // tool diagnostics
        double tip = 1e30, txmin = 1e30, txmax = -1e30;
        for (long i = tool_a - 1; i < tool_b; i++) {
            tip = std::min(tip, z[i] - rad[i]); txmin = std::min(txmin, x[i] - rad[i]); txmax = std::max(txmax, x[i] + rad[i]);
        }
        fprintf(f_frames, "%ld %ld %.6f %.4f %.4f %.4f\n", frame, istep, t, tip, txmin, txmax);
        if (analysed == 0) printf("first frame: istep %ld, t = %.4f s, tool tip z = %.2f mm, tool x %.1f..%.1f\n",
                                  istep, t, tip, txmin, txmax);

        // grid build (culled to the domain)
        std::fill(cnt.begin(), cnt.end(), 0);
        long nin = 0;
        for (long i = 0; i < ne; i++) {
            int ix = (int)floorf(((float)x[i] - g_o[0]) / h), iy = (int)floorf(((float)y[i] - g_o[1]) / h),
                iz = (int)floorf(((float)z[i] - g_o[2]) / h);
            if (ix < 0 || iy < 0 || iz < 0 || ix >= g_n[0] || iy >= g_n[1] || iz >= g_n[2]) { cellof[i] = -1; continue; }
            int c = (iz * g_n[1] + iy) * g_n[0] + ix;
            cellof[i] = c; cnt[c + 1]++; nin++;
        }
        for (long c = 0; c < ncell; c++) cnt[c + 1] += cnt[c];
        std::vector<int> fillp(cnt.begin(), cnt.end() - 1);
        for (long i = 0; i < ne; i++) {
            if (cellof[i] < 0) continue;
            int s = fillp[cellof[i]]++;
            sph[s] = {(float)x[i], (float)y[i], (float)z[i], rad[i] * rad[i]};
            styp[s] = stype[i];
        }

        GridView gv{g_o[0], g_o[1], g_o[2], 1.0f / h, g_n[0], g_n[1], g_n[2], nullptr, nullptr, nullptr};
        if (use_gpu) {
#ifdef __CUDACC__
            CK(cudaMemcpy(d_sph, sph.data(), nin * sizeof(float4), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_type, styp.data(), nin, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_start, cnt.data(), (ncell + 1) * sizeof(int), cudaMemcpyHostToDevice));
            gv.start = d_start; gv.sph = d_sph; gv.type = d_type;
            for (auto &bx : boxes) {
                int nt = 256; long nb = (bx.npts + nt - 1) / nt;
                k_classify<<<(unsigned)nb, nt>>>(gv, d_pts + bx.off, bx.npts, bx.mode, d_flag + bx.off);
            }
            CK(cudaGetLastError());
            CK(cudaMemcpy(flag.data(), d_flag, ntot, cudaMemcpyDeviceToHost));
#endif
        } else {
            gv.start = cnt.data(); gv.sph = sph.data(); gv.type = styp.data();
            for (auto &bx : boxes) {
                const long off = bx.off; const int mode = bx.mode;
                #pragma omp parallel for schedule(static)
                for (long i = 0; i < bx.npts; i++)
                    flag[off + i] = classify_point(gv, pts[off + i].x, pts[off + i].y, pts[off + i].z, mode);
            }
        }

        // counting
        for (auto &bx : boxes) {
            const unsigned char *fl = flag.data() + bx.off;
            if (!bx.is_sens) {
                long ns = 0, nx = 0;
                std::vector<long> ps[3], pv[3];
                for (int k = 0; k < 3; k++) if (bx.nbin[k]) { ps[k].assign(bx.nbin[k], 0); pv[k].assign(bx.nbin[k], 0); }
                for (long i = 0; i < bx.npts; i++) {
                    unsigned char f = fl[i];
                    ns += (f == 1); nx += (f == 2);
                    if (f != 2) for (int k = 0; k < 3; k++) if (bx.nbin[k]) { pv[k][bx.bin[k][i]]++; ps[k][bx.bin[k][i]] += (f == 1); }
                }
                long nv = bx.npts - nx;
                double phi = nv ? (double)ns / nv : NAN;
                double sg = nv ? sqrt(phi * (1 - phi) / nv) : NAN;
                fprintf(bx.f_main, "%ld %ld %.6f %.6f %.6f %.6f\n", frame, istep, t, phi, sg, (double)nx / bx.npts);
                bx.phi_hist.push_back(phi);
                for (int k = 0; k < 3; k++) if (bx.nbin[k]) {
                    fprintf(bx.f_prof[k], "%.6f", t);
                    for (int j = 0; j < bx.nbin[k]; j++)
                        fprintf(bx.f_prof[k], pv[k][j] ? " %.6f" : " NaN", pv[k][j] ? (double)ps[k][j] / pv[k][j] : 0.0);
                    fprintf(bx.f_prof[k], "\n");
                }
            } else {
                size_t nsb = bx.subs.size();
                std::vector<long> ss(nsb, 0), sv(nsb, 0);
                for (long i = 0; i < bx.npts; i++) {
                    unsigned char f = fl[i]; if (f == 2) continue;
                    uint16_t m = bx.submask[i];
                    while (m) { int s = __builtin_ctz(m); m &= m - 1; sv[s]++; ss[s] += (f == 1); }
                }
                fprintf(bx.f_sens, "%.6f", t);
                for (size_t s = 0; s < nsb; s++) fprintf(bx.f_sens, sv[s] ? " %.6f" : " NaN", sv[s] ? (double)ss[s] / sv[s] : 0.0);
                fprintf(bx.f_sens, "\n");
                bx.phi_hist.push_back(sv[0] ? (double)ss[0] / sv[0] : NAN);
            }
        }
        times.push_back(t);
        analysed++;
        if (analysed % 25 == 0) {
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            printf("frame %ld  t = %.3f s  phi[%s] = %.5f  (%.2f s/frame, %.1f GB read)\n", frame, t,
                   boxes[0].name.c_str(), boxes[0].phi_hist.back(), el / analysed, (rd.consumed + rd.pos) / 1e9);
            fflush(stdout);
        }
        if (max_frames > 0 && analysed >= max_frames) break;
    }
    fclose(fin);
    for (auto &bx : boxes) {
        if (bx.f_main) fclose(bx.f_main);
        if (bx.f_sens) fclose(bx.f_sens);
        for (int k = 0; k < 3; k++) if (bx.f_prof[k]) fclose(bx.f_prof[k]);
    }
    fclose(f_frames);
    if (analysed == 0) { fprintf(stderr, "no frame analysed\n"); return 1; }

    // ---- summary: mean of the first / last avg_n analysed frames ----
    FILE *fs = fopen((outdir + "/MC_summary.txt").c_str(), "w");
    long na = std::min(avg_n, analysed);
    fprintf(fs, "# %ld frames analysed, t = %.4f .. %.4f s; averages over the first/last %ld frames\n",
            analysed, times.front(), times.back(), na);
    fprintf(fs, "# box phi_first phi_first_avg phi_last phi_last_avg delta_avg\n");
    printf("\n%-12s %9s %9s %9s %9s %9s\n", "box", "first", "first_avg", "last", "last_avg", "delta");
    for (auto &bx : boxes) {
        auto &v = bx.phi_hist;
        double a0 = 0, a1 = 0;
        for (long i = 0; i < na; i++) { a0 += v[i]; a1 += v[v.size() - 1 - i]; }
        a0 /= na; a1 /= na;
        std::string nm = bx.is_sens ? bx.name + "(base)" : bx.name;
        fprintf(fs, "%s %.6f %.6f %.6f %.6f %.6f\n", nm.c_str(), v.front(), a0, v.back(), a1, a1 - a0);
        printf("%-12s %9.5f %9.5f %9.5f %9.5f %+9.5f\n", nm.c_str(), v.front(), a0, v.back(), a1, a1 - a0);
    }
    fclose(fs);
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\n%ld frames in %.1f s -> %s\nMonte Carlo complete\n", analysed, el, outdir.c_str());
#ifdef __CUDACC__
    if (use_gpu) { cudaFree(d_pts); cudaFree(d_sph); cudaFree(d_type); cudaFree(d_start); cudaFree(d_flag); }
#endif
    return 0;
}

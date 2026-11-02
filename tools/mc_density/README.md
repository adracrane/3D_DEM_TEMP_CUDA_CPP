# mc_density

Monte Carlo solid fraction (Phi) of the ballast, computed from `out2.dat` in one streaming pass.

```bash
make                    # GPU build (or: make cpu)
./mc_density path/to/out2.dat            # built-in boxes, results in path/to/mc/
./mc_density out2.dat --boxes my_boxes.txt --outdir mc_test --max_frames 5
```

The time axis is `t = (iloop + istep) * dt`, with `iloop` and `dt` read from `in_bc.dat` and `in_gm.dat` in the same folder as `out2.dat`. Use `--iloop` and `--dt` to override them.

## Point classes
- **solid**: the point is inside a ballast sphere (spheres 1..`--ballast`, default 46711).
- **excluded**: the point is inside a tool sphere (`--tool`, default 54122:57136) when `mode>=1`, or inside a container or step sphere when `mode=2`. Excluded points are removed from the volume: `Phi = solid / (N - excluded)`.

All frames use the same points (seeded per box). This makes changes over time more precise than sigma = sqrt(Phi(1-Phi)/N).

## Box file
```
# name     x0  x1  y0  y1  z0  z1  npts     mode slice axes
box zone   400 530  70 530  50 250 1000000  0    10    xyz
sens zone 25 4000000      # zone with each bound moved +-25 mm (15 variants)
```
The built-in set (written to `mc/mc_boxes_used.txt`) has these boxes:
- `zone`: the under-sleeper zone.
- `valid`: the lab validation box.
- `global`: 600x600x200 with the tool excluded.
- `profx`, `profy`, `profz`: 10 mm slice profiles, with tool and container excluded.
- `zone_sens`: the ±25 mm sensitivity variants of `zone`.

## Outputs (in `mc/`)
| file | content |
|---|---|
| `MC_<box>.txt` | frame, istep, time, Phi, sigma, excluded fraction |
| `MC_<box>_prof_<x/y/z>.txt` | Phi per slice; row 1 holds the slice centres |
| `MC_zone_sens.txt` | Phi for each bound variant |
| `MC_frames.txt` | tool tip z and tool x extent per frame |
| `MC_summary.txt` | first and last Phi, each averaged over `--avg_n` frames (default 10 = 0.2 s) |

A truncated last frame (run still going, or a crash) is skipped with a message.

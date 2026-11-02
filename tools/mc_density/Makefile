# mc_density — Monte Carlo solid fraction from out2.dat
#   make            GPU build (needs nvcc)
#   make cpu        CPU build with OpenMP (no CUDA needed, ~50x slower)
NVCC ?= nvcc
ARCH ?= native
CXX  ?= g++

mc_density: mc_density.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -o $@ $<

cpu: mc_density_cpu
mc_density_cpu: mc_density.cu
	$(CXX) -O3 -std=c++17 -fopenmp -x c++ -o $@ $<

clean:
	rm -f mc_density mc_density_cpu
.PHONY: cpu clean

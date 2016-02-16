# what flags you want to pass to the C compiler & linker
CXX = mpicxx
CUDACC = nvcc
CUFLAGS = -arch=sm_20 -Xptxas -v -ccbin mpiCC

# this lists the binaries to produce, the (non-PHONY, binary) targets in
# the previous manual Makefile
SRC = analyze.cpp arrayMPI.tcc compare.cpp dataMPI.tcc decompose.cpp decomposition.cpp gpu_computing_cap.cpp gpu_computing.cpp image.cpp list_repeats.cpp main2.cpp main.cpp matrix_gomology.cpp matrixMPI.tcc myMPI_cap.cpp myMPI.cpp options.cpp profile.cpp profiling.cpp sequence.cpp support.cpp time_measure.cpp

if HAVE_USE_CUDA
SRC += gpu_computing_compare2.cu gpu_computing_compare.cu gpu_computing_decompose.cu
endif

bin_PROGRAMS = mpisbars
mpisbars_SOURCES = $(SRC)

.cu.o:
	$(CUDACC) $(CUFLAGS) -c $(CFLAGS) $< -DUSE_MPI=1 -DUSE_CUDA=1 -o $@

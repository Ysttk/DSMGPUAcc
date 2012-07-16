CC = nvcc -lcuda -g -pg -arch sm_13 -Xcompiler "-B /opt/binutils2.21/bin"

 

all: ompr_dsm.o ompr_utils.o ompr_heap.o ompr_prefetch.o

ompr_dsm.o: ompr_dsm.cu  omp_runtime.h ompr_utils.h ompr_heap.h ompr_dsm.h ompr_prefetch.h
	$(CC) -c ompr_dsm.cu

ompr_utils.o: ompr_utils.cu
	$(CC) -c $<

ompr_heap.o: ompr_heap.cu
	$(CC) -c $<
ompr_prefetch.o: ompr_prefetch.cu
	$(CC) -c $<

ompr_computing_patition.o:


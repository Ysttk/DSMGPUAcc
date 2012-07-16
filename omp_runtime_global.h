#ifndef OMP_RUNTIME_GLOBAL_H
#define OMP_RUNTIME_GLOBAL_H

#include <stdint.h>

extern "C" {

#define DSM_MAX_DIMENSION 5

typedef int32_t HRESULT;
typedef HRESULT OMPRresult;

#define SUCCESS 0
#define FAILD_UNKNOWN  -1
#define NOT_GPU_THREAD -2
#define INVALID_ACCESS_TYPE -3
#define ALREADY_INITED  -4
#define LAST_ERROR -5

#define GPU_NUM 2
#define CPU_NUM 4 //including gpu threads


struct ArrayTranspose{
    int one;
    int other;
};


typedef struct {
  enum { RONLY, WONLY, RW} AccessType;
  ArrayTranspose arrayTranspose;
} Operation;

typedef struct tagDataInfo {
  uint64_t base;
  int lens[DSM_MAX_DIMENSION];
  int dimsOffset[DSM_MAX_DIMENSION];
} DataInfo;

typedef void(*CallKernelFuncType)(pid_t,cudaStream_t&) ;

typedef enum {
  FLOAT, DOUBLE, INT32, INT64, INT16, INT8
} ElementType;


OMPRresult omprDSMGlobalInit(int n);
OMPRresult omprInit(pid_t ompThreadId);
OMPRresult omprEndFor(pid_t ompThreadId);

OMPRresult omprSplit(pid_t ompThreadId,void* base,int dims, ...);

OMPRresult omprCPUInput(DataInfo* pInfo, pid_t threadId, Operation& opr,
    void* base, ElementType type, int dims, ...);
OMPRresult omprCPUOutput(pid_t threadId, Operation& opr, 
    void* base, ElementType type, int dims, ...);

OMPRresult omprGPUInput(DataInfo* pInfo,pid_t threadId, int streamId, 
    Operation& opr, void* base, ElementType type, int dims, ...);
OMPRresult omprGPUOutput(pid_t threadId, int streamId, Operation& opr, 
    void* base, ElementType type, int dims, ...);

OMPRresult omprGPUWriteBegin(pid_t threadId, int streamId, void* base,
    ElementType eType, int dims, ...);

OMPRresult omprCudaKernelCall(pid_t threadId, int streamId, 
    CallKernelFuncType func);

OMPRresult omprGPUDataUseless(pid_t threadId,  void* base, ElementType type, 
    int dims, ...);


OMPRresult omprGPUSynchronize(pid_t threadId);

OMPRresult omprGPUSynchronizeA(pid_t threadId, void* base, ElementType type,
    int dims, ...);

OMPRresult prop_barrier_match(pid_t threadId);

//prefetch worker thread must be stop after agent exit
OMPRresult stopPrefetchWorkerThreads();
}



#endif



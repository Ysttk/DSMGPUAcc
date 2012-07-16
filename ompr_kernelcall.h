#ifndef OMPR_KERNELCALL_H
#define OMPR_KERNELCALL_H
#include "ompr_dsm.h"

extern CUDAInfo Cudas[GPU_NUM]; //ompr_dsm.cu
extern HaloBlockListType FoldHaloBlocks;  //ompr_dsm.cu
extern cudaStream_t Streams[GPU_NUM*2][DSM_MAX_STREAMS]; //ompr_dsm.cu


template <class ArgType>
OMPRresult omprCudaKernelCall(pid_t threadId, int streamId, 
    void (*func)(pid_t, cudaStream_t&, ArgType*), ArgType* arg) {

  assert(threadId==GPU1_THREAD || threadId==GPU2_THREAD);

  Cudas[threadId].LockAndPushContext();

  if (Streams[threadId][streamId] == INVALID_STREAM) 
    CudaSafe(cudaStreamCreate(&Streams[threadId][streamId]));

  func(threadId, Streams[threadId][streamId], arg);
  Cudas[threadId].ReleaseAndPopContext();

  return SUCCESS;
}

#endif

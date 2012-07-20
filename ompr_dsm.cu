#include <map>
#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <algorithm>
#include "ompr_dsm.h"
#include <stdarg.h>
#include <cuda.h>
#include <pthread.h>
#include <error.h>
#include "ompr_utils.h"
#include "ompr_heap.h"
#include <cstdio>
#include <unistd.h>

#include "ompr_prefetch.h"

/**************global declaration*******/

#define  OtherGPUThd(x) ((Device_Thread)(((x)+1)%2))

CUDAInfo Cudas[GPU_NUM];

pthread_mutex_t modifyMap;
pthread_mutex_t cudaFuncLock;
pthread_mutex_t OmprInitLock[2];
pthread_rwlock_t ModifyDataState;
DataStateType DataState;
HaloBlockListType FoldHaloBlocks; //these halo region havn't found block 
//contain valid value in dsm menu

PortableMemoryManager* PorMemManager;

DeviceMemoryManager* DevMemManager[GPU_NUM];

CPUMemoryManager* CPUMemManager;

//int BlockSharp[] = {32, 32, 32, 32, 32};  //default block shape for all arrays

bool initialized[DSM_MAX_PTHREADS]; //if true, don't need to call cuInit() again.


#ifdef PREFETCH
//prefetch 
DSMDFTraceManager traceMgr;
#endif


cudaStream_t Streams[GPU_NUM*2][DSM_MAX_STREAMS];

pthread_barrier_t initBarrier;

//ok, yes, I'm lazy...
#define DSM_FUNC_INIT(x) \
  va_list args;           \
va_start(args,dims);    \
for (int i=0; i<dims; i++) {  \
  lb[i] = va_arg(args,int);   \
  ub[i] = va_arg(args,int);   \
  shape[i] = ub[i]-lb[i]+1;	\
  loff[i] = va_arg(args, int); \
  uoff[i] = va_arg(args, int); \
  len[i] = va_arg(args,int);  \
  loff[i] = loff[i]>lb[i]? lb[i] : loff[i]; \
  uoff[i] = (uoff[i]+ub[i])>=len[i]? len[i]-ub[i]-1 : uoff[i]; \
}                             \

void checkCUDAError(const char *msg);
HRESULT AdjustDataBlockByHaloRegion(Device_Thread thread, BlockOnGPUState& state, 
    int* lb, int* ub, int* loff, int* uoff, int* shape, int dim, 
    ElementType type, DataObjState* dataState);

HRESULT UpdateHaloRegion(int threadId,BlockOnGPUState& gpuState,
    DataObjState& blockState);

HRESULT UpdateGPUBlock(cudaStream_t& stream, DataObjState* state,
    BlockOnGPUState& destGPU, volatile State& cpuState, BlockOnGPUState& srcGPU,
    Device_Thread srcThreadId, void* cpuArrayBase);

/*********end of global declaration*******/

/*********class implementation************/

bool operator < (const HaloBlock& b1, const HaloBlock& b2) {
  if (b1.base < b2.base) return true;
  if (b1.base > b2.base) return false;

  for (int i=0; i<DSM_MAX_DIMENSION; i++) {
    if (b1.lb[i] < b2.lb[i]) return true;
    if (b1.ub[i] < b2.ub[i]) return true;
  }
  return false;
}

bool operator < (const BlockKey& k1, const BlockKey& k2) {
  if (k1.base < k2.base) return true;
  else if (k1.base > k2.base) return false;

  for (int i=0; i<DSM_MAX_DIMENSION; i++){
    if (k1.lb[i] < k2.lb[i]) return true;
    if (k1.lb[i] > k2.lb[i]) return false;
  }

  for (int i=0; i<DSM_MAX_DIMENSION; i++) {
    if (k1.ub[i] < k2.ub[i]) return true;
    if (k1.ub[i] > k2.ub[i]) return false;
  }
  //if (k1.lb[i] > k2.lb[i] || k2.ub[i] < k1.ub[i]) return false;
  return false;
}

bool in(IRectangleBlock* large,IRectangleBlock* small,int dims) {
  for (int i=0; i<dims; i++)
    if (small->LB(i) < large->LB(i) ||
        small->UB(i) > large->UB(i))
      return false;
  return true;
}

HRESULT BlockOnGPUState::UnRegiste(HaloState* state) {
  for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
      itr!=haloRegionObserverList->end(); itr++)
    if (*itr == state) {
      haloRegionObserverList->erase(itr);
      break;
    }
  return SUCCESS;
}

HRESULT BlockOnGPUState::Registe(HaloState* state) {
  haloRegionObserverList->push_back(state);
  return SUCCESS;
}

HRESULT DataObjState::UnRegiste(HaloState* state) {
  for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
      itr!=haloRegionObserverList->end(); itr++)
    if (*itr == state) {
      haloRegionObserverList->erase(itr);
      break;
    }
  return SUCCESS;
}

HRESULT DataObjState::Registe(HaloState* state) {
  haloRegionObserverList->push_back(state);
  return SUCCESS;
}

OMPRresult BlockOnGPUState::getDataObjOnGPU(Device_Thread threadId, Operation& opr, 
    int loff[], int uoff[], int shape[], int dims, DataInfo* pInfo) 
{

  long int totalSize=1;  //the size of the region including the shadow region
  long int offset=0;	//offset between Addr and addr on gpu
  for (int i=dims-1; i>=0; i--) {
    totalSize *= shape[i]+loff[i]+uoff[i];
    offset = offset*(shape[i]+loff[i]+uoff[i])+loff[i];
  }

  if (state == SInvalid && Addr == NULL) {
    GPUAlloc(threadId, &Addr, totalSize*ElementSize(dataObjRef->eleType));
    addr = (char*)Addr + offset*ElementSize(dataObjRef->eleType);
  }

  int lb[DSM_MAX_DIMENSION], ub[DSM_MAX_DIMENSION];
  for (int i=0; i<DSM_MAX_DIMENSION; i++) {
    lb[i] = dataObjRef->dimsOffset[i];
    ub[i] = lb[i]+ dataObjRef->shape[i]+1;
  }
  AdjustDataBlockByHaloRegion(threadId, *this, lb, ub, loff, uoff, 
      dataObjRef->shape, dataObjRef->dims, dataObjRef->eleType, dataObjRef);

  UpdateHaloRegion(threadId, *this, *dataObjRef);

  OMPRresult result = (this ->* getDataFuncList[state])(threadId, opr);

  pInfo->base = (uint64_t)Addr;

  for (int i=0; i<dataObjRef->dims; i++)
    pInfo->lens[i] = dataObjRef->shape[i] + loff[i] + uoff[i];

  memcpy(pInfo->dimsOffset, loff, sizeof(int)*DSM_MAX_DIMENSION);

  return result;
}

OMPRresult BlockOnGPUState::getDataInvalid(Device_Thread threadId, Operation& opr)
{
  UpdateGPUBlock(stream, dataObjRef, *this, dataObjRef->cpuState,
      *opposeDeviceRef, (Device_Thread)OtherGPUThd(threadId), dataObjRef->base);

  dataObjRef->WaitGPU(threadId);

  if (opr.AccessType == Operation::RONLY) {
    Cudas[threadId].LockAndPushContext();
    CudaSafe(cudaEventCreate(&validEvent));
    CudaSafe(cudaEventRecord(validEvent, stream));
    Cudas[threadId].ReleaseAndPopContext();
  }
  if (opr.AccessType == Operation::RW)  {
    dataObjRef->OldCPU();
    if (opposeDeviceRef->state != SInvalid) 
      dataObjRef->OldGPU(OtherGPUThd(threadId));
  }

  return SUCCESS;
}

OMPRresult BlockOnGPUState::getDataValid(Device_Thread threadId, Operation& opr)
{
  return SUCCESS;
}

OMPRresult BlockOnGPUState::getDataOld(Device_Thread threadId, Operation& opr)
{
  return SUCCESS;
}

inline void BlockOnGPUState::FreeSpace(Device_Thread threadId) {
  if (Addr!=NULL) {
    //Cudas[threadId].LockAndPushContext();
    DevMemManager[threadId]->freeBlock(Addr);
    Addr = NULL;
    //Cudas[threadId].ReleaseAndPopContext();
  }
}

OMPRresult BlockOnGPUState::getDataWaiting(Device_Thread threadId, Operation& opr) 
{
  Cudas[threadId].LockAndPushContext();
  CudaSafe(cudaEventSynchronize(validEvent));
  //CudaSafe(cudaEventDestroy(validEvent));
  Cudas[threadId].ReleaseAndPopContext();

  dataObjRef->ValidGPU(threadId);
  validEvent = INVALID_EVENT;
  if (dataObjRef->cpuState == SOld) dataObjRef->InvalidCPU();
  if (opposeDeviceRef->state == SOld) 
    dataObjRef->InvalidGPU(OtherGPUThd(threadId));

  return SUCCESS;
}

OMPRresult DataObjState::getDataObjOnCPU(Device_Thread threadId, Operation& opr, 
    int loff[], int uoff[], int shape[], int dims, DataInfo* pInfo) {
  return SUCCESS;
}


OMPRresult DataObjState::AddCoverBlock(int lb[], int ub[]) {
  BlankBlock blackBlock(lb,ub);
  BlankBlock tmpBlock;
  std::vector<BlankBlock*>* newBlankBlockList= new std::vector<BlankBlock*>();

  pthread_mutex_lock(&(ModifyBlankBlockList));
  for (std::vector<BlankBlock*>::iterator itr= blankBlockList->begin();
      itr!= blankBlockList->end(); ++itr) {
    //if(*itr == NULL) continue;
    if (RecContain<BlankBlock>(*itr, &blackBlock, dims)==CONTAINED) {
      continue;
    } else if (IsIntersect(*itr, &blackBlock, dims)==SUCCESS) {
      std::vector<BlankBlock*> splitList;
      IntersectAndSplitRec(*itr, &blackBlock, dims, &tmpBlock, &splitList);
      for (std::vector<BlankBlock*>::iterator tmpItr = splitList.begin(); 
          tmpItr != splitList.end(); ++tmpItr) {
        newBlankBlockList->push_back(*tmpItr);
      }
      splitList.clear();
    } else 
      newBlankBlockList->push_back(*itr);
  }

  blankBlockList->clear();
  delete blankBlockList;
  blankBlockList = newBlankBlockList;
  pthread_mutex_unlock(&(ModifyBlankBlockList));

  return SUCCESS;
}

OMPRresult DataObjState::AddRedistributionBlock(int lb[], int ub[]) {
  return AddCoverBlock(lb,ub);
}

OMPRresult DataObjState::AddOverrideBlock(int lb[], int ub[]) {
  return AddCoverBlock(lb, ub);
}

/*********end of class implimentation****/

__global__ void copy_kernel4(double* dest, double* src, int num,
    int cDLen0, int cDLen1, int cDLen2, int cDLen3,
    int cSLen0, int cSLen1, int cSLen2, int cSLen3,
    int cDOffset0, int cDOffset1, int cDOffset2, int cDOffset3,
    int cSOffset0, int cSOffset1, int cSOffset2, int cSOffset3,
    int cshape0, int cshape1, int cshape2, int cshape3) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  tid *= 2;
  for (int itr = tid; itr<num; itr+=gridDim.x*blockDim.x*2) {
    int idx = itr/*+startOff*/;
    int dimIdx0 = idx % cshape0;
    idx /= cshape0;
    int dimIdx1 = idx % cshape1;
    idx /= cshape1;
    int dimIdx2 = idx % cshape2;
    idx /= cshape2;
    int dimIdx3 = idx;

    int srcIdx = dimIdx3+cSOffset3;
    int destIdx = dimIdx3+cDOffset3;
    srcIdx = srcIdx*cSLen2+ dimIdx2+cSOffset2;
    destIdx = destIdx*cDLen2 + dimIdx2 + cDOffset2;
    srcIdx = srcIdx*cSLen1+ dimIdx1+cSOffset1;
    destIdx = destIdx*cDLen1 + dimIdx1 + cDOffset1;
    srcIdx = srcIdx*cSLen0 + dimIdx0 + cSOffset0;
    destIdx = destIdx*cDLen0 + dimIdx0 + cDOffset0;

    dest[destIdx] = src[srcIdx];
    if (itr+1 < num) 
      dest[destIdx+1] = src[srcIdx+1];
  }

}

__global__ void copy_kernel3(double* dest, double* src, int num, 
    int cDLen0, int cDLen1, int cDLen2,
    int cSLen0, int cSLen1, int cSLen2,
    int cDOffset0, int cDOffset1, int cDOffset2,
    int cSOffset0, int cSOffset1, int cSOffset2,
    int cshape0, int cshape1, int cshape2) {


  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  idx *= 2;

  for (int itr = idx; itr<num; itr+=blockDim.x*gridDim.x*2) {

    int dimIdx0 = idx % cshape0;
    idx /= cshape0;
    int dimIdx1 = idx % cshape1;
    idx /= cshape1;
    int dimIdx2 = idx;

    int srcIdx = dimIdx2+cSOffset2;
    int destIdx = dimIdx2+cDOffset2;
    srcIdx = srcIdx*cSLen1+ dimIdx1+cSOffset1;
    destIdx = destIdx*cDLen1 + dimIdx1 + cDOffset1;
    srcIdx = srcIdx*cSLen0 + dimIdx0 + cSOffset0;
    destIdx = destIdx*cDLen0 + dimIdx0 + cDOffset0;

    dest[destIdx] = src[srcIdx];
    if (itr+1 < num)
      dest[destIdx+1] = src[srcIdx+1];
  }

}


void KernelCopyAsync(void* dest, int* dLen, int* dOffset,
    void* src, int* sLen, int* sOffset,
    int* shape, int dim, ElementType type, cudaStream_t &stream) {
  //cudaThreadSynchronize();
  static double atime =0.0;
  //double start = gettime();
  int size=1;
  for (int i=0; i<dim; i++)
    size *= shape[i];
  size*=8; size/=8;  //every thread copy eight byte 

  int reShape[DSM_MAX_DIMENSION];
  int reDLen[DSM_MAX_DIMENSION], reSLen[DSM_MAX_DIMENSION],
      reDOffset[DSM_MAX_DIMENSION], reSOffset[DSM_MAX_DIMENSION];
  for (int i=0; i<dim; i++) {
    reShape[i]=shape[i]*4/4;
    reDLen[i] = dLen[i]*4/4;
    reDOffset[i] = dOffset[i]*4/4;
    reSLen[i] = sLen[i]*4/4;
    reSOffset[i] = sOffset[i]*4/4;
  }

  int thrdPerBlock = size>256?256:size;
  int blockN = size/256/2+1;
  if (blockN > 65535) blockN=65535;
  int residue = size%256;

  if (dim==3) {
    if (blockN > 0)
      copy_kernel3<<<blockN, thrdPerBlock, 0, stream>>>((double*)dest, (double*)src, size,
          reDLen[0], reDLen[1], reDLen[2],
          reSLen[0], reSLen[1], reSLen[2],
          reDOffset[0], reDOffset[1], reDOffset[2],
          reSOffset[0], reSOffset[1], reSOffset[2],
          reShape[0], reShape[1], reShape[2]);
  } else if (dim==4) {
    if (blockN > 0) {
      copy_kernel4<<<blockN, thrdPerBlock, 0, stream>>>((double*)dest, (double*)src, size,
          reDLen[0], reDLen[1], reDLen[2], reDLen[3],
          reSLen[0], reSLen[1], reSLen[2], reSLen[3],
          reDOffset[0], reDOffset[1], reDOffset[2], reDOffset[3],
          reSOffset[0], reSOffset[1], reSOffset[2], reSOffset[3],
          reShape[0], reShape[1], reShape[2], reShape[3]);
      //cudaStreamSynchronize(stream);
    }
    //checkCUDAError("");
  } else {
    assert(0);
  }
}

HRESULT KernelCopy(void* dest, int* dLen, int* dOffset,
    void* src, int* sLen, int* sOffset,
    int* shape, int dim, ElementType type) {
  int size=1;
  for (int i=0; i<dim; i++)
    size *= shape[i];
  size*=ElementSize(type); size/=8;  //every thread copy eight byte, a double 

  int reShape[DSM_MAX_DIMENSION];
  int reDLen[DSM_MAX_DIMENSION], reSLen[DSM_MAX_DIMENSION],
      reDOffset[DSM_MAX_DIMENSION], reSOffset[DSM_MAX_DIMENSION];
  for (int i=0; i<dim; i++) {
    reShape[i]=shape[i];
    reDLen[i] = dLen[i];
    reDOffset[i] = dOffset[i];
    reSLen[i] = sLen[i];
    reSOffset[i] = sOffset[i];
  }

  reShape[0] = shape[0]*ElementSize(type)/8;
  reDLen[0] = dLen[0]*ElementSize(type)/8;
  reDOffset[0] = dOffset[0]*ElementSize(type)/8;
  reSLen[0] = sLen[0]*ElementSize(type)/8;
  reSOffset[0] = sOffset[0]*ElementSize(type)/8;

  int thrdPerBlock = size>256?256:size;
  int blockN = size/256;
  if(blockN > 65535) blockN = 65535;
  int residue = size%256;

  if(dim == 4){
    if(blockN > 0) {
      copy_kernel4<<<blockN, thrdPerBlock>>>((double*)dest, (double*)src, size,
          reDLen[0], reDLen[1], reDLen[2], reDLen[3],
          reSLen[0], reSLen[1], reSLen[2], reSLen[3],
          reDOffset[0], reDOffset[1], reDOffset[2], reDOffset[3],
          reSOffset[0], reSOffset[1], reSOffset[2], reSOffset[3],
          reShape[0], reShape[1], reShape[2], reShape[3]);
    }
    return 0;

  } else if (dim==3) {
    if (blockN > 0) {
      copy_kernel3<<<blockN, thrdPerBlock>>>((double*)dest, (double*)src, size,
          reDLen[0], reDLen[1], reDLen[2], 
          reSLen[0], reSLen[1], reSLen[2], 
          reDOffset[0], reDOffset[1], reDOffset[2],
          reSOffset[0], reSOffset[1], reSOffset[2],
          reShape[0], reShape[1], reShape[2]);
    }
  }


}


OMPRresult omprDSMGlobalInit(int n) {

  int re= pthread_barrier_init(&initBarrier, NULL, n);
  re = pthread_mutex_init(&modifyMap, NULL);
  re = pthread_mutex_init(&cudaFuncLock, NULL);
  re = pthread_rwlock_init(&ModifyDataState, NULL);

  re = pthread_mutex_init(&OmprInitLock[0], NULL);
  re = pthread_mutex_init(&OmprInitLock[1], NULL);

  return SUCCESS;
}

OMPRresult stopPrefetchWorkerThreads()
{
#ifdef PREFETCH
  traceMgr.stopWorkerThreads();
#endif
  return SUCCESS;
}

bool enter = 0;

OMPRresult omprInit(pid_t ompThreadId) {
  if (!initialized[ompThreadId]) {
    cuInit(0);
    initialized[ompThreadId] = true;
  }
  //double check
  if(!enter){ 
    pthread_mutex_lock(&(OmprInitLock[ompThreadId]));
    if(!enter){
      if ((ompThreadId==GPU1_THREAD )
          || (ompThreadId==GPU2_THREAD )) {
        int flags =0;
        CUDAInfo& info = Cudas[ompThreadId];
        if (info.device == 0 && info.context ==0) {
          CuSafe(cuDeviceGet(&info.device,ompThreadId));
          CuSafe(cuCtxCreate(&info.context, flags, info.device));
          CuSafe(cuCtxPopCurrent(&info.context));
          pthread_mutex_init(&info.contextLock, NULL);
        }
      }

      if (ompThreadId == GPU1_THREAD) {
        PorMemManager = new PortableMemoryManager();
        CPUMemManager = new CPUMemoryManager();
      }
      if (ompThreadId == GPU1_THREAD || 
          ompThreadId == GPU2_THREAD)
        DevMemManager[ompThreadId] = new DeviceMemoryManager((Device_Thread)ompThreadId);
      int re = pthread_barrier_wait(&initBarrier);
      enter = 1;
    }
    pthread_mutex_unlock(&OmprInitLock[ompThreadId]);
  }

  return SUCCESS;
}

OMPRresult omprDSMGlobalFinalize() {
  delete PorMemManager;
  delete DevMemManager[0];
  delete DevMemManager[1];
  delete CPUMemManager;
  return SUCCESS;
}

OMPRresult omprEndFor(pid_t ompThreadId) {
  //PrefetchEndFor(ompThreadId);

  return SUCCESS;
}


/*
Argus:
async:  synchronized or asynchronized copy;
stream: used when copy asynchronized;
dims: dims of the block;
dimsLen:  length of each dim of the array on cpu
gpuDimsLen: length of each dim of the whole block on gpu;
shape:	the shape of the block need to be copy
cpuDimsOffset:  offset of the first element of each dim of the array on cpu;
gpuDimsOffset:  offset of the first element of each dim of the 
whole block on gpu
src:  base address of the array/block;
dest: same as above;
eleType:  type of element of the array;
kind: memcpy kind;
 */
HRESULT TransferData(BOOL async, cudaStream_t& stream, int dims,
    int* dimsLen,int* gpuDimsLen, int* shape,int* cpuDimsOffset,
    int* gpuDimsOffset, void* src, void* dest, ElementType eleType, 
    cudaMemcpyKind kind) {

  int sizeUnit=1;
  int hContinuousDim=0;	//the highest continuous dim + 1
  while (shape[hContinuousDim]==dimsLen[hContinuousDim] &&
      shape[hContinuousDim]==gpuDimsLen[hContinuousDim])
    hContinuousDim++;
  if (hContinuousDim == dims-1 ||
      hContinuousDim == dims) {
    int size=1;
    for (int i=0; i<dims; i++) {
      size*=shape[i];
    }
    long cpuOffset=cpuDimsOffset[dims-1],
         gpuOffset=gpuDimsOffset[dims-1];
    for (int i=dims-1; i>0; i--) {
      cpuOffset = cpuOffset*dimsLen[i-1]+ cpuDimsOffset[i-1];
      gpuOffset = gpuOffset*gpuDimsLen[i-1]+ gpuDimsOffset[i-1];
    }

    if (kind == cudaMemcpyDeviceToHost) {
      if (async == TRUE) {
        CudaSafe(cudaMemcpyAsync((char*)dest+cpuOffset*ElementSize(eleType), 
              (char*)src+gpuOffset*ElementSize(eleType),
              ElementSize(eleType)*size, kind, stream));
      } else {
        CudaSafe(cudaMemcpy((char*)dest+cpuOffset*ElementSize(eleType), 
              (char*)src+gpuOffset*ElementSize(eleType),
              ElementSize(eleType)*size, kind));
      }
    } else if (kind==cudaMemcpyHostToDevice) {
      if (async == TRUE) {
        //checkCUDAError("cpy");
        //printf("cpy bre no error\n");
        CudaSafe(cudaMemcpyAsync((char*)dest+gpuOffset*ElementSize(eleType), 
              (char*)src+cpuOffset*ElementSize(eleType),
              ElementSize(eleType)*size, kind, stream));
      } else {
        CudaSafe(cudaMemcpy((char*)dest+gpuOffset*ElementSize(eleType), 
              (char*)src+cpuOffset*ElementSize(eleType),
              ElementSize(eleType)*size, kind));
      }
    }

    return SUCCESS;
  }

  for (int i=0; i<=hContinuousDim; i++) sizeUnit *= shape[i];

  int itrDims = dims-hContinuousDim-1;

  int tmp[16];
  int offset=0;
  std::stack<int*>* spaceStack = new std::stack<int*>();
  std::stack<int*>& space = *spaceStack;
  space.push(tmp);
  for (int i=0; i<itrDims; i++) tmp[i] =0;
  offset += itrDims;
  while (!space.empty()) {
    int* offBase = space.top();
    space.pop();
    for (int i=0; i<itrDims; i++)
      if (offBase[i] < shape[i+hContinuousDim+1]-1) {
        memcpy(&tmp[offset], offBase, sizeof(int)*
            (itrDims));
        tmp[offset+i]++;
        for (int j=0; j<i; j++) tmp[offset+j]=0;
        space.push(&tmp[offset]);
        offset += itrDims;
        if (offset > 16) offset = 0;
        break;
      }

    bool large = true;	//test wether the offset is larger then
    //it should be
    for (int i=0; i<itrDims; i++) 
      if ((i+hContinuousDim+1)!=0 && 
          offBase[i] != shape[i+hContinuousDim+1]) 
        large=false;
    if (large) continue;

    int tmpOffset[DSM_MAX_DIMENSION];
    for (int i=0; i<=hContinuousDim; i++) tmpOffset[i]=0;
    for (int i=hContinuousDim+1; i<dims; i++)
      tmpOffset[i] = offBase[i-hContinuousDim-1];

    int off = tmpOffset[dims-1]+gpuDimsOffset[dims-1], 
        cpuOff=tmpOffset[dims-1]+cpuDimsOffset[dims-1];
    //int off=0, cpuOff=0;
    for (int i=dims-1; i>0; i--) {
      off = off*gpuDimsLen[i-1] +tmpOffset[i-1]+gpuDimsOffset[i-1];

      cpuOff = cpuOff*dimsLen[i-1]+tmpOffset[i-1]+cpuDimsOffset[i-1];

    }

    if (kind == cudaMemcpyDeviceToHost) {
      if (async == TRUE) {
        CudaSafe(cudaMemcpyAsync((char*)dest+cpuOff*ElementSize(eleType), 
              (char*)src+off*ElementSize(eleType),
              ElementSize(eleType)*sizeUnit, kind, stream));
      } else {
        CudaSafe(cudaMemcpy((char*)dest+cpuOff*ElementSize(eleType), 
              (char*)src+off*ElementSize(eleType),
              ElementSize(eleType)*sizeUnit, kind));
      }
    } else if (kind==cudaMemcpyHostToDevice) {
      if (async == TRUE) {
        CudaSafe(cudaMemcpyAsync((char*)dest+off*ElementSize(eleType), 
              (char*)src+cpuOff*ElementSize(eleType),
              ElementSize(eleType)*sizeUnit, kind, stream));
      } else {
        CudaSafe(cudaMemcpy((char*)dest+off*ElementSize(eleType), 
              (char*)src+cpuOff*ElementSize(eleType),
              ElementSize(eleType)*sizeUnit, kind));
      }
    }
  }

  delete spaceStack;
  return SUCCESS;

}

HRESULT TransferOld2CPU(cudaStream_t& stream, pid_t srcThreadId, DataObjState* state, 
    BlockOnGPUState& srcState, void* dest) {  //support shadow region

  /*based on the assumption of chapter4.1
   */
  int shape[DSM_MAX_DIMENSION], dimsOffset[DSM_MAX_DIMENSION];
  int gpuLen[DSM_MAX_DIMENSION];
  for (int i=0; i<state->dims; i++) {
    shape[i] = state->shape[i] /*+ srcState.loff[i] + srcState.uoff[i]*/;
    dimsOffset[i] = state->dimsOffset[i] /*- srcState.loff[i]*/;
    gpuLen[i] = state->shape[i]+srcState.loff[i]+srcState.uoff[i];
  }

  Cudas[srcThreadId].LockAndPushContext();
  HRESULT result = TransferData(FALSE, stream, state->dims, state->dimsLen, 
      gpuLen, shape, dimsOffset, srcState.loff, srcState.Addr, dest, 
      state->eleType,cudaMemcpyDeviceToHost);
  Cudas[srcThreadId].ReleaseAndPopContext();
  //state->OldCPU();

  return result;

}

HRESULT TransferWaiting2CPU(cudaStream_t& stream, pid_t srcThreadId, DataObjState* state, 
    BlockOnGPUState& srcGPU, void* dest) {

  Cudas[srcThreadId].LockAndPushContext();
  //CudaSafe(cudaEventSynchronize(srcGPU.validEvent));
  //CudaSafe(cudaEventDestroy(srcGPU.validEvent));
  Cudas[srcThreadId].ReleaseAndPopContext();

  if (srcThreadId == GPU1_THREAD) 
    state->ValidGPU1();
  else if (srcThreadId == GPU2_THREAD)
    state->ValidGPU2();

  int shape[DSM_MAX_DIMENSION], dimsOffset[DSM_MAX_DIMENSION];
  int gpuLen[DSM_MAX_DIMENSION];
  for (int i=0; i<state->dims; i++) {
    shape[i] = state->shape[i] /*+ srcGPU.loff[i] + srcGPU.uoff[i]*/;
    dimsOffset[i] = state->dimsOffset[i] /*- srcGPU.loff[i]*/;
    gpuLen[i] = state->shape[i] + srcGPU.loff[i] + srcGPU.uoff[i];
  }

  Cudas[srcThreadId].LockAndPushContext();
  TransferData(FALSE, stream, state->dims, state->dimsLen, gpuLen,
      shape, dimsOffset, srcGPU.loff, srcGPU.Addr, dest, 
      state->eleType, cudaMemcpyDeviceToHost);  
  //must be device to host cpy
  Cudas[srcThreadId].ReleaseAndPopContext();

  state->ValidCPU();

  return SUCCESS;

}

HRESULT UpdateCPUBlock(DataObjState* state, void* cpuArrayBase) {

  if (state->cpuState == SInvalid) {
    if (state->gpu1.state == SValid) {

      int shape[DSM_MAX_DIMENSION], dimsOffset[DSM_MAX_DIMENSION];
      int gpuLen[DSM_MAX_DIMENSION];
      for (int i=0; i<state->dims; i++) {
        shape[i] = state->shape[i] /*+ state->gpu1.loff[i] + state->gpu1.uoff[i]*/;
        dimsOffset[i] = state->dimsOffset[i] /*- state->gpu1.loff[i]*/;
        gpuLen[i] = state->shape[i] + state->gpu1.loff[i] + 
          state->gpu1.uoff[i];
      }
      Cudas[GPU1_THREAD].LockAndPushContext();
      cudaStream_t tmp;
      TransferData(FALSE, tmp, state->dims, state->dimsLen, gpuLen,
          shape, dimsOffset, state->gpu1.loff, state->gpu1.Addr, 
          cpuArrayBase, state->eleType, cudaMemcpyDeviceToHost);
      Cudas[GPU1_THREAD].ReleaseAndPopContext();
      state->ValidCPU();

    } else if (state->gpu2.state == SValid) {

      int shape[DSM_MAX_DIMENSION], dimsOffset[DSM_MAX_DIMENSION];
      int gpuLen[DSM_MAX_DIMENSION];
      for (int i=0; i<state->dims; i++) {
        shape[i] = state->shape[i] /*+ state->gpu2.loff[i] + state->gpu2.uoff[i]*/;
        dimsOffset[i] = state->dimsOffset[i] /*- state->gpu2.loff[i]*/;
        gpuLen[i] = state->shape[i] + state->gpu2.loff[i] +
          state->gpu2.uoff[i];
      }
      Cudas[GPU2_THREAD].LockAndPushContext();
      cudaStream_t tmp;
      TransferData(FALSE, tmp, state->dims, state->dimsLen, gpuLen,
          shape, dimsOffset, state->gpu2.loff,state->gpu2.Addr, 
          cpuArrayBase, state->eleType, cudaMemcpyDeviceToHost);
      Cudas[GPU2_THREAD].ReleaseAndPopContext();

      state->ValidCPU();
    } else if (state->gpu1.state == SOld || 
        state->gpu2.state == SOld ) {

      Device_Thread srcThreadId;
      BlockOnGPUState* srcGPU;
      if (state->gpu1.state == SOld) {
        srcThreadId = GPU1_THREAD;
        srcGPU = &state->gpu1;
      } else {
        srcThreadId = GPU2_THREAD;
        srcGPU = &state->gpu2;
      }

      int shape[DSM_MAX_DIMENSION], dimsOffset[DSM_MAX_DIMENSION];
      int gpuLen[DSM_MAX_DIMENSION];
      for (int i=0; i<state->dims; i++) {
        shape[i] = state->shape[i] /*+ srcGPU->loff[i] + srcGPU->uoff[i]*/;
        dimsOffset[i] = state->dimsOffset[i] /*- srcGPU->loff[i]*/;
        gpuLen[i] = state->shape[i] + srcGPU->loff[i] +
          srcGPU->uoff[i];
      }
      Cudas[srcThreadId].LockAndPushContext();
      cudaStream_t tmp;
      TransferData(FALSE, tmp, state->dims, state->dimsLen, gpuLen,
          shape, dimsOffset, srcGPU->loff, srcGPU->Addr, cpuArrayBase,
          state->eleType, cudaMemcpyDeviceToHost);
      Cudas[srcThreadId].ReleaseAndPopContext();
      state->ValidCPU();

    } else if (state->gpu1.state == SWaiting ||
        state->gpu2.state == SWaiting ) {
      assert(0);
    }

    return SUCCESS;
  } 

  return SUCCESS;
}



HRESULT UpdateHaloRegionFromCPU(int threadId, BlockOnGPUState& gpuState,
    DataObjState& dataBlock, const HaloBlock& haloBlock) {

  int haloRegionShape[DSM_MAX_DIMENSION];
  int gpuDimsLen[DSM_MAX_DIMENSION];
  int haloRegionOffsetOnGPU[DSM_MAX_DIMENSION];

  for (int i=0; i<dataBlock.dims; i++) {
    haloRegionShape[i] = haloBlock.ub[i]-haloBlock.lb[i]+1;
    gpuDimsLen[i] = dataBlock.shape[i]+
      gpuState.loff[i]+gpuState.uoff[i];
    haloRegionOffsetOnGPU[i] = haloBlock.lb[i]-(dataBlock.dimsOffset[i]
        - gpuState.loff[i]);

  }

  Cudas[threadId].LockAndPushContext();
  cudaStream_t tmp;
  TransferData(FALSE, tmp, dataBlock.dims, dataBlock.dimsLen, 
      gpuDimsLen, haloRegionShape, const_cast<int*>(haloBlock.lb), 
      haloRegionOffsetOnGPU, dataBlock.base, gpuState.Addr,
      dataBlock.eleType, cudaMemcpyHostToDevice);
  Cudas[threadId].ReleaseAndPopContext();

  return SUCCESS;

}




/*
 * if state of halo region is invalid, transfer valid data and valid 
 * the halo region, else nothing to do.
 */
double upTime[2], upTime1[2], upTime2[2], upTime3[2];
HRESULT UpdateHaloRegion(int threadId,BlockOnGPUState& gpuState,
    DataObjState& blockState) {

  double start = gettime();
  for (HaloMapType::iterator itr = gpuState.haloMap->begin();
      itr!=gpuState.haloMap->end(); ++itr) {

    HaloState* s = itr->second;

    if (s->state == SInvalid) {

      if (s->validBlockKey == NULL && 
          s->subject == NULL) {
        //if the halo block has not its key, so the 
        //valid data must be at CPU
        UpdateHaloRegionFromCPU(threadId, gpuState, 
            blockState, itr->first);
        s->state = SValid;
        continue;
      }
      upTime1[threadId] += gettime()-start;
      printf("update halo 1 %lf in %d\n", upTime1[threadId], threadId);

      s->subject->UnRegiste(s);

      pthread_rwlock_rdlock(&ModifyDataState);
      DataObjState* haloBlockState = DataState.
        find(*s->validBlockKey)->second;
      pthread_rwlock_unlock(&ModifyDataState);
      if (haloBlockState == NULL) {
        printf("halo region without source block!\n");
        assert(0);
      }

      haloBlockState->WaitUntilInited();
      upTime2[threadId] += gettime()-start;
      printf("update halo 2 %lf in %d\n", upTime2[threadId], threadId);

      BlockOnGPUState* selfState, *srcGPUState;
      State cpuState=haloBlockState->cpuState;

      if (threadId == GPU1_THREAD) {
        selfState=&haloBlockState->gpu1;
        srcGPUState = &haloBlockState->gpu2;
      } else {
        selfState= &haloBlockState->gpu2;
        srcGPUState = &haloBlockState->gpu1;
      }

      //int count=0;
      bool updated=false;
      while (!updated) {  //a large spin lock ^_^
        //count++;

        double start1 = gettime();
        if (selfState->state == SValid ||
            selfState->state == SWaiting) {
          updated = true;
          if (selfState->state == SWaiting) {
            Cudas[threadId].LockAndPushContext();
            //CudaSafe(cudaEventSynchronize(selfState->validEvent));
            Cudas[threadId].ReleaseAndPopContext();
          }
          HaloBlock& haloBlock = const_cast<HaloBlock&>(itr->first);
          int shape[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++)
            shape[i] = haloBlock.ub[i] - haloBlock.lb[i]+1;
          int sLen[DSM_MAX_DIMENSION], dLen[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++) {
            sLen[i] = haloBlockState->shape[i]+selfState->loff[i]
              + selfState->uoff[i];
            dLen[i] = blockState.shape[i] + gpuState.loff[i]
              + gpuState.uoff[i];
          }

          int gpuOffset[DSM_MAX_DIMENSION],selfOffset[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++) {
            gpuOffset[i] = haloBlock.lb[i]-(blockState.dimsOffset[i]
                -gpuState.loff[i]);
            selfOffset[i] = haloBlock.lb[i]-haloBlockState->dimsOffset[i];
          }

          double tmp = gettime();
          Cudas[threadId].LockAndPushContext();
          KernelCopy(gpuState.Addr, dLen, gpuOffset,
              selfState->addr, sLen, selfOffset, 
              shape, blockState.dims, blockState.eleType);
          Cudas[threadId].ReleaseAndPopContext();
          start1 += gettime()-tmp;

          s->state = SValid;

          selfState->Registe(s);

        } else if (cpuState == SValid) {

          updated = true;
          HaloBlock& haloBlock = const_cast<HaloBlock&>(itr->first);
          int shape[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++)
            shape[i] = haloBlock.ub[i] - haloBlock.lb[i]+1;
          int dLen[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++) {
            dLen[i] = blockState.shape[i] + gpuState.loff[i]
              + gpuState.uoff[i];
          }

          int offset[DSM_MAX_DIMENSION],gpuOffset[DSM_MAX_DIMENSION];
          for (int i=0; i<haloBlockState->dims; i++) {
            offset[i] = haloBlock.lb[i];
            gpuOffset[i] = haloBlock.lb[i]-(blockState.dimsOffset[i]
                -gpuState.loff[i]);
          }

          Cudas[threadId].LockAndPushContext();
          cudaStream_t tmp;
          double tmpT = gettime();
          TransferData(FALSE, tmp, blockState.dims,
              haloBlockState->dimsLen, dLen, shape,
              offset, gpuOffset,
              haloBlockState->base, gpuState.Addr, 
              blockState.eleType, cudaMemcpyHostToDevice);
          //if (threadId==GPU2_THREAD && offset[1]==0 && offset[0]==0){ 
          //    printf("something error\n");
          //     assert(0);
          // }
          start1 += gettime()-tmpT;

          Cudas[threadId].ReleaseAndPopContext();

          s->state = SValid;
          haloBlockState->Registe(s);

        } else if (srcGPUState->state == SValid ||
            srcGPUState->state == SWaiting) {

          updated = true;
          if (srcGPUState->state == SWaiting) {
            int count=0;
            while (srcGPUState->validEvent == NULL) count++;
            printf("spin lock times:%d\n",count);
            Cudas[OtherGPUThd(threadId)].LockAndPushContext();
            //CudaSafe(cudaEventSynchronize(srcGPUState->validEvent));
            Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
          }
          HaloBlock& haloBlock = const_cast<HaloBlock&>(itr->first);
          int shape[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++)
            shape[i] = haloBlock.ub[i] - haloBlock.lb[i]+1;
          int sLen[DSM_MAX_DIMENSION], dLen[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++) {
            sLen[i] = haloBlockState->shape[i]+srcGPUState->loff[i]
              + srcGPUState->uoff[i];
            dLen[i] = blockState.shape[i] + gpuState.loff[i]
              + gpuState.uoff[i];
          }

          int srcGPUDimsOffset[DSM_MAX_DIMENSION];
          for (int i=0; i<blockState.dims; i++)
            srcGPUDimsOffset[i] = haloBlock.lb[i]-haloBlockState->dimsOffset[i]
              +srcGPUState->loff[i];

          Cudas[OtherGPUThd(threadId)].LockAndPushContext();
          cudaStream_t tmp;
          double tmpT = gettime();
          TransferData(FALSE, tmp, blockState.dims, 
              haloBlockState->dimsLen, sLen, shape,
              haloBlock.lb, srcGPUDimsOffset, 
              srcGPUState->Addr, haloBlockState->base,
              haloBlockState->eleType, cudaMemcpyDeviceToHost);
          start1 += gettime()-tmpT;
          Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();


          int offset[DSM_MAX_DIMENSION],gpuOffset[DSM_MAX_DIMENSION];
          for (int i=0; i<haloBlockState->dims; i++) {
            offset[i] = haloBlock.lb[i];
            gpuOffset[i] = haloBlock.lb[i]-(blockState.dimsOffset[i]
                -gpuState.loff[i]);
          }
          Cudas[threadId].LockAndPushContext();
          tmpT = gettime();
          TransferData(FALSE, tmp, blockState.dims,
              haloBlockState->dimsLen, dLen, shape,
              offset, gpuOffset,
              haloBlockState->base, gpuState.Addr, 
              blockState.eleType, cudaMemcpyHostToDevice);
          start1 += gettime()-tmpT;
          Cudas[threadId].ReleaseAndPopContext();

          s->state = SValid;
          srcGPUState->Registe(s);

        } else  {
          //printf("halo region without valid/waiting source block!\n");
          //assert(0);
        }

        sleep(0);
        upTime3[threadId] += gettime()-start1;
        printf("update halo 3 %lf in %d\n", upTime3[threadId], threadId);
      }
      //printf("spin count:%d\n", count);

    }
  }

  upTime[threadId] += gettime()-start;
  printf("update halo time: %lf in %d\n", upTime[threadId], threadId);
  return SUCCESS;
}

/*
   find out all halo regions decripted by haloBlock*off* and lb/ub
Arg:
dims: start with 1, dimention of the array;
haloBlockLoffL, haloBlockLoffU:
lower/upper bound of lower offset of the block;
haloBlockUOffL, haloBlockUOffU:
... of upper offset of the block;
blockL: lower bound of the data block, not halo block;
blockU: upper bound of the data block, not halo block;
 */
std::vector<HaloBlock*>* GetHaloRegions(int dims, int* haloBlockLoffL,
    int* haloBlockLoffU, int* haloBlockUoffL, int* haloBlockUoffU,
    int* blockL, int* blockU) {
  if (dims==1) {
    std::vector<HaloBlock*>* list = 
      new/*(CPUMemManager->allocBlock(sizeof(std::vector<HaloBlock*>)))*/ 
      std::vector<HaloBlock*>();
    HaloBlock* halo;
    if (haloBlockLoffL[0]<=haloBlockLoffU[0]) {
      HaloBlock* halo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
      halo->lb[0] = haloBlockLoffL[0];
      halo->ub[0] = haloBlockLoffU[0];
      list->push_back(halo);
    }
    if (haloBlockUoffL[0]<=haloBlockUoffU[0]) {
      halo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
      halo->lb[0] = haloBlockUoffL[0];
      halo->ub[0] = haloBlockUoffU[0];
      list->push_back(halo);
    }
    return list;
  } 
  std::vector<HaloBlock*>* list = GetHaloRegions(dims-1, haloBlockLoffL,
      haloBlockLoffU, haloBlockUoffL, haloBlockUoffU, blockL, blockU);

  std::vector<HaloBlock*>* tmpList =
    new/*(CPUMemManager->allocBlock(sizeof(std::vector<HaloBlock*>)))*/ std::vector<HaloBlock*>();
  for (std::vector<HaloBlock*>::iterator itr = list->begin();
      itr != list->end(); ++itr) {
    HaloBlock* halo = *itr;
    //middle halo block
    halo->lb[dims-1] = blockL[dims-1];
    halo->ub[dims-1] = blockU[dims-1];
    //lower halo block
    HaloBlock* newHalo;
    if (haloBlockLoffL[dims-1] <= haloBlockLoffU[dims-1]) {
      newHalo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
      memcpy(newHalo->lb, halo->lb, sizeof(int)*dims);
      memcpy(newHalo->ub, halo->ub, sizeof(int)*dims);
      newHalo->lb[dims-1] = haloBlockLoffL[dims-1];
      newHalo->ub[dims-1] = haloBlockLoffU[dims-1];
      tmpList->push_back(newHalo);
    }
    //upper halo block
    if (haloBlockUoffL[dims-1] <= haloBlockUoffU[dims-1]) {
      newHalo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
      memcpy(newHalo->lb, halo->lb, sizeof(int)*dims);
      memcpy(newHalo->ub, halo->ub, sizeof(int)*dims);
      newHalo->lb[dims-1] = haloBlockUoffL[dims-1];
      newHalo->ub[dims-1] = haloBlockUoffU[dims-1];
      tmpList->push_back(newHalo);
    }
  }

  HaloBlock* newHalo;

  if (haloBlockLoffL[dims-1] <= haloBlockLoffU[dims-1]) {
    newHalo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
    memcpy(newHalo->lb, blockL, sizeof(int)*dims);
    memcpy(newHalo->ub, blockU, sizeof(int)*dims);
    newHalo->lb[dims-1] = haloBlockLoffL[dims-1];
    newHalo->ub[dims-1] = haloBlockLoffU[dims-1];
    list->push_back(newHalo);
  }
  if (haloBlockUoffL[dims-1] <= haloBlockUoffU[dims-1]) {
    newHalo = new/*(CPUMemManager->allocBlock(sizeof(HaloBlock)))*/ HaloBlock();
    memcpy(newHalo->lb, blockL, sizeof(int)*dims);
    memcpy(newHalo->ub, blockU, sizeof(int)*dims);
    newHalo->lb[dims-1] = haloBlockUoffL[dims-1];
    newHalo->ub[dims-1] = haloBlockUoffU[dims-1];
    list->push_back(newHalo);
  }

  for (std::vector<HaloBlock*>::iterator itr= tmpList->begin();
      itr!=tmpList->end(); ++itr)
    list->push_back(*itr);

  delete tmpList;

  return list;
}

HRESULT InitFoldHaloBlock(BlockOnGPUState& state, 
    std::vector<HaloBlock*>& list) {
  for (std::vector<HaloBlock*>::iterator itr = list.begin();
      itr!=list.end(); ++itr) {
    state.haloMap->insert(std::pair<HaloBlock, HaloState*>
        (**itr, new HaloState()));
    HaloState* haloState = state.haloMap->find(**itr)->second;
    haloState->state = SInvalid;
    haloState->validBlockKey = NULL;
    haloState->subject = NULL;

    FoldHaloBlocks.insert(std::pair<HaloBlock, HaloState*>
        (**itr, haloState));
  }
  return SUCCESS;
}

HRESULT InitHaloBlock(BlockOnGPUState& state, 
    std::map<HaloBlock*,BlockKey*>& block2Key, 
    std::vector<HaloBlock*>& foldList, int thrd) {

  if (state.haloMap == NULL) {
    state.haloMap = new HaloMapType();
    for (std::map<HaloBlock*,BlockKey*>::iterator itr = block2Key.begin();
        itr!=block2Key.end(); ++itr) {
      HaloBlock* halo = itr->first;
      BlockKey* key = itr->second;
      pthread_rwlock_rdlock(&ModifyDataState);
      DataObjState* subject = DataState.find(*(itr->second))->second;
      pthread_rwlock_unlock(&ModifyDataState);

      HaloState* haloState = new HaloState();
      state.haloMap->insert(std::pair<HaloBlock,HaloState*>
          (*halo, haloState));
      haloState->state = SInvalid;
      haloState->validBlockKey = key;
      haloState->subject = subject;
      subject->Registe(haloState);

    }

    InitFoldHaloBlock(state, foldList);

    return SUCCESS;
  }

  return SUCCESS;
}

/*
 * divide halo region into multiple halo regions, each falls into
 * a data region, if not found, insert the halo region into a fold
 * halo region list.
 */
HRESULT InitHaloBlocks(pid_t threadId, int* lb, int*ub, int*loff,
    int* uoff, int dims, ElementType type, DataObjState* state,
    int* len, void* base) {

  int haloLoffL[DSM_MAX_DIMENSION], haloLoffU[DSM_MAX_DIMENSION],
  haloUoffL[DSM_MAX_DIMENSION], haloUoffU[DSM_MAX_DIMENSION];
  for (int i=0; i<dims; i++) {
    haloLoffL[i] = MAX(lb[i]-loff[i],0); haloLoffU[i] = lb[i]-1;
    haloUoffL[i] = ub[i]+1; haloUoffU[i] = MIN(MAX(ub[i]+uoff[i],0),len[i]-1);
  }
  std::vector<HaloBlock*>* list = GetHaloRegions(dims, haloLoffL,
      haloLoffU, haloUoffL, haloUoffU, lb, ub);
  for (std::vector<HaloBlock*>::iterator itr = list->begin();
      itr!=list->end(); ++itr) 
    (*itr)->base = base;

  std::map<HaloBlock*, BlockKey*> block2Key;
  std::vector<HaloBlock*> foldHaloBlock;

  for (std::vector<HaloBlock*>::iterator itr=list->begin();
      itr!=list->end(); ++itr) {

    bool found=false;
    pthread_rwlock_rdlock(&ModifyDataState);
    DataStateType::iterator dataPair=DataState.begin();
    DataStateType::iterator end=DataState.end();
    pthread_rwlock_unlock(&ModifyDataState);
    while (dataPair != end) {
      const BlockKey* key = &dataPair->first;
      if ( key->base==base && in(const_cast<BlockKey*>(key), *itr, dims)) {
        block2Key.insert(std::pair<HaloBlock*,BlockKey*>
            (*itr,const_cast<BlockKey*>(&dataPair->first)));
        found=true;
        break;
      }
      pthread_rwlock_rdlock(&ModifyDataState);
      ++dataPair;
      pthread_rwlock_unlock(&ModifyDataState);
    }

    if (found) continue;

    foldHaloBlock.push_back(*itr);
  }

  if (threadId == GPU1_THREAD) {
    InitHaloBlock(state->gpu1, block2Key, foldHaloBlock, GPU1_THREAD);
  } else {
    InitHaloBlock(state->gpu2, block2Key, foldHaloBlock, GPU2_THREAD);
  }

  for (std::vector<HaloBlock*>::iterator itr = list->begin();
      itr!=list->end(); ++itr) 
    delete *itr;
  delete list;

  return SUCCESS;
}

/*
 * if halo region of the block changed, change the shape of halo region in block,
 * and invalid halo region, if not, not to do.
 */
HRESULT AdjustDataBlockByHaloRegion(Device_Thread thread, BlockOnGPUState& state, 
    int* lb, int* ub, int* loff, int* uoff, int* shape, int dim, 
    ElementType type, DataObjState* dataState) {

  int size=1;
  bool reAlloc=false;


  for (int i=0; i<dim; i++) {
    if (loff[i] > state.loff[i] ||
        uoff[i] > state.uoff[i]) {
      reAlloc = true;
    }

    size *= loff[i]+shape[i]+uoff[i];
  }
  size*=ElementSize(type);


  int offset=0;
  for (int i=dim-1; i>=0; i--) 
    offset = offset*(shape[i]+loff[i]+uoff[i])+loff[i];

  if (reAlloc) {

    int oldLoff[DSM_MAX_DIMENSION], oldUoff[DSM_MAX_DIMENSION];
    memcpy(oldLoff, state.loff, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(oldUoff, state.uoff, sizeof(int)*DSM_MAX_DIMENSION);

    void* srcAddr = state.Addr;
    GPUAlloc(thread, &state.Addr, size);
    state.addr = (char*)state.Addr + offset*ElementSize(type);
    memcpy(state.loff, loff, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(state.uoff, uoff, sizeof(int)*DSM_MAX_DIMENSION);
    printf("%x on GPU%d addr %x realloc to %x with size %dKB\n",
        dataState->base, thread, srcAddr, state.Addr, size/1024);

    int validShape[DSM_MAX_DIMENSION];
    int sLen[DSM_MAX_DIMENSION], dLen[DSM_MAX_DIMENSION];
    for (int i=0; i<dim; i++) {
      validShape[i] = shape[i];	//here we don't consider the halo region
      sLen[i] = shape[i]+oldLoff[i]+oldUoff[i];
      dLen[i] = shape[i]+loff[i]+uoff[i];
    }

    cudaError_t last_error ;
    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing before kernel copy- %s\n",thread,cudaGetErrorString(last_error));
      assert(0);
    }
    Cudas[thread].LockAndPushContext();

    KernelCopy(state.Addr, dLen, loff,srcAddr, sLen, oldLoff,
        validShape, dim, type);
    checkCUDAError("halo copy");
    Cudas[thread].ReleaseAndPopContext();
    //copy block from old place to new place using kernel func

    GPUFree(thread, srcAddr); //just free old space
    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing kernel copy- %s\n",thread,cudaGetErrorString(last_error));
      assert(0);
    }

    state.ClearHaloMap();

    //TODO: halo region init
    InitHaloBlocks(thread, lb, ub, loff, uoff, dim, type, 
        dataState,dataState->dimsLen, dataState->base);

  }

  return SUCCESS;
}
//struct for redistribute
typedef struct tagTransferData {
  int dims;
  int shape[DSM_MAX_DIMENSION];
  void* dAddr;
  int dLen[DSM_MAX_DIMENSION],dOffset[DSM_MAX_DIMENSION];
  void* sAddr;
  int sLen[DSM_MAX_DIMENSION],sOffset[DSM_MAX_DIMENSION];
}transferData;

//compute two blocks's intersect block
bool intersect(const DataObjState* a,const DataObjState* b,int *dimsOffset,int *shape) {
  int ub[DSM_MAX_DIMENSION];
  for(int i = 0; i < a->dims; i++) {
    dimsOffset[i] = MAX(a->dimsOffset[i] , b->dimsOffset[i]);
    ub[i] = MIN(a->dimsOffset[i] + a->shape[i] , b->dimsOffset[i] + b->shape[i]);
  }
  for(int i = 0; i < a->dims; i++) {
    shape[i] = ub[i] - dimsOffset[i];
    if(shape[i] <= 0)
      return false;//not intersect
  }
  return true;//
}

//not support cpu to gpu get
//2011-7-19 add cpu get
void GetTransferData(int threadId,bool& flag,void* destAddr,DataObjState* state, 
    std::vector<DataObjState*>& gpuB,std::vector<DataObjState*>& cpu,
    std::vector<transferData*>& GpuBToGpuA,std::vector<transferData*>& cpuToGpuA, std::set<DataObjState*>& redistriBlocks) {
  int tmpOffset[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  if((gpuB.size() == 0 && cpu.size() == 0) || state == NULL)
    return ;
  for(std::vector<DataObjState*>::iterator preIter = gpuB.begin(); preIter != gpuB.end();++preIter) {
    if(intersect(state,*preIter,tmpOffset,shape)) {
      flag = true;
#ifdef PREFETCH 
      //printf("enter:%d %d %d %d \n",tmp.offset[0],tmp.offset[1],tmp.shape[0],tmp.shape[1]);
      bool findEquql = false;
      for(std::vector<RedistributionPoint>::iterator resIter = state->prefetchData.vecPoint.begin();
          resIter != state->prefetchData.vecPoint.end(); ++resIter){
        bool equal = true;
        for(int i = 0; i < state->dims && equal; i++){
          if(tmpOffset[i] != resIter->offset[i] || shape[i] != resIter->shape[i])
            equal = false; 
        }
        if(equal){
          findEquql = true;
          break;
        }
      }

      if(findEquql){
        printf("use prefetch data vecpoint\n");
        //added by shoubaojiang
        redistriBlocks.insert(*preIter);
        continue;
      }
#endif 
      transferData* transdata = new transferData();
      transdata->dims = state->dims;
      if(threadId == GPU1_THREAD) {
        transdata->dAddr = destAddr;
        transdata->sAddr = (*preIter)->gpu2.Addr;
        //if((*preIter)->GPUState(GPU2_THREAD) == SWaiting)
        //CudaSafe(cudaEventSynchronize((*preIter)->gpu2.validEvent));
      } else if(threadId == GPU2_THREAD) {
        transdata->dAddr = destAddr;
        transdata->sAddr = (*preIter)->gpu1.Addr;
        //if((*preIter)->GPUState(GPU1_THREAD) == SWaiting)
        //CudaSafe(cudaEventSynchronize((*preIter)->gpu1.validEvent));
      } 
      for(int k = 0; k < state->dims; k++) {
        transdata->shape[k] = shape[k];

        if(threadId == GPU1_THREAD){
          transdata->dLen[k] = state->shape[k] + state->gpu1.loff[k] + state->gpu1.uoff[k];
          transdata->dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu1.loff[k];
        } else if(threadId == GPU2_THREAD) {
          transdata->dLen[k] = state->shape[k] + state->gpu2.loff[k] + state->gpu2.uoff[k];
          transdata->dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu2.loff[k];
        }

        if(threadId == GPU1_THREAD) {
          transdata->sLen[k] = (*preIter)->shape[k] + (*preIter)->gpu2.loff[k] + (*preIter)->gpu2.uoff[k];
          transdata->sOffset[k] = tmpOffset[k] - (*preIter)->dimsOffset[k] + (*preIter)->gpu2.loff[k];
        } else if(threadId == GPU2_THREAD) {
          transdata->sLen[k] = (*preIter)->shape[k] + (*preIter)->gpu1.loff[k] + (*preIter)->gpu1.uoff[k];
          transdata->sOffset[k] = tmpOffset[k] - (*preIter)->dimsOffset[k] + (*preIter)->gpu1.loff[k];
        }
      }
      //added by shoubaojiang
      redistriBlocks.insert(*preIter);
      //end of add
      GpuBToGpuA.push_back(transdata);

    }
  } 

  /*
     for(std::vector<DataObjState*>::iterator iter = cpu.begin(); iter != cpu.end();++iter) { 
     if(intersect(state,*iter,tmpOffset,shape)) {
     RedistributionPoint tmp;
     memcpy(tmp.offset,tmpOffset,state->dims*sizeof(int));
     memcpy(tmp.shape,shape,state->dims*sizeof(int));

#ifdef PREFETCH 
if(isPrefetch){
state->prefetchData.vecPoint.push_back(tmp);
} else {
  //printf("enter:%d %d %d %d \n",tmp.offset[0],tmp.offset[1],tmp.shape[0],tmp.shape[1]);
  std::vector<RedistributionPoint>::iterator result =\
  find(state->prefetchData.vecPoint.begin(),state->prefetchData.vecPoint.end(),tmp);
  if(result != state->prefetchData.vecPoint.end()){
//printf("use prefetch data vecpoint\n");
//added by shoubaojiang
redistriBlocks.insert(*iter);
continue;
}
}
#endif
transferData* transdata = new transferData();
transdata->dims = state->dims;
transdata->dAddr = destAddr;
transdata->sAddr = state->base;
for(int k = 0; k < state->dims; k++) {
transdata->shape[k] = shape[k];
if(threadId == GPU1_THREAD){
transdata->dLen[k] = state->shape[k] + state->gpu1.loff[k] + state->gpu1.uoff[k];
transdata->dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu1.loff[k];
} else if(threadId == GPU2_THREAD) {
transdata->dLen[k] = state->shape[k] + state->gpu2.loff[k] + state->gpu2.uoff[k];
transdata->dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu2.loff[k];
}
}
memcpy(transdata->sLen,state->dimsLen,sizeof(transdata->sLen));
memcpy(transdata->sOffset,tmpOffset,sizeof(tmpOffset));
//added by shoubaojiang
redistriBlocks.insert(*iter);
//end of add
cpuToGpuA.push_back(transdata);

}
}*/
}

HRESULT CpuCopy(void* dest, int* dLen, int* dOffset,
    void* src, int* sLen, int* sOffset,
    int* shape, int dim, ElementType type) {

  int size=1;
  for (int i=0; i<dim; i++)
    size *= shape[i];
  size*=ElementSize(type); size/=4;  //every thread copy four byte, a int

  int reShape[DSM_MAX_DIMENSION];
  for (int i=0; i<dim; i++)
    reShape[i]=shape[i]*ElementSize(type)/4;

  int dimIdx[DSM_MAX_DIMENSION];
  int srcIdx;
  int destIdx;
  int *destAddr = (int*)dest,*srcAddr = (int*)src;
  for (int j = 0; j < size; j++){
    int idx = j;
    srcIdx = destIdx = 0;
    for (int i=0; i<dim-1; i++) {
      dimIdx[i] = idx % reShape[i];
      idx /= reShape[i];
    }

    dimIdx[dim-1] = idx;
    for (int i=dim-1; i>=0; i--) {
      srcIdx = srcIdx*sLen[i]+dimIdx[i]+sOffset[i];
      destIdx = destIdx*dLen[i]+dimIdx[i]+dOffset[i];
    }
    //memcpy(&destAddr[destIdx],&srcAddr[srcIdx],shape[dim-1]);
    destAddr[destIdx] = srcAddr[srcIdx];
    //destAddr[0] = srcAddr[0];
  }
  //printf("destIdx : %d srcIdx: %d\n",destIdx,srcIdx);
}

void TransferCToG(int threadId,ElementType type, std::vector<transferData*>& CpuToGpu) {
  int dLen[DSM_MAX_DIMENSION];
  int dOffset[DSM_MAX_DIMENSION];
  if(CpuToGpu.size() == 0)
    return ;
  memset(dLen,0,sizeof(dLen));
  int dims = CpuToGpu.at(0)->dims;
  for(std::vector<transferData*>::iterator iter = CpuToGpu.begin(); iter != CpuToGpu.end();++iter) {
    dLen[dims-1] += (*iter)->shape[dims-1];
    for(int i = 0; i < (*iter)->dims - 1; i++){
      dLen[i] =  (*iter)->shape[i];
    }
  }

  //kernelcopy to tmp space
  int size = 1;
  for(int i = 0; i < dims; i++) {
    size *= dLen[i] ;
  }
  size *= ElementSize(type);


  void* cpuforGPU ;
  //Cudas[threadId].LockAndPushContext();
  //CudaSafe(cudaHostAlloc((void**)&cpuforGPU,size,cudaHostAllocPortable));
  //Cudas[threadId].ReleaseAndPopContext();
  cpuforGPU = PorMemManager->allocBlock(size);

  memset(dOffset,0,sizeof(dOffset));
  for(std::vector<transferData*>::iterator iter = CpuToGpu.begin(); iter != CpuToGpu.end(); ++iter) {
    CpuCopy(cpuforGPU,dLen,dOffset,(*iter)->sAddr,(*iter)->sLen,(*iter)->sOffset,(*iter)->shape,dims,type);	
    memcpy((*iter)->sOffset,dOffset,sizeof(dOffset));
    dOffset[dims-1] += (*iter)->shape[dims-1];
  }

  //....
  void *tmpSpace;
  GPUAlloc((Device_Thread)threadId, &tmpSpace, size);
  Cudas[threadId].LockAndPushContext();
  //CudaSafe(cudaMalloc(&tmpSpace,size));

  CudaSafe(cudaMemcpy(tmpSpace,cpuforGPU,size,cudaMemcpyHostToDevice));
  //CudaSafe(cudaFreeHost(cpuforGPU));
  //free(cpuforGPU);
  for(std::vector<transferData*>::iterator iter = CpuToGpu.begin(); iter != CpuToGpu.end(); ++iter) {
    KernelCopy((*iter)->dAddr,(*iter)->dLen,(*iter)->dOffset,tmpSpace,dLen,(*iter)->sOffset,(*iter)->shape,dims,type);
    delete (*iter);
    *iter = NULL;
    checkCUDAError("gpu reo");
  }
  //CudaSafe(cudaFree(tmpSpace));
  GPUFree((Device_Thread)threadId, tmpSpace);

  Cudas[threadId].ReleaseAndPopContext();
}


void TransferGToG(int threadId,ElementType type, std::vector<transferData*>& GpuBToGpuA,
    cudaStream_t& srcStream,cudaStream_t& destStream){
  int dLen[DSM_MAX_DIMENSION];
  int dOffset[DSM_MAX_DIMENSION];
  if(GpuBToGpuA.size() == 0)
    return ;
  memset(dLen,0,sizeof(dLen));
  int dims = GpuBToGpuA.at(0)->dims;
  for(std::vector<transferData*>::iterator iter = GpuBToGpuA.begin(); iter != GpuBToGpuA.end();++iter){
    dLen[dims-1] += (*iter)->shape[dims-1];

    for(int i = 0; i < (*iter)->dims - 1; i++){
      dLen[i] =  (*iter)->shape[i];
    }
  }

  //kernelcopy to tmp space
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= dLen[i] ;
  }
  size *= ElementSize(type);

  void* tmpSpace;
  tmpSpace = DevMemManager[OtherGPUThd(threadId)]->allocBlock(size);

  Cudas[OtherGPUThd(threadId)].LockAndPushContext();

  //printf("malloc size:%d\n",size);	
  //CudaSafe(cudaMalloc(&tmpSpace,size));

  memset(dOffset,0,sizeof(dOffset));
  for(std::vector<transferData*>::iterator iter = GpuBToGpuA.begin(); iter != GpuBToGpuA.end(); ++iter){
    KernelCopy(tmpSpace,dLen,dOffset,(*iter)->sAddr,(*iter)->sLen,
        (*iter)->sOffset,(*iter)->shape,dims,type);	
    //KernelCopyAsync(tmpSpace,dLen,dOffset,(*iter)->sAddr,(*iter)->sLen,
    //    (*iter)->sOffset,(*iter)->shape,dims,type,srcStream);	
    checkCUDAError("trans to cpu");
    //dOffset[dims-1] += (*iter)->shape[dims-1];
    memcpy((*iter)->sOffset,dOffset,sizeof(dOffset));
    dOffset[dims-1] += (*iter)->shape[dims-1];
  }


  void* cpuforGPU;// = malloc(size);
  //CudaSafe(cudaHostAlloc((void**)&cpuforGPU,size,cudaHostAllocDefault));
  cpuforGPU = PorMemManager->allocBlock(size);
  //printf("cpu inn\n");
  CudaSafe(cudaMemcpy(cpuforGPU,tmpSpace,size,cudaMemcpyDeviceToHost));
  //CudaSafe(cudaMemcpyAsync(cpuforGPU,tmpSpace,size,cudaMemcpyDeviceToHost,srcStream));
  //CudaSafe(cudaFree(tmpSpace));

  Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
  DevMemManager[OtherGPUThd(threadId)]->freeBlock(tmpSpace);


  ///
  ///
  tmpSpace = DevMemManager[threadId]->allocBlock(size);
  Cudas[threadId].LockAndPushContext();

  //....
  //CudaSafe(cudaMalloc(&tmpSpace,size));

  CudaSafe(cudaMemcpy(tmpSpace,cpuforGPU,size,cudaMemcpyHostToDevice));
  //CudaSafe(cudaMemcpyAsync(tmpSpace,cpuforGPU,size,cudaMemcpyHostToDevice,destStream));

  PorMemManager->freeBlock(cpuforGPU);


  //free(cpuforGPU);
  for(std::vector<transferData*>::iterator iter = GpuBToGpuA.begin(); iter != GpuBToGpuA.end(); ++iter)
  {
    KernelCopy((*iter)->dAddr,(*iter)->dLen,(*iter)->dOffset,
        tmpSpace,dLen,(*iter)->sOffset,(*iter)->shape,dims,type);
    //KernelCopyAsync((*iter)->dAddr,(*iter)->dLen,(*iter)->dOffset,
    //    tmpSpace,dLen,(*iter)->sOffset,(*iter)->shape,dims,type,destStream);
    checkCUDAError("redis gpu");
    delete (*iter);
    *iter = NULL;
  }
  //CudaSafe(cudaFree(tmpSpace));

  Cudas[threadId].ReleaseAndPopContext();
  DevMemManager[threadId]->freeBlock(tmpSpace);

  //Cudas[OtherGPUThd(threadId)].LockAndPushContext();
  //CudaSafe(cudaFreeHost(cpuforGPU));
  //Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();

}

HRESULT Redistributed(int threadId,bool isPrefetch,void* destAddr,DataObjState* state,
    std::vector<DataObjState*>& gpu1Blocks,
    std::vector<DataObjState*>& gpu2Blocks,
    std::vector<DataObjState*>& cpuBlocks,
    cudaStream_t& srcStream,cudaStream_t& destStream,Operation& opr)
{
  if(gpu1Blocks.size() ==0 && gpu2Blocks.size() == 0 &&  cpuBlocks.size() == 0)
    return -1;//don't have data   
  bool intersectFlag = false;
  int sLen[DSM_MAX_DIMENSION], dLen[DSM_MAX_DIMENSION];
  int sOffset[DSM_MAX_DIMENSION],dOffset[DSM_MAX_DIMENSION],tmpOffset[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];

  //current gpu copy
  //Cudas[threadId].LockAndPushContext();  
  //This and the rwlock ModifyDataState will form a dead lock, remember not to nest a lock with another
  if(threadId == GPU1_THREAD) {
    for(std::vector<DataObjState*>::iterator preIter = gpu1Blocks.begin(); preIter != gpu1Blocks.end();++preIter){
      if(intersect(state,*preIter,tmpOffset,shape)){
        //if((*preIter)->GPUState(GPU1_THREAD) == SWaiting)
        //CudaSafe(cudaEventSynchronize((*preIter)->gpu1.validEvent));
        for(int k = 0; k < state->dims; k++){
          dLen[k] = state->shape[k] + state->gpu1.loff[k] + state->gpu1.uoff[k];
          sLen[k] = (*preIter)->shape[k] + (*preIter)->gpu1.loff[k] + (*preIter)->gpu1.uoff[k];
          dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu1.loff[k];
          sOffset[k] = tmpOffset[k] - (*preIter)->dimsOffset[k] + (*preIter)->gpu1.loff[k];
        }
        Cudas[threadId].LockAndPushContext();  
        KernelCopy(destAddr,dLen,dOffset,(*preIter)->gpu1.Addr,
            sLen,sOffset,shape,state->dims,state->eleType);
        //KernelCopyAsync(destAddr,dLen,dOffset,(*preIter)->gpu1.Addr,
        //  sLen,sOffset,shape,state->dims,state->eleType,destStream);
        Cudas[threadId].ReleaseAndPopContext();
        //cudaThreadSynchronize();
        checkCUDAError("copy");
        intersectFlag = true;

        //added by shoubaojiang
        int lb[DSM_MAX_PTHREADS],ub[DSM_MAX_PTHREADS];
        for (int i=0; i<state->dims; i++) {
          lb[i] = state->dimsOffset[i];
          ub[i] = state->dimsOffset[i]+state->shape[i]-1;
        }
        (*preIter)->AddRedistributionBlock(lb, ub);
        if ((*preIter)->IsAllRedistributed()) {
          pthread_rwlock_wrlock(&ModifyDataState);
          DataStateType::iterator itr = DataState.begin();
          DataStateType::iterator itr_end = DataState.end();
          //pthread_rwlock_unlock(&ModifyDataState);

          while (itr!=itr_end) {
            if (itr->second == (*preIter)) {
              (*preIter)->gpu1.FreeSpace(GPU1_THREAD);
              (*preIter)->gpu2.FreeSpace(GPU2_THREAD);
              //pthread_rwlock_wrlock(&ModifyDataState);
              DataState.erase(itr++);
              //pthread_rwlock_unlock(&ModifyDataState);
              //break;
            } else {
              //pthread_rwlock_rdlock(&ModifyDataState);
              ++itr;
              //pthread_rwlock_unlock(&ModifyDataState);
            }
          }
          pthread_rwlock_unlock(&ModifyDataState);
        }
        //end of add
      }
    }
  } else if(threadId == GPU2_THREAD ) {
    for(std::vector<DataObjState*>::iterator preIter = gpu2Blocks.begin(); preIter != gpu2Blocks.end();++preIter){
      if(intersect(state,*preIter,tmpOffset,shape)){
        //if((*preIter)->GPUState(GPU2_THREAD) == SWaiting)
        //CudaSafe(cudaEventSynchronize((*preIter)->gpu2.validEvent));
        for(int k = 0; k < state->dims; k++){
          dLen[k] = state->shape[k] + state->gpu2.loff[k] + state->gpu2.uoff[k];
          sLen[k] = (*preIter)->shape[k] + (*preIter)->gpu2.loff[k] + (*preIter)->gpu2.uoff[k];
          dOffset[k] = tmpOffset[k] - state->dimsOffset[k] + state->gpu2.loff[k];
          sOffset[k] = tmpOffset[k] - (*preIter)->dimsOffset[k] + (*preIter)->gpu2.loff[k];
        }
        Cudas[threadId].LockAndPushContext();  
        KernelCopy(destAddr,dLen,dOffset,(*preIter)->gpu2.Addr,
            sLen,sOffset,shape,state->dims,state->eleType);
        //KernelCopyAsync(destAddr,dLen,dOffset,(*preIter)->gpu2.Addr,
        //  sLen,sOffset,shape,state->dims,state->eleType,destStream);
        Cudas[threadId].ReleaseAndPopContext();

        //cudaThreadSynchronize();
        checkCUDAError("pre before redis");
        intersectFlag = true;

        //added by shoubaojiang
        int lb[DSM_MAX_PTHREADS],ub[DSM_MAX_PTHREADS];
        for (int i=0; i<state->dims; i++) {
          lb[i] = state->dimsOffset[i];
          ub[i] = state->dimsOffset[i]+state->shape[i]-1;
        }
        (*preIter)->AddRedistributionBlock(lb, ub);
        if ((*preIter)->IsAllRedistributed()) {

          pthread_rwlock_wrlock(&ModifyDataState);
          DataStateType::iterator itr = DataState.begin();
          DataStateType::iterator itr_end = DataState.end();
          //pthread_rwlock_unlock(&ModifyDataState);
          while (itr!=itr_end) {
            if (itr->second == (*preIter)) {
              (*preIter)->gpu1.FreeSpace(GPU1_THREAD);
              (*preIter)->gpu2.FreeSpace(GPU2_THREAD);
              //pthread_rwlock_wrlock(&ModifyDataState);
              DataState.erase(itr++);
              //pthread_rwlock_unlock(&ModifyDataState);
              //break;
            } else {
              //pthread_rwlock_rdlock(&ModifyDataState);
              ++itr;
              //pthread_rwlock_unlock(&ModifyDataState);
            }
          }
          pthread_rwlock_unlock(&ModifyDataState);
        }
        //end of add

      }
    }
  }
  //Cudas[threadId].ReleaseAndPopContext();
  //The same reason as above.

  std::vector<transferData*> OthGpuToGpu,cpuToGpu;
  std::set<DataObjState*> srcData;

  GetTransferData(threadId,intersectFlag,destAddr,state,
      threadId == GPU1_THREAD?gpu2Blocks:gpu1Blocks,
      cpuBlocks,OthGpuToGpu,cpuToGpu, srcData);

  if(OthGpuToGpu.size()) {
    //intersectFlag = true;
    TransferGToG(threadId,state->eleType,OthGpuToGpu,srcStream,destStream);
  }
  if(cpuToGpu.size()){
    //TransferCToG(threadId,state->eleType,cpuToGpu); 
  }

  //added by shoubaojiang
  for (std::set<DataObjState*>::iterator iter= srcData.begin();
      iter != srcData.end(); ++iter) {
    DataObjState* srcDataState = (*iter);
    int lb[DSM_MAX_DIMENSION],ub[DSM_MAX_DIMENSION];
    for (int i=0; i<DSM_MAX_DIMENSION; i++) {
      lb[i] = state->dimsOffset[i];
      ub[i] = state->dimsOffset[i] + state->shape[i]-1;
    }
    srcDataState->AddRedistributionBlock(lb,ub);
    if (srcDataState->IsAllRedistributed()) {
      pthread_rwlock_wrlock(&ModifyDataState);
      typename DataStateType::iterator itr=DataState.begin();
      typename DataStateType::iterator itr_end=DataState.end();
      //pthread_rwlock_unlock(&ModifyDataState);

      while (itr!=itr_end) {
        if (itr->second == srcDataState) {
          srcDataState->gpu1.FreeSpace(GPU1_THREAD);
          srcDataState->gpu2.FreeSpace(GPU2_THREAD);
          //pthread_rwlock_wrlock(&ModifyDataState);
          DataState.erase(itr++);
          //pthread_rwlock_unlock(&ModifyDataState);
          //break;
        } else {
          //pthread_rwlock_rdlock(&ModifyDataState);
          ++itr;
          //pthread_rwlock_unlock(&ModifyDataState);
        }
      }
      pthread_rwlock_unlock(&ModifyDataState);
    }
  }
  //end of add

  if(threadId == GPU1_THREAD || threadId == GPU2_THREAD) 
    state->pGpus[threadId]->stream = destStream;

  if(intersectFlag)
    return SUCCESS;
  else
    return -1; 
}

#define TransBlockX 16
#define TransBlockY 16
__global__ void Transpose3d_12_8_kernel(double* d_src, double* d_dest, int dim1Len,
    int dim2Len, int dim3Len)
{

  __shared__ double tmp[TransBlockY][TransBlockX];

  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;

  int new_x = threadIdx.x+blockIdx.y*blockDim.y;
  int new_y = threadIdx.y+blockIdx.x*blockDim.x;

  for (int k=0; k<dim3Len; k++)
  {
    if (x<dim1Len && y<dim2Len)
      tmp[threadIdx.y][threadIdx.x] = d_src[k*dim1Len*dim2Len+y*dim1Len+x];
    __syncthreads();
    if (new_x<dim2Len && new_y<dim1Len)
      d_dest[k*dim1Len*dim2Len+new_y*dim2Len+ new_x] = tmp[threadIdx.x][threadIdx.y];
  }

}

__global__ void Transpose4d5_12_8_kernel(double* d_src, double* d_dest, int dim1Len,
    int dim2Len, int dim3Len, int dim4Len)
{
  __shared__ double tmp[TransBlockY][TransBlockX];

  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  int new_x = threadIdx.x+blockIdx.y*blockDim.y;
  int new_y = threadIdx.y+blockIdx.x*blockDim.x;

  for (int k=0; k<dim4Len; k++)
  {
    for (int m=0; m<dim3Len; m++)
    {
      if (x<dim1Len && y<dim2Len)
        tmp[threadIdx.y][threadIdx.x] = d_src[k*dim3Len*dim2Len*dim1Len+m*dim1Len*dim2Len+y*dim1Len+x];
      __syncthreads();
      if (new_x<dim2Len && new_y<dim1Len)
        d_dest[k*dim3Len*dim1Len*dim2Len+m*dim1Len*dim2Len+new_y*dim2Len+new_x] = tmp[threadIdx.x][threadIdx.y];
    }
  }
}

__global__ void Transpose3d_23_8_kernel(double* d_src, double* d_dest,int dim1Len,
    int dim2Len, int dim3Len)
{

  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  if (x>=dim1Len || y>=dim2Len) return;

  for (int k=0; k<dim3Len; k++)
  {
    d_dest[y*dim3Len*dim1Len+k*dim1Len+x] = d_src[k*dim2Len*dim1Len+y*dim1Len+x] ;
  }
}

__global__ void Transpose4d5_24_8_kernel(double* d_src, double* d_dest, int dim1Len,
    int dim2Len, int dim3Len, int dim4Len )
{

  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  if (x>=dim1Len || y>=dim2Len) return ;

  for (int k=0; k<dim4Len; k++)
  {
    for (int m=0; m<dim3Len; m++)
    {
      d_dest[y*dim3Len*dim4Len*dim1Len+ m*dim4Len*dim1Len+ k*dim1Len+x] =
        d_src[k*dim3Len*dim2Len*dim1Len+m*dim2Len*dim1Len+y*dim1Len+x];
    }
  }
}


double* Transpose3d_12_8(Device_Thread threadId, double* d_array,int dim1Len, int dim2Len, int dim3Len)
{
  //FIXME: single gpu only now
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  printf("3d_12_8 malloc space %lld \n",dim1Len*dim2Len*dim3Len*sizeof(double));

  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose3d_12_8_kernel<<<grid,block>>>(d_array, dest, dim1Len, dim2Len, dim3Len);
  checkCUDAError("transpose 3d 12");
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}


double* Transpose4d5_12_8(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len, int dim4Len)
{
  //FIXME: single gpu only now
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim4Len*dim3Len*dim2Len*dim1Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  printf("4d_12_8 malloc space %lld \n",dim1Len*dim2Len*dim3Len*dim4Len*sizeof(double));
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose4d5_12_8_kernel<<<grid, block>>>(d_array, dest, dim1Len, dim2Len, dim3Len, dim4Len);
  checkCUDAError("transpose 4d 23");
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}

double* Transpose3d_23_8Async(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len,
    cudaStream_t stream)
{
  //FIXME: single gpu only
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose3d_23_8_kernel<<<grid, block, 0, stream>>>(d_array, dest, dim1Len, dim2Len, dim3Len);
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}


double* Transpose3d_23_8(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len)
{
  //FIXME: single gpu only
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose3d_23_8_kernel<<<grid, block>>>(d_array, dest, dim1Len, dim2Len, dim3Len);
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}

double* Transpose4d5_24_8Async(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len,
    int dim4Len, cudaStream_t stream)
{
  //FIXME
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*dim4Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose4d5_24_8_kernel<<<grid, block, 0, stream>>>(d_array, dest, dim1Len, dim2Len, dim3Len, dim4Len);
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}

double* Transpose4d5_24_8(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len, int dim4Len)
{
  //FIXME
  //7.13: support multiple gpu by shoubj
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*dim4Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);
  Transpose4d5_24_8_kernel<<<grid, block>>>(d_array, dest, dim1Len, dim2Len, dim3Len, dim4Len);
  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, d_array);
  return dest;
}

double* Transpose3d_rsh1(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len)
{
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);

  Transpose3d_12_8_kernel<<<grid, block>>>(d_array,dest, dim1Len, dim2Len, dim3Len);
  grid.x= (dim2Len+TransBlockX-1)/TransBlockX; grid.y = (dim1Len+TransBlockY-1)/TransBlockY;
  Transpose3d_23_8_kernel<<<grid, block>>>(dest, d_array, dim2Len, dim1Len, dim3Len);

  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, dest);
  return d_array;
}

double* Transpose4d5_rsh1(Device_Thread threadId, double* d_array, int dim1Len, int dim2Len, int dim3Len, int dim4Len)
{
  double* dest;
  GPUAlloc(threadId, (void**)&dest, dim1Len*dim2Len*dim3Len*dim4Len*sizeof(double));
  Cudas[threadId].LockAndPushContext();
  dim3 block(TransBlockX, TransBlockY);
  dim3 grid((dim1Len+TransBlockX-1)/TransBlockX, (dim2Len+TransBlockY-1)/TransBlockY);

  Transpose4d5_12_8_kernel<<<grid, block>>>(d_array, dest, dim1Len, dim2Len, dim3Len, dim4Len);
  grid.x=(dim2Len+TransBlockX-1)/TransBlockX; grid.y=(dim1Len+TransBlockY-1)/TransBlockY;
  Transpose4d5_24_8_kernel<<<grid, block>>>(dest, d_array, dim2Len, dim1Len, dim3Len, dim4Len);

  Cudas[threadId].ReleaseAndPopContext();
  GPUFree(threadId, dest);
  return d_array;
}


HRESULT omprArrayTranspose(ArrayTranspose arrayTranspose,pid_t threadId,
    ElementType type,int dims,DataObjState* state )
{
  BlockOnGPUState* gpuState;
  if (threadId == GPU1_THREAD)
    gpuState = &state->gpu1;
  else
    gpuState = &state->gpu2;
  int dimsLen[DSM_MAX_PTHREADS];
  for(int i = 0; i < dims; i++){
    dimsLen[i] = state->shape[i] + gpuState->loff[i] + gpuState->uoff[i] ;
  }
  if(dims == 3){
    switch(ElementSize(type)){
      case 8:
        if(arrayTranspose.one == 1 && arrayTranspose.other == 2 ||
            arrayTranspose.one == 2 && arrayTranspose.other == 1 ){
          //Cudas[threadId].LockAndPushContext();
          /*
             if(gpuState->state == SWaiting){ 
             if(threadId == GPU1_THREAD){
             CudaSafe(cudaEventSynchronize(state->gpu1.validEvent));
             } else if(threadId == GPU2_THREAD){
             CudaSafe(cudaEventSynchronize(state->gpu2.validEvent));
             }
             }*/

          gpuState->Addr = (void*) Transpose3d_12_8((Device_Thread)threadId, (double*) gpuState->Addr, dimsLen[0], 
              dimsLen[1],dimsLen[2]);
          //Cudas[threadId].ReleaseAndPopContext();

          int offset=0;	//offset between Addr and addr on gpu
          for (int i=dims-1; i>=0; i--) {
            offset = offset*(state->shape[i]+gpuState->loff[i]+gpuState->uoff[i])+gpuState->loff[i];
          }
          gpuState->addr = (char*)gpuState->Addr + offset*ElementSize(type);
          //TODO add shape and Blockkey modify
        } else if(arrayTranspose.one == 6 && arrayTranspose.other == 6){
          printf("pid %d, transpose 3d rsh1\n",threadId);
          //Cudas[threadId].LockAndPushContext();
          gpuState->Addr = (void*) Transpose3d_rsh1((Device_Thread)threadId, (double*) gpuState->Addr, dimsLen[0], dimsLen[1],dimsLen[2]);
          //Cudas[threadId].ReleaseAndPopContext();

          int offset=0;	//offset between Addr and addr on gpu
          for (int i=dims-1; i>=0; i--) {
            offset = offset*(state->shape[i]+gpuState->loff[i]+gpuState->uoff[i])+gpuState->loff[i];
          }
          gpuState->addr = (char*)gpuState->Addr + offset*ElementSize(type);
          //TODO add shape and Blockkey modify
        }
        break;
      default:break;
    }
  } else if(dims == 4){
    switch(ElementSize(type)){
      case 8:
        if(arrayTranspose.one == 1 && arrayTranspose.other == 2 ||
            arrayTranspose.one == 2 && arrayTranspose.other == 1 ){
          if(state->dimsLen[2] == 5){
            //Cudas[threadId].LockAndPushContext();
            /*
               if(gpuState->state == SWaiting){
               if(threadId == GPU1_THREAD){
               CudaSafe(cudaEventSynchronize(state->gpu1.validEvent));
               } else if(threadId == GPU2_THREAD){
               CudaSafe(cudaEventSynchronize(state->gpu2.validEvent));
               }
               }*/

            gpuState->Addr =  (void*)Transpose4d5_12_8((Device_Thread)threadId, (double*) gpuState->Addr, 
                dimsLen[0], dimsLen[1],5,dimsLen[3]);
            //Cudas[threadId].ReleaseAndPopContext();

            int offset=0;	//offset between Addr and addr on gpu
            for (int i=dims-1; i>=0; i--) {
              offset = offset*(state->shape[i]+gpuState->loff[i]+gpuState->uoff[i])+gpuState->loff[i];
            }
            gpuState->addr = (char*)gpuState->Addr + offset*ElementSize(type);
            //TODO add shape and Blockkey modify
          } else if ( state->dimsLen[2] == 15) {

          }
        } else if(arrayTranspose.one == 6 && arrayTranspose.other == 6){
          //Cudas[threadId].LockAndPushContext();
          gpuState->Addr = (void*) Transpose4d5_rsh1((Device_Thread)threadId, (double*) gpuState->Addr, dimsLen[0], dimsLen[1],dimsLen[2],dimsLen[3]);
          //Cudas[threadId].ReleaseAndPopContext();

          int offset=0;	//offset between Addr and addr on gpu
          for (int i=dims-1; i>=0; i--) {
            offset = offset*(state->shape[i]+gpuState->loff[i]+gpuState->uoff[i])+gpuState->loff[i];
          }
          gpuState->addr = (char*)gpuState->Addr + offset*ElementSize(type);
          //TODO add shape and Blockkey modify


        } else if(arrayTranspose.one == 2 && arrayTranspose.other == 4 ||
            arrayTranspose.one == 4 && arrayTranspose.other == 2 ){
          if(state->dimsLen[2] == 5){
            //Cudas[threadId].LockAndPushContext();
            /*
               if(gpuState->state == SWaiting){
               if(threadId == GPU1_THREAD){
               CudaSafe(cudaEventSynchronize(state->gpu1.validEvent));
               } else if(threadId == GPU2_THREAD){
               CudaSafe(cudaEventSynchronize(state->gpu2.validEvent));
               }
               }*/

            gpuState->Addr =  (void*)Transpose4d5_24_8((Device_Thread)threadId, (double*) gpuState->Addr, 
                dimsLen[0], dimsLen[1],5,dimsLen[3]);
            //Cudas[threadId].ReleaseAndPopContext();

            int offset=0;	//offset between Addr and addr on gpu
            for (int i=dims-1; i>=0; i--) {
              offset = offset*(state->shape[i]+gpuState->loff[i]+gpuState->uoff[i])+gpuState->loff[i];
            }
            gpuState->addr = (char*)gpuState->Addr + offset*ElementSize(type);
            //TODO add shape and Blockkey modify
          }

        }
        break;
      default:break;
    }

  }
  return 0;

}


HRESULT Redistribution(DataInfo* pInfo,int threadId,int streamId,Operation& opr,
    BlockKey& key, DataObjState* state){
  //need redisribution's block
  //DataObjState* state = DataState.find(key)->second;
  if (state == NULL) assert(0);
  //data block in all memory
  std::vector<DataObjState*> gpu1Blocks;
  std::vector<DataObjState*> gpu2Blocks;
  std::vector<DataObjState*> cpuBlocks;

  //get all block with the same cpu base
  pthread_rwlock_rdlock(&ModifyDataState);
  DataStateType::iterator iter = DataState.begin(); 
  DataStateType::iterator end = DataState.end(); 
  pthread_rwlock_unlock(&ModifyDataState);
  while (iter != end) {
    if( iter->first.base == state->base && iter->second != state){
      //blocks in gpu1 
      if(iter->second->GPUState(GPU1_THREAD) == SValid || iter->second->GPUState(GPU1_THREAD) == SWaiting){
        gpu1Blocks.push_back(iter->second);
      } else if(iter->second->GPUState(GPU2_THREAD) == SValid || 
          iter->second->GPUState(GPU2_THREAD) == SWaiting ){
        gpu2Blocks.push_back(iter->second);
      } else if(iter->second->cpuState == SValid) {
        //printf("addr:%x\n",state->base);
        cpuBlocks.push_back(iter->second);
      } else { 
        printf("block no valid in all memory!!\n");
        return -1; 
      }
    }
    pthread_rwlock_rdlock(&ModifyDataState);
    ++iter;
    pthread_rwlock_unlock(&ModifyDataState);
  }
  long int totalSize = 1;
  long int offset = 0;

  bool isPrefetch = false;

  if(threadId == GPU1_THREAD){
    //prefetch data use    
#ifdef PREFETCH 
    if(state->prefetchData.pid == GPU1_THREAD){	 
      state->gpu1.FreeSpace(GPU1_THREAD);
      //if(state->gpu1.Addr != NULL)
      //  GPUFree((Device_Thread)threadId, state->gpu1.Addr);
      assert(state->prefetchData.Addr);
      printf("retribu use prefetch data\n");
      state->gpu1.Addr = state->prefetchData.Addr;	 
      state->gpu1.addr = state->prefetchData.addr;	 
      state->gpu1.stream = state->prefetchData.stream;    
      isPrefetch = true;
    }
    /*
       if(state->prefetchData.Addr != NULL){
       printf("pid %d ",state->prefetchData.pid);
       GPUFree(GPU1_THREAD,state->prefetchData.Addr);
       state->prefetchData.clear();
       assert(0); 
       }*/
#endif 
    for (int i = state->dims-1; i >= 0; i--) {
      totalSize *= state->shape[i]+state->gpu1.loff[i]+state->gpu1.uoff[i];
      offset = offset*(state->shape[i]+state->gpu1.loff[i]+state->gpu1.uoff[i])+state->gpu1.loff[i];
    }
    if (state->gpu1.state == SInvalid && state->gpu1.Addr == NULL) {
      GPUAlloc(GPU1_THREAD,&(state->gpu1.Addr),totalSize*ElementSize(state->eleType)); 
      state->gpu1.addr = (char*)state->gpu1.Addr + offset*ElementSize(state->eleType);
    }
    //AdjustDataBlockByHaloRegion(GPU1_THREAD, state->gpu1, key.lb, key.ub,
    //    state->gpu1.loff, state->gpu1.uoff, state->shape, state->dims, state->eleType, state);

    //UpdateHaloRegion(GPU1_THREAD, state->gpu1, *state);
  } else if(threadId == GPU2_THREAD){
    //prefetch data use    
#ifdef PREFETCH
    if(state->prefetchData.pid == GPU2_THREAD){	 
      printf("retribu use prefetch data\n");
      state->gpu2.FreeSpace(GPU2_THREAD);
      //if(state->gpu2.Addr != NULL)
      //  GPUFree((Device_Thread)threadId, state->gpu2.Addr);
      assert(state->prefetchData.Addr);
      state->gpu2.Addr = state->prefetchData.Addr;	 
      state->gpu2.addr = state->prefetchData.addr;	 
      state->gpu2.stream = state->prefetchData.stream;    
      isPrefetch = true;
    }
#endif
    for (int i = state->dims-1; i >= 0; i--) {
      totalSize *= state->shape[i]+state->gpu2.loff[i]+state->gpu2.uoff[i];
      offset = offset*(state->shape[i]+state->gpu2.loff[i]+state->gpu2.uoff[i])+state->gpu2.loff[i];
    }
    if (state->gpu2.state == SInvalid && state->gpu2.Addr == NULL) {
      GPUAlloc(GPU2_THREAD,&(state->gpu2.Addr),totalSize*ElementSize(state->eleType)); 
      state->gpu2.addr = (char*)state->gpu2.Addr + offset*ElementSize(state->eleType);
    }
    //AdjustDataBlockByHaloRegion(GPU2_THREAD, state->gpu2, key.lb, key.ub,
    //    state->gpu2.loff, state->gpu2.uoff, state->shape, state->dims, state->eleType, state);

    //UpdateHaloRegion(GPU1_THREAD, state->gpu1, *state);
  }

  if (Streams[threadId][streamId] == 0) {
    Cudas[threadId].LockAndPushContext();
    CudaSafe(cudaStreamCreate(&Streams[threadId][streamId]));
    Cudas[threadId].ReleaseAndPopContext();
  }  

  if (Streams[OtherGPUThd(threadId)][streamId] == 0) {
    Cudas[OtherGPUThd(threadId)].LockAndPushContext();
    CudaSafe(cudaStreamCreate(&Streams[OtherGPUThd(threadId)][streamId]));
    Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
  }  

  cudaStream_t& destStream = Streams[threadId][streamId];
  cudaStream_t& srcStream = Streams[OtherGPUThd(threadId)][streamId];

  HRESULT hre = Redistributed(threadId,false,
      threadId==GPU1_THREAD?state->gpu1.Addr:state->gpu2.Addr,
      state,gpu1Blocks,gpu2Blocks,cpuBlocks,srcStream,destStream,opr);
  //printf("redistribute res : %d \n",hre);
#ifdef PREFETCH
  state->prefetchData.clear();
#endif

  checkCUDAError("redistributed success?");

  if(hre == -1)
    return -1;
  if(hre == SUCCESS && threadId == GPU1_THREAD) {
    //state->ValidGPU1();
    state->WaitGPU1();

    if (opr.AccessType == Operation::RONLY) {
      Cudas[threadId].LockAndPushContext();
      CudaSafe(cudaEventCreate(&state->gpu1.validEvent));
      assert(state->gpu1.stream);
      CudaSafe(cudaEventRecord(state->gpu1.validEvent, state->gpu1.stream));
      Cudas[threadId].ReleaseAndPopContext();
    }

    if (opr.AccessType == Operation::RW) {
      //printf("old cpu\n");
      state->OldCPU();
      if (state->gpu2.state != SInvalid) state->OldGPU2();
    }
    if(opr.arrayTranspose.one && opr.arrayTranspose.other){
      omprArrayTranspose(opr.arrayTranspose,threadId, state->eleType,state->dims,state);
    }  
    pInfo->base = (uint64_t)state->gpu1.addr;

    for (int i=0; i<state->dims; i++)
      pInfo->lens[i] = state->shape[i] + 
        state->gpu1.loff[i] + state->gpu1.uoff[i];

    memcpy(pInfo->dimsOffset, state->gpu1.loff, 
        sizeof(int)*DSM_MAX_DIMENSION);

    return SUCCESS;
  } else if(hre == SUCCESS && threadId == GPU2_THREAD) {
    //state->ValidGPU2();
    state->WaitGPU2();

    if (opr.AccessType == Operation::RONLY) {
      Cudas[threadId].LockAndPushContext();
      CudaSafe(cudaEventCreate(&state->gpu2.validEvent));
      assert(state->gpu2.stream);
      CudaSafe(cudaEventRecord(state->gpu2.validEvent, state->gpu2.stream));
      Cudas[threadId].ReleaseAndPopContext();
    }

    if (opr.AccessType == Operation::RW) {
      //printf("old cpu\n");
      state->OldCPU();
      if (state->gpu1.state != SInvalid) state->OldGPU1();
    }
    if(opr.arrayTranspose.one && opr.arrayTranspose.other){
      omprArrayTranspose(opr.arrayTranspose,threadId, state->eleType,state->dims,state);
    }  
    pInfo->base = (uint64_t)state->gpu2.addr;

    for (int i=0; i<state->dims; i++)
      pInfo->lens[i] = state->shape[i] + 
        state->gpu2.loff[i] + state->gpu2.uoff[i];

    memcpy(pInfo->dimsOffset, state->gpu2.loff, 
        sizeof(int)*DSM_MAX_DIMENSION);

    return SUCCESS;
  }
}


double wrTime[2];

OMPRresult omprGPUInput(DataInfo* pInfo, pid_t threadId, int streamId,
    Operation& opr, void* base, ElementType type,int dims, ...) {

  assert(threadId == GPU1_THREAD ||
      threadId == GPU2_THREAD);

  double start = gettime();
  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];


#ifdef PREFETCH
  if(threadId == GPU2_PREFETCHTHREAD || threadId == GPU1_PREFETCHTHREAD) {
    if (opr.AccessType == Operation::RONLY || opr.AccessType == Operation::RW){
      return traceMgr.prefetchInput(threadId,streamId,key,dims,type,len,loff,uoff);
    } else
      return SUCCESS;
    //return Prefetch(threadId,streamId,key);
  }
#endif

  HRESULT hre = 
    CheckAndInitDataBlockState(key,base,shape,len,lb,dims,type,loff,uoff,true);


  long totalSize=1;  //the size of the region including the shadow region
  long offset=0;	//offset between Addr and addr on gpu
  for (int i=dims-1; i>=0; i--) {
    totalSize *= shape[i]+loff[i]+uoff[i];
    offset = offset*(shape[i]+loff[i]+uoff[i])+loff[i];
  }


  if (Streams[threadId][streamId] == 0) {
    Cudas[threadId].LockAndPushContext();
    CudaSafe(cudaStreamCreate(&Streams[threadId][streamId]));
    Cudas[threadId].ReleaseAndPopContext();
  }

  cudaStream_t& stream = Streams[threadId][streamId];

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);

  if (state==NULL) {
    pthread_rwlock_rdlock(&ModifyDataState);
    DataObjState* rstate = DataState.find(key)->second;
    pthread_rwlock_unlock(&ModifyDataState);
    assert(0);
  }



  if (opr.AccessType == Operation::WONLY) {

    state->pGpus[threadId]->stream = stream;

    pthread_rwlock_wrlock(&ModifyDataState);
    for (typename DataStateType::iterator iter = DataState.begin(); 
        iter!=DataState.end(); ++iter) {
      DataObjState* tmpState = iter->second;
      if (tmpState->dims == dims && tmpState->base==state->base && tmpState!=state) {
        tmpState->AddOverrideBlock(lb, ub);
        if (tmpState->IsAllOverride()) {
          tmpState->gpu1.FreeSpace(GPU1_THREAD);
          tmpState->gpu2.FreeSpace(GPU2_THREAD);
          DataState.erase(iter);
        }
      }
    }
    pthread_rwlock_unlock(&ModifyDataState);

    if (state->pGpus[threadId]->state == SInvalid && state->pGpus[threadId]->Addr==NULL) {
      GPUAlloc((Device_Thread)threadId, &state->pGpus[threadId]->Addr, 
          totalSize*ElementSize(type));
      state->pGpus[threadId]->addr = (char*)state->pGpus[threadId]->Addr + 
        offset*ElementSize(type);
    }

    pInfo->base = (uint64_t)state->pGpus[threadId]->Addr;
    for (int i=0; i<dims; i++)
      pInfo->lens[i] = state->shape[i] + 
        state->pGpus[threadId]->loff[i] + state->pGpus[threadId]->uoff[i];
    memcpy(pInfo->dimsOffset, state->pGpus[threadId]->loff, sizeof(int)*DSM_MAX_DIMENSION);



    wrTime[threadId] += gettime()-start;
    printf("pthread write time:%lf in %d\n", wrTime[threadId], threadId);
    return SUCCESS;
  }


  //if (hre == SUCCESS){ //|| !state->prefetchData.vecPoint.empty()) {
  //this block is newly created, check if the block intersect with 
  //other in entry
  double tmp=gettime();
  if(Redistribution(pInfo,threadId,streamId,opr,key, state) == SUCCESS){
    tmp = gettime()-tmp;

    wrTime[threadId] += gettime()-start-tmp;
    printf("pthread write time:%lf in %d\n", wrTime[threadId], threadId);
    return SUCCESS;
  }
  tmp = gettime()-tmp;
  start+= tmp;


  tmp = gettime();
  if ((threadId == GPU1_THREAD && state->gpu1.haloMap==NULL) ||
      (threadId == GPU2_THREAD && state->gpu2.haloMap==NULL)) {
    InitHaloBlocks(threadId, lb, ub, loff, uoff, dims, type,state,
        len, base);
  }
  start += tmp-gettime();



  char* cpuArrayBase = (char*)base;

  if (threadId == GPU1_THREAD) {
    //prefetch data use    

#ifdef PREFETCH
    BOOL isPrefetch = false;
    if(state->prefetchData.pid == GPU1_THREAD){	 
      printf("use prefetch data\n");
      state->gpu1.FreeSpace(GPU1_THREAD);
      state->gpu1.Addr = state->prefetchData.Addr;	 
      state->gpu1.addr = state->prefetchData.addr;	 
      if(state->prefetchData.stream!=0)
        state->gpu1.stream = state->prefetchData.stream;    
      state->prefetchData.clear();
      isPrefetch = true;
    }
#endif
    //
    cudaError_t last_error;
    if (state->gpu1.state == SInvalid && state->gpu1.Addr == NULL) {
      GPUAlloc(GPU1_THREAD, &state->gpu1.Addr, totalSize*ElementSize(type));
      state->gpu1.addr = (char*)state->gpu1.Addr + offset*ElementSize(type);
    }

    tmp = gettime();
    AdjustDataBlockByHaloRegion(GPU1_THREAD, state->gpu1, lb, ub,
        loff, uoff, state->shape, dims, type, state);
    start += gettime()-tmp;

    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing alloc- %s\n",threadId,cudaGetErrorString(last_error));
      assert(0);
    }

    tmp = gettime();
    UpdateHaloRegion(GPU1_THREAD, state->gpu1, *state);
    start += gettime()-tmp;

    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing the halo region- %s\n",threadId,cudaGetErrorString(last_error));
      assert(0);
    }

    if (state->gpu1.state == SInvalid) {
#ifdef PREFETCH
      if(!isPrefetch){
#endif
        tmp = gettime();
        UpdateGPUBlock(stream, state, state->gpu1, state->cpuState,
            state->gpu2, GPU2_THREAD, cpuArrayBase);
        start += gettime()-tmp;
        checkCUDAError("update gpu block");
#ifdef PREFETCH
      }
#endif

      tmp = gettime();
      state->WaitGPU1();
      if (opr.AccessType == Operation::RONLY) {
        Cudas[threadId].LockAndPushContext();
        CudaSafe(cudaEventCreate(&state->gpu1.validEvent));
        CudaSafe(cudaEventRecord(state->gpu1.validEvent, state->gpu1.stream));
        Cudas[threadId].ReleaseAndPopContext();
      }
      if (opr.AccessType == Operation::RW)  {
        state->OldCPU();
        if (state->gpu2.state != SInvalid) state->OldGPU2();
      }
      start += gettime()-tmp;

    } else if (state->gpu1.state == SOld) {

      //nothing to do


    } else if (state->gpu1.state == SWaiting) {

      tmp = gettime();
      Cudas[GPU1_THREAD].LockAndPushContext();
      //CudaSafe(cudaEventSynchronize(state->gpu1.validEvent));
      //CudaSafe(cudaEventDestroy(state->gpu1.validEvent));
      Cudas[GPU1_THREAD].ReleaseAndPopContext();

      state->ValidGPU1();
      state->gpu1.validEvent = INVALID_EVENT;
      if (state->cpuState == SOld) state->InvalidCPU();
      if (state->gpu2.state == SOld) state->InvalidGPU2();
      start += gettime()- tmp;
    }
    tmp = gettime();
    if(opr.arrayTranspose.one && opr.arrayTranspose.other){
      omprArrayTranspose(opr.arrayTranspose,threadId, type,dims,state);
    }
    start += gettime()-tmp;
    pInfo->base = (uint64_t)state->gpu1.Addr;
    //pInfo->base = (uint64_t)state->gpu1.addr;

    for (int i=0; i<state->dims; i++)
      pInfo->lens[i] = state->shape[i] + 
        state->gpu1.loff[i] + state->gpu1.uoff[i];

    memcpy(pInfo->dimsOffset, state->gpu1.loff, 
        sizeof(int)*DSM_MAX_DIMENSION);

    wrTime[threadId] += gettime()-start;
    printf("pthread write time:%lf in %d\n", wrTime[threadId], threadId);
    return SUCCESS;

  } else if (threadId == GPU2_THREAD) {
#ifdef PREFETCH
    BOOL isPrefetch = false;
    if(state->prefetchData.pid == GPU2_THREAD){	   
      printf("use prefetch data\n");
      state->gpu2.FreeSpace(GPU2_THREAD);
      state->gpu2.Addr = state->prefetchData.Addr;	   
      state->gpu2.addr = state->prefetchData.addr;	   
      if(state->prefetchData.stream!=0)
        state->gpu2.stream = state->prefetchData.stream;    
      state->prefetchData.clear();
      isPrefetch = true;
    }
#endif
    cudaError_t last_error;
    if (state->gpu2.state == SInvalid && state->gpu2.Addr == NULL) {
      GPUAlloc(GPU2_THREAD, &state->gpu2.Addr, totalSize*ElementSize(type));
      state->gpu2.addr = (char*)state->gpu2.Addr + offset*ElementSize(type);
    }

    tmp = gettime();
    AdjustDataBlockByHaloRegion(GPU2_THREAD, state->gpu2, lb, ub,
        loff, uoff, state->shape, dims, type, state);
    start += gettime()-tmp;

    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing the before alloc- %s\n",threadId,cudaGetErrorString(last_error));
      assert(0);
    }
    tmp = gettime();
    UpdateHaloRegion(GPU2_THREAD, state->gpu2, *state);
    start += gettime()-tmp;
    last_error = cudaGetLastError();
    if(last_error != cudaSuccess)
    {
      printf("pid : %d An error happened when executing the halo region- %s\n",threadId,cudaGetErrorString(last_error));
      assert(0);
    }

    if (state->gpu2.state == SInvalid) {
#ifdef PREFETCH
      if(!isPrefetch){
#endif
        //printf("no error\n");
        tmp = gettime();
        UpdateGPUBlock(stream, state, state->gpu2, state->cpuState,
            state->gpu1, GPU1_THREAD, cpuArrayBase);
        start += gettime()-tmp;
        checkCUDAError("update gpu");
#ifdef PREFETCH
      }
#endif

      tmp = gettime();
      state->WaitGPU2();
      if (opr.AccessType == Operation::RONLY) {
        Cudas[threadId].LockAndPushContext();
        CudaSafe(cudaEventCreate(&state->gpu2.validEvent));
        CudaSafe(cudaEventRecord(state->gpu2.validEvent, state->gpu2.stream));
        Cudas[threadId].ReleaseAndPopContext();
      }

      if (opr.AccessType == Operation::RW) {
        state->OldCPU();
        if (state->gpu1.state != SInvalid) state->OldGPU1();
      }
      start += gettime()-tmp;
    } else if (state->gpu2.state == SOld) {

      //nothing to do
      /*
         UpdateGPUBlock(stream, state, state->gpu2, state->cpuState,
         state->gpu1, GPU1_THREAD, cpuArrayBase);
       */

    } else if (state->gpu2.state == SWaiting) {

      tmp =gettime();
      Cudas[GPU2_THREAD].LockAndPushContext();
      //CudaSafe(cudaEventSynchronize(state->gpu2.validEvent));
      //CudaSafe(cudaEventDestroy(state->gpu2.validEvent));
      Cudas[GPU2_THREAD].ReleaseAndPopContext();

      state->ValidGPU2();
      state->gpu2.validEvent = INVALID_EVENT;
      if (state->cpuState == SOld) state->InvalidCPU();
      if (state->gpu1.state == SOld) state->InvalidGPU1();
      start += gettime()-tmp;
    }

    tmp = gettime();
    if(opr.arrayTranspose.one && opr.arrayTranspose.other){
      omprArrayTranspose(opr.arrayTranspose,threadId, type,dims,state);
    }
    start += gettime()-tmp;
    pInfo->base = (uint64_t)state->gpu2.Addr;

    for (int i=0; i<state->dims; i++)
      pInfo->lens[i] = state->shape[i] + 
        state->gpu2.loff[i] + state->gpu2.uoff[i];

    memcpy(pInfo->dimsOffset, state->gpu2.loff, 
        sizeof(int)*DSM_MAX_DIMENSION);

    wrTime[threadId] += gettime()-start;
    printf("pthread write time:%lf in %d\n", wrTime[threadId], threadId);
    return SUCCESS;

  } else {
    assert(0);
    //UpdateCPUBlock(state, cpuArrayBase);
    //return *itr;
  }

  wrTime[threadId] += gettime()-start;
  printf("pthread write time:%lf in %d\n", wrTime[threadId], threadId);
  return FAILD_UNKNOWN;

}


OMPRresult omprGPUOutput(pid_t threadId, int streamId, Operation& opr, 
    void* base, ElementType type, int dims, ...) {

  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];

#ifdef PREFETCH
  if(threadId == GPU2_PREFETCHTHREAD || threadId == GPU1_PREFETCHTHREAD) {
    if (opr.AccessType==Operation::RW || opr.AccessType==Operation::WONLY){
      return traceMgr.prefetchOutput(threadId,streamId,key,dims,type,len,loff,uoff);
    } else
      return SUCCESS;
  }
#endif

  CheckAndInitDataBlockState(key,base,shape,len,lb,dims,type,loff,uoff,true);

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);

  assert(state);
  if (threadId == GPU1_THREAD ) {

    if (state->gpu1.state == SInvalid) {
      if (opr.AccessType == Operation::RW) {

        Cudas[GPU1_THREAD].LockAndPushContext();
        if (state->gpu1.validEvent != INVALID_EVENT)
          CudaSafe(cudaEventDestroy(state->gpu1.validEvent));

        CudaSafe(cudaEventCreate(&state->gpu1.validEvent));
        CudaSafe(cudaEventRecord(state->gpu1.validEvent,
              state->gpu1.stream));
        Cudas[GPU1_THREAD].ReleaseAndPopContext();

        state->WaitGPU1();
      } 
    } else if (state->gpu1.state == SOld) {
      if (opr.AccessType == Operation::RW) 
        assert(0);
    } else if (state->gpu1.state == SWaiting) {
      if (opr.AccessType == Operation::RW) {

        Cudas[GPU1_THREAD].LockAndPushContext();
        if (state->gpu1.validEvent != INVALID_EVENT)
          CudaSafe(cudaEventDestroy(state->gpu1.validEvent));
        CudaSafe(cudaEventCreate(&state->gpu1.validEvent));
        CudaSafe(cudaEventRecord(state->gpu1.validEvent,
              state->gpu1.stream));
        Cudas[GPU1_THREAD].ReleaseAndPopContext();

        state->OldCPU();
        if (state->gpu2.state != SInvalid) state->OldGPU2();

      }
    } else if (state->gpu1.state == SValid) {
      if (opr.AccessType == Operation::RW) {
        Cudas[GPU1_THREAD].LockAndPushContext();
        if (state->gpu1.validEvent != INVALID_EVENT)
          CudaSafe(cudaEventDestroy(state->gpu1.validEvent));
        CudaSafe(cudaEventCreate(&state->gpu1.validEvent));
        CudaSafe(cudaEventRecord(state->gpu1.validEvent,
              state->gpu1.stream));
        Cudas[GPU1_THREAD].ReleaseAndPopContext();
        state->InvalidGPU1();
        state->WaitGPU1();
        state->InvalidCPU();
        state->InvalidGPU2();

      }
    }

    //return state->gpu1.addr;
  } else if (threadId == GPU2_THREAD) {
    if (state->gpu2.state == SInvalid) {
      if (opr.AccessType == Operation::RW) {

        Cudas[GPU2_THREAD].LockAndPushContext();
        //if (state->gpu2.validEvent != INVALID_EVENT)
        //  CudaSafe(cudaEventDestroy(state->gpu2.validEvent));
        CudaSafe(cudaEventCreate(&state->gpu2.validEvent));
        CudaSafe(cudaEventRecord(state->gpu2.validEvent,
              state->gpu2.stream));
        Cudas[GPU2_THREAD].ReleaseAndPopContext();

        state->WaitGPU2();
      }
    } else if (state->gpu2.state == SOld) {
      if (opr.AccessType == Operation::RW)
        assert(0);
    } else if (state->gpu2.state == SWaiting) {
      if (opr.AccessType == Operation::RW) {

        Cudas[GPU2_THREAD].LockAndPushContext();
        if (state->gpu2.validEvent != INVALID_EVENT)
          CudaSafe(cudaEventDestroy(state->gpu2.validEvent));
        CudaSafe(cudaEventCreate(&state->gpu2.validEvent));
        CudaSafe(cudaEventRecord(state->gpu2.validEvent,
              state->gpu2.stream));
        Cudas[GPU2_THREAD].ReleaseAndPopContext();

        state->OldCPU();
        if (state->gpu1.state != SInvalid) state->OldGPU1();

      }
    } else if (state->gpu2.state == SValid) {
      if (opr.AccessType == Operation::RW) {
        Cudas[GPU2_THREAD].LockAndPushContext();
        if (state->gpu2.validEvent == INVALID_EVENT)
          CudaSafe(cudaEventCreate(&state->gpu2.validEvent));
        CudaSafe(cudaEventRecord(state->gpu2.validEvent,
              state->gpu2.stream));
        Cudas[GPU2_THREAD].ReleaseAndPopContext();
        //state->InvalidGPU2();
      }
      if (opr.AccessType == Operation::RW ||
          opr.AccessType == Operation::WONLY) {
        state->WaitGPU2();

        state->InvalidCPU();
        state->InvalidGPU1();
      }
    }
    //return state->gpu2.addr;

  } else {
    //assert(0);
  }

  return SUCCESS;
}


OMPRresult omprCPUInput (DataInfo* pInfo, pid_t threadId, Operation& opr, 
    void* base, ElementType type,int dims, ...) {

  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];

  CheckAndInitDataBlockState(key,base,shape,len,lb,dims,type,loff,uoff,true);

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);

  if (opr.AccessType == Operation::WONLY)
    return SUCCESS;

  char* cpuArrayBase = (char*)base;

  UpdateCPUBlock(state, cpuArrayBase);

  int haloLoffL[DSM_MAX_DIMENSION], haloLoffU[DSM_MAX_DIMENSION],
      haloUoffL[DSM_MAX_DIMENSION], haloUoffU[DSM_MAX_DIMENSION];

  for (int i=0; i<dims; i++) {
    haloLoffL[i] = MAX(lb[i]-loff[i],0); haloLoffU[i] = lb[i]-1;
    haloUoffL[i] = ub[i]+1; haloUoffU[i] = MIN(MAX(ub[i]+uoff[i],0),len[i]-1);
  }
  std::vector<HaloBlock*>* list = GetHaloRegions(dims, haloLoffL, 
      haloLoffU, haloUoffL, haloUoffU, lb, ub);
  for (std::vector<HaloBlock*>::iterator itr = list->begin();
      itr!=list->end(); ++itr)
    (*itr)->base = base;

  for (std::vector<HaloBlock*>::iterator itr = list->begin();
      itr!=list->end(); ++itr) {
    pthread_rwlock_rdlock(&ModifyDataState);
    DataStateType::iterator dataPair = DataState.begin();
    DataStateType::iterator end = DataState.end();
    pthread_rwlock_unlock(&ModifyDataState);
    while (dataPair != end) {
      const BlockKey* key = &dataPair->first;
      if (key->base == base && in(const_cast<BlockKey*>(key), *itr, dims)) {
        DataObjState* state = dataPair->second;
        if (state->cpuState == SValid) {
          //nothing to do
        } else  {
          void* src;
          int gpuDimsLen[DSM_MAX_DIMENSION];
          int shape[DSM_MAX_DIMENSION];
          int gpuDimsOffset[DSM_MAX_DIMENSION];
          int gpuThreadId;

          if (state->gpu1.state == SValid ||
              state->gpu1.state == SWaiting) {
            gpuThreadId = GPU1_THREAD;
            //if (state->gpu1.state == SWaiting)
            //CudaSafe(cudaEventSynchronize(state->gpu1.validEvent));
            src = state->gpu1.Addr;
            for (int i=0; i<dims; i++) {
              gpuDimsLen[i] = state->shape[i] + state->gpu1.loff[i] +
                state->gpu1.uoff[i];
              shape[i] = (*itr)->ub[i] - (*itr)->lb[i]+1;
              gpuDimsOffset[i] = (*itr)->lb[i] - (state->dimsOffset[i] -
                  state->gpu1.loff[i]);
            }

          } else if (state->gpu2.state == SValid ||
              state->gpu2.state == SWaiting) {
            gpuThreadId = GPU2_THREAD;
            //if (state->gpu2.state == SWaiting)
            //CudaSafe(cudaEventSynchronize(state->gpu2.validEvent));
            src = state->gpu2.Addr;
            for (int i=0; i<dims; i++) {
              gpuDimsLen[i] = state->shape[i] + state->gpu2.loff[i] +
                state->gpu2.uoff[i];
              shape[i] = (*itr)->ub[i] - (*itr)->lb[i] +1;
              gpuDimsOffset[i] = (*itr)->lb[i] - (state->dimsOffset[i] -
                  state->gpu2.loff[i]);
            }

          } else {
            printf("halo region on cpu haven't valid value\n");
            assert(0);
          }

          Cudas[gpuThreadId].LockAndPushContext();
          cudaStream_t tmp;
          TransferData(FALSE, tmp, dims, state->dimsLen, gpuDimsLen,
              shape, (*itr)->lb, gpuDimsOffset, src, base, state->eleType, 
              cudaMemcpyDeviceToHost);
          Cudas[gpuThreadId].ReleaseAndPopContext();
        }

        break;
      }
      pthread_rwlock_rdlock(&ModifyDataState);
      ++dataPair;
      pthread_rwlock_unlock(&ModifyDataState);
    }//if not found, so the valid value must be on cpu since gpu 
    //thread never modify it.

  }

  long offset=lb[dims-1];
  for (int i=1; i<dims; i++) {
    offset = offset*len[dims-i-1]+lb[dims-1-i];
  }

  pInfo->base = (uint64_t)((char*)base+offset*ElementSize(type));
  memcpy(pInfo->lens, state->shape, sizeof(int)*DSM_MAX_DIMENSION);
  memcpy(pInfo->dimsOffset, state->dimsOffset, sizeof(int)*DSM_MAX_DIMENSION);
  return SUCCESS;

}


OMPRresult omprCPUOutput (pid_t threadId, Operation& opr, void* base, 
    ElementType type,int dims, ...) {

  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];

  CheckAndInitDataBlockState(key,base,shape,len,lb,dims,type,loff,uoff,true);
  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);

  int cpuOrigOff = state->dimsOffset[state->dims-1];
  for (int i=state->dims-1; i>0; i--) {
    cpuOrigOff = cpuOrigOff*state->dimsLen[i] + state->dimsOffset[i-1];
  }
  char* cpuArrayBase  = (char*)base;

  state->ValidCPU();
  if (opr.AccessType == Operation::RW ||
      opr.AccessType == Operation::WONLY) {
    state->InvalidGPU1();
    state->InvalidGPU2();
  }

  return SUCCESS;
}


OMPRresult omprCudaKernelCall(pid_t threadId, int streamId, 
    CallKernelFuncType func) {

  assert(threadId==GPU1_THREAD || threadId==GPU2_THREAD);

  Cudas[threadId].LockAndPushContext();

  if (Streams[threadId][streamId] == INVALID_STREAM) 
    CudaSafe(cudaStreamCreate(&Streams[threadId][streamId]));

  func(threadId, Streams[threadId][streamId]);
  Cudas[threadId].ReleaseAndPopContext();

  traceMgr.agentBarrierInput(threadId);

  return SUCCESS;
}

OMPRresult omprGPUSynchronizeA(pid_t threadId, void* base, ElementType type, int dims, ...) {

  if(threadId == GPU1_PREFETCHTHREAD || threadId == GPU2_PREFETCHTHREAD)
    return SUCCESS;

  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);
  assert(state);


  BlockOnGPUState* gpuState;
  if (threadId == GPU1_THREAD)
    gpuState = &state->gpu1;
  else
    gpuState = &state->gpu2;

  if (gpuState->state == SWaiting) {
    Cudas[threadId].LockAndPushContext();
    //checkCUDAError("before sync");
    CudaSafe(cudaEventSynchronize(gpuState->validEvent));
    //CudaSafe(cudaEventDestroy(gpuState->validEvent));
    //gpuState->validEvent = INVALID_EVENT;
    Cudas[threadId].ReleaseAndPopContext();

    if (threadId == GPU1_THREAD) {

      state->ValidGPU1();

      if (state->gpu2.state == SOld)
        state->InvalidGPU2();
      if (state->cpuState == SOld)
        state->InvalidCPU();
    } else if (threadId == GPU2_THREAD) {

      state->ValidGPU2();

      if (state->gpu1.state == SOld)
        state->InvalidGPU1();
      if (state->cpuState == SOld)
        state->InvalidCPU();
    }

  }
  return SUCCESS;

}


OMPRresult omprGPUSynchronize(pid_t threadId) {

  if(threadId == GPU1_PREFETCHTHREAD || threadId == GPU2_PREFETCHTHREAD)
    return SUCCESS;

  //int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];


  pthread_rwlock_rdlock(&ModifyDataState);
  DataStateType::iterator itr = DataState.begin();
  DataStateType::iterator end = DataState.end();
  pthread_rwlock_unlock(&ModifyDataState);
  while (itr != end) {
    BlockOnGPUState* gpuState;
    DataObjState* state = itr->second;
    if (threadId == GPU1_THREAD)
      gpuState = &itr->second->gpu1;
    else
      gpuState = &itr->second->gpu2;

    if (gpuState->state == SWaiting) {
      Cudas[threadId].LockAndPushContext();
      //CudaSafe(cudaEventSynchronize(gpuState->validEvent));
      //CudaSafe(cudaEventDestroy(gpuState->validEvent));
      //gpuState->validEvent = INVALID_EVENT;
      Cudas[threadId].ReleaseAndPopContext();

      if (threadId == GPU1_THREAD) {
        itr->second->ValidGPU1();
        if (itr->second->gpu2.state == SOld)
          itr->second->InvalidGPU2();
        if (itr->second->cpuState == SOld)
          itr->second->InvalidCPU();
      } else if (threadId == GPU2_THREAD) {
        itr->second->ValidGPU2();
        if (itr->second->gpu1.state == SOld)
          itr->second->InvalidGPU1();
        if (itr->second->cpuState == SOld)
          itr->second->InvalidCPU();
      }

    }

    pthread_rwlock_rdlock(&ModifyDataState);
    ++itr;
    pthread_rwlock_unlock(&ModifyDataState);
  }


  return SUCCESS;
}

OMPRresult prop_barrier_match(pid_t threadId)
{
#ifdef PREFETCH
  if(threadId < 2)
    traceMgr.agentBarrierInput(threadId);
  else 
    traceMgr.prefetchBarrierInput(threadId);
#endif

  return SUCCESS;
}

OMPRresult omprGPUDataUseless(pid_t threadId, void* base, ElementType type, 
    int dims, ...) {

  if (threadId >= GPU_NUM) return SUCCESS;

  int ub[DSM_MAX_DIMENSION], lb[DSM_MAX_DIMENSION],len[DSM_MAX_DIMENSION];
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
  int shape[DSM_MAX_DIMENSION];
  DSM_FUNC_INIT(dims);

  BlockKey key;
  memset(key.lb,0,sizeof(key.lb));
  memset(key.ub,0,sizeof(key.ub));
  key.base=base;
  for (int i=0; i<dims; i++)
    key.lb[i]=lb[i], key.ub[i]=ub[i];

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* state = DataState.find(key)->second;
  pthread_rwlock_unlock(&ModifyDataState);
  assert(state);

  DevMemManager[threadId]->freeBlock(state->pGpus[threadId]->Addr);
  state->WriteLock();
  state->pGpus[threadId]->Addr = state->pGpus[threadId]->addr =NULL;
  state->RWUnlock();
  state->InvalidGPU(threadId);

  return SUCCESS;
}


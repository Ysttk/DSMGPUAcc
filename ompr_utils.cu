#include "ompr_dsm.h"
#include "ompr_utils.h"
#include "ompr_heap.h"
#include <vector>
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

extern DataStateType DataState;
extern pthread_mutex_t modifyMap;
extern pthread_rwlock_t ModifyDataState;
extern CPUMemoryManager* CPUMemManager;

int ElementSize(ElementType type) {
  switch (type) {
    case INT8: return 1;
    case INT16: return 2;
    case FLOAT:
    case INT32: return 4;
    case DOUBLE:
    case INT64: return 8;
  }
}

/*
Arg:
base: base address of the array;
shape:	shape of each block, here assume each block is the same shape;
offset:  offset of the block;
lens: length of each dim of the array;
dims: dimention of the array;
eType: element type of the array;
loff: lower offset (halo region) of the block/area;
uoff: upper offset ...

function return SUCCESS if the entry was newly created, else return
FAILD;
*/
HRESULT CheckAndInitDataBlockState(BlockKey& key,void* base,int* shape,int* lens,
    int* offset,int dims,ElementType eType,int* loff,int* uoff,bool initialize ) {
//HRESULT CheckAndInitDataBlockState(pid_t pid, BlockKey& key,void* base,int* shape,int* lens,
//    int* offset,int dims,ElementType eType,int* loff,int* uoff,bool initialize ) {

    DataStateType::iterator dataPair = DataState.find(key);
    bool result = (dataPair == DataState.end());
    if (result) {
      //printf("create a new menu\n");
      pthread_rwlock_wrlock(&ModifyDataState);
      //pthread_mutex_lock(&modifyMap);
      //std::pair< DataStateType::iterator,bool > ret;
      //void* tmpSpace =  malloc(sizeof(DataObjState));
      void* tmpSpace =  CPUMemManager->allocBlock(sizeof(DataObjState));
      DataObjState* tmp = new (tmpSpace) DataObjState(base, dims, shape, lens, offset);
      std::pair<BlockKey, DataObjState*> value(key, tmp);
      //DataState.insert(std::pair<BlockKey,DataObjState*>(key,new DataObjState(base, dims,
      //        shape, lens, offset)));
      DataState.insert(value);
      //if(ret.second){printf("success\n");}
      //pthread_mutex_unlock(&modifyMap);
      pthread_rwlock_unlock(&ModifyDataState);
    }


    pthread_rwlock_rdlock(&ModifyDataState);
    dataPair = DataState.find(key);
    DataObjState* state= dataPair->second;
    pthread_rwlock_unlock(&ModifyDataState);

    if ( !state->IsInitialized() && initialize) {

      state->base = base;
      memcpy(state->shape, shape, sizeof(int)*dims);
      memcpy(state->dimsLen, lens, sizeof(int)*dims);
      memcpy(state->dimsOffset, offset, sizeof(int)*dims);
      int size=1;
      for (int i=0; i<dims; i++) {
        size *= shape[i];
      }
      state->dims = dims; state->size = size;state->eleType = eType;
      state->cpuState = SValid;
      state->gpu1.state = state->gpu2.state = SInvalid;
      state->gpu1.validEvent = state->gpu2.validEvent = INVALID_EVENT;
      state->gpu1.stream = state->gpu2.stream = INVALID_STREAM;
      state->gpu1.addr = state->gpu2.addr = NULL;
      state->gpu1.Addr = state->gpu2.Addr = NULL;
      memcpy(state->gpu1.loff, loff, sizeof(int)*dims);
      memcpy(state->gpu1.uoff, uoff, sizeof(int)*dims);
      memcpy(state->gpu2.loff, loff, sizeof(int)*dims);
      memcpy(state->gpu2.uoff, uoff, sizeof(int)*dims);

      for (HaloBlockListType::iterator itr = FoldHaloBlocks.begin();
          itr!=FoldHaloBlocks.end(); ++itr) {
        const HaloBlock* haloB = &itr->first;

        if (key.base == haloB->base &&
            in(&key, const_cast<HaloBlock*>(&itr->first), dims)&&
            itr->second != NULL) {
          HaloState* haloState = itr->second;
          haloState->subject = state;
          state->Registe(haloState);
          haloState->validBlockKey = new BlockKey();
          memcpy(haloState->validBlockKey, &key, sizeof(BlockKey));

          //FoldHaloBlocks.erase(itr);

          itr->second=NULL;
        }
      }

      state->SetInitialized();
      return SUCCESS;
    }

    return ALREADY_INITED;
}


void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err)
  {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    //exit(-1);
    assert(0);
  }
}

double gettime()
{
  struct timeval sampleval;
  double time;
  gettimeofday(&sampleval, NULL);
  time = sampleval.tv_sec + (sampleval.tv_usec/1000000.0);
  return (time);
}


pid_t getPrefetchIdFromAgentId(pid_t threadId)
{
    if(threadId == GPU1_THREAD)
        return GPU1_PREFETCHTHREAD;
    else if(threadId == GPU2_THREAD)
        return GPU2_PREFETCHTHREAD;
    else return -1;
}


pid_t getAgentIdFromPrefetchId(pid_t threadId)
{
    if(threadId == GPU1_PREFETCHTHREAD)
        return GPU1_THREAD;
    else if(threadId == GPU2_PREFETCHTHREAD)
        return GPU2_THREAD;
    else return -1;
}

bool isAgentId(pid_t threadId)
{
    if(threadId == GPU1_THREAD || threadId == GPU2_THREAD)
        return true;

    return false;
}

bool isPrefetchId(pid_t threadId)
{
    if(threadId == GPU1_PREFETCHTHREAD|| threadId == GPU2_PREFETCHTHREAD)
        return true;

    return false;
}

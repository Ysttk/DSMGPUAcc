#ifndef OMPR_DSM_H
#define OMPR_DSM_H
#include "omp_runtime_global.h"
#include <pthread.h>
#include <map>
#include <list>
#include <cuda.h>
#include <stdio.h>
//#include <sys/cache.h>

#define NO_UNION_BETWEEN_RECS LAST_ERROR-1
#define NOT_CONTAINED LAST_ERROR-2
#define CONTAINED LAST_ERROR-3
#define NOT_EQUAL LAST_ERROR-4

#include "ompr_utils.h"


#define OMPR_MAX_HELP_THREADS OMPR_GPU_THREAD_NUM+OMPR_CPU_THREAD_NUM
#define OMPR_GPU_THREAD_NUM GPU_NUM
#define OMPR_CPU_THREAD_NUM 2
#define OMPR_AGENT_THREAD_NUM \
  (OMPR_CPU_THREAD_NUM+OMPR_GPU_THREAD_NUM)
#define DSM_MAX_PTHREADS 6
#define DSM_MAX_STREAMS  100

#define INVALID_EVENT   0
#define INVALID_STREAM  0

extern pthread_mutex_t cudaFuncLock;

class DataObjState;
class BlockOnGPUState;

typedef enum {SInvalid=0, SValid, SOld, SWaiting, SumState} State;

inline char* StateString(State state) {
  switch (state) {
    case SInvalid: return "Invalid";
    case SValid: return "Valid";
    case SOld: return "Old";
    case SWaiting: return "Waiting";
  }
  return NULL;
}

typedef enum {Invalid_Device=-1,  GPU1, GPU2, CPU, DeviceSum} Device_Type;
typedef enum { 
  Invalid_Thread=-1,
  GPU1_THREAD, 
  GPU2_THREAD , 
  CPU_THREAD, 
  GPU1_PREFETCHTHREAD,
  GPU2_PREFETCHTHREAD
} Device_Thread;
/*GPU_Thread will be deprected in the furture, use Device_Thread instead.*/
typedef Device_Thread GPU_Thread; 

inline Device_Type DeviceThread2Type(Device_Thread threadId) {
  if (static_cast<Device_Type>(threadId) < DeviceSum) 
    return static_cast<Device_Type>(threadId);
  else if (static_cast<Device_Type>(threadId-DeviceSum) < DeviceSum)
    return static_cast<Device_Type>(threadId-DeviceSum);
  else
    return Invalid_Device;
}

inline Device_Thread DeviceType2Thread(Device_Type type) {
  if (type != Invalid_Device &&
      type != DeviceSum)
    return static_cast<Device_Thread>(type);
  else
    return Invalid_Thread;
}

class IRectangleBlock {
  public:
    virtual int LB(int idx)=0;
    virtual int UB(int idx)=0;
};

bool in(IRectangleBlock* large,IRectangleBlock* small,int dims);

class HaloBlock : public IRectangleBlock{
public:
  void* base;
  int lb[DSM_MAX_DIMENSION],
      ub[DSM_MAX_DIMENSION];
  //valid halo region == lb - ub +1, 
  //that is begin with lb, end at ub 
  //these two position is relavent to array base on cpu

public:
  HaloBlock() {
  }

  int LB(int idx) { return lb[idx];}
  int UB(int idx) { return ub[idx];}

};

class IHaloRegionObserver;
class BlockKey;

typedef struct {
  volatile State state;
  BlockKey* validBlockKey;
  //int offset[DSM_MAX_DIMENSION];
    //offset of the first element of each dim on source block 
    //relavent to source block base address(BlockOnGPUState.addr)
  IHaloRegionObserver* subject;
    //if NULL, then the block contain valid value is only on CPU
} HaloState;

class IHaloRegionObserver {
public:
  virtual HRESULT UnRegiste(HaloState* state)=0;
  virtual HRESULT Registe(HaloState* state)=0; 
};

typedef std::map<HaloBlock, HaloState*> HaloMapType;


extern HRESULT GPUFree(Device_Thread thread, void* Addr);

class BlockOnGPUState : public IHaloRegionObserver {

private:
  DataObjState* dataObjRef;
  BlockOnGPUState* opposeDeviceRef;

public:
  volatile State state;
  void* addr; //record the base addr of the block, if NULL then not alloced
  void* Addr; //record the base addr of the block including shadow region
		  //only valid when addr!=NULL
  cudaStream_t stream;      //record cuda stream
  cudaEvent_t validEvent;   //enable only state==SOld
  int loff[DSM_MAX_DIMENSION], uoff[DSM_MAX_DIMENSION];
    //offset of shadow region (or halo region)
    //loff: the low offset of each dimention, for ex: loff of -1 is 1
    //uoff: the up offset of each dimention, 

  std::vector<HaloState*>* haloRegionObserverList;
    //these halo region obtain the value of this cpy

  HaloMapType* haloMap;
    //this cpy have these halo region
  
  HRESULT UnRegiste(HaloState* state);
  HRESULT Registe(HaloState* state);
  

  HRESULT InvalidHaloRegionObserver() {
    for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
	    itr!=haloRegionObserverList->end(); ++itr) {
      (*itr)->state = SInvalid;
    }
    
    return SUCCESS;
  }

private:
  OMPRresult (BlockOnGPUState::*(getDataFuncList[SumState]))(Device_Thread, Operation&);
  //GetDataFuncType getDataFuncList[SumState];
  OMPRresult getDataInvalid(Device_Thread threadId, Operation& opr);
  OMPRresult getDataOld(Device_Thread threadId, Operation& opr);
  OMPRresult getDataWaiting(Device_Thread threadId, Operation& opr);
  OMPRresult getDataValid(Device_Thread threadId, Operation& opr);


public:
  OMPRresult getDataObjOnGPU(Device_Thread threadId, Operation& opr, int loff[], int uoff[],
      int shape[], int dims, DataInfo* pInfo);

  BlockOnGPUState(DataObjState* ref, BlockOnGPUState* gpuRef):
    dataObjRef(ref),opposeDeviceRef(gpuRef) 
  {
    state = SInvalid;
    haloMap = NULL; 
    haloRegionObserverList = new std::vector<HaloState*>();

    getDataFuncList[SInvalid] = &BlockOnGPUState::getDataInvalid;
    getDataFuncList[SValid] = &BlockOnGPUState::getDataValid;
    getDataFuncList[SOld] = &BlockOnGPUState::getDataOld;
    getDataFuncList[SWaiting] = &BlockOnGPUState::getDataWaiting;
  }

  void FreeSpace(Device_Thread threadId);

  ~BlockOnGPUState() {
    for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
        itr != haloRegionObserverList->end(); ++itr)
      delete *itr;

    delete haloRegionObserverList;
    delete haloMap;
  }


  void ClearHaloMap() {
    for (HaloMapType::iterator itr = haloMap->begin();
        itr!=haloMap->end(); ++itr) {
      HaloState* state = itr->second;
      state->subject->UnRegiste(state);
      delete state;
    }
    delete haloMap;
    haloMap = NULL;
  }


};


//data prefetch's status
typedef enum {START = 0, RUNNING, COMPLETE} PrefetchState;

//prefetch's data
struct RedistributionPoint{
	int offset[DSM_MAX_DIMENSION];
	int shape[DSM_MAX_DIMENSION];
        RedistributionPoint(){
            memset(offset,0,sizeof(offset));
            memset(shape,0,sizeof(shape));
        }
        bool operator ==(RedistributionPoint b)
        {
           for(int i = 0; i < DSM_MAX_DIMENSION; i++){
               if(this->offset[i] != b.offset[i]){
                   return false;
               } 
               if(this->offset[i] != b.offset[i]){
                   return false;
               }
           } 
           return true;
        }
};
struct PrefetchObjState {
	int pid;
	void* Addr;
	void* addr;
	PrefetchState status;
	cudaStream_t stream;
	std::vector<RedistributionPoint> vecPoint;
	PrefetchObjState() {
		pid = -1;
		Addr = NULL;
		addr = NULL;
		status = START;
		stream = NULL;
		vecPoint.clear();
	};
	void clear(){
		pid = -1;
		Addr = NULL;
		addr = NULL;
		status = START;
		stream = NULL;
		vecPoint.clear();
	};
	~PrefetchObjState() { 	
		pid = -1;
		Addr = NULL;
		addr = NULL;
		status = START;
		stream = NULL;
		vecPoint.clear();
	};
};

class BlankBlock {
  private:
    int lb[DSM_MAX_DIMENSION], ub[DSM_MAX_DIMENSION];

  public:
    BlankBlock(int alb[], int aub[]) {
      memcpy(lb, alb, sizeof(int)*DSM_MAX_DIMENSION);
      memcpy(ub, aub, sizeof(int)*DSM_MAX_DIMENSION);
    }

    BlankBlock(BlankBlock* block) {
      for (int i=0; i<DSM_MAX_DIMENSION; i++)
        lb[i] = block->LB(i), ub[i]=block->UB(i);
    }

    BlankBlock() {}


    int LB(int idx) {
      return lb[idx];
    }

    int UB(int idx) {
      return ub[idx];
    }

    void SetDimBound(int idx, int alb, int aub) {
      lb[idx]=alb; ub[idx]=aub;
    }

    void SetAllDimBound(int alb[], int aub[]) {
      memcpy(lb, alb, sizeof(int)*DSM_MAX_DIMENSION);
      memcpy(ub, aub, sizeof(int)*DSM_MAX_DIMENSION);
    }
};

class DataObjState : public IHaloRegionObserver {

  public:
    //cpu block info
    volatile State cpuState;
    void* base; //the base address of the array on cpu

  std::vector<HaloState*>* haloRegionObserverList;
    //these halo region obtain the value of this cpy
  HRESULT InvalidHaloRegionObserver() {
    for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
        itr!=haloRegionObserverList->end(); ++itr) {
      (*itr)->state = SInvalid;
    }
    
    return SUCCESS;
  }

  HRESULT UnRegiste(HaloState* state);
  HRESULT Registe(HaloState* state);

  bool initialized; 
    //if the block state was created with other halo region,
    //then uninitialized, or initialized

  BlockOnGPUState gpu1,gpu2;
  BlockOnGPUState* pGpus[DeviceSum-1];
  PrefetchObjState prefetchData;

  int dims,size;
  ElementType eleType;
  int shape[DSM_MAX_DIMENSION];   //block shape 
  int dimsLen[DSM_MAX_DIMENSION]; 
    //len of each dim of the array on cpu
  int dimsOffset[DSM_MAX_DIMENSION]; 
    //offset of each dim of the array on cpu
    //not include halo region

private:
  pthread_mutex_t modifyState, 
		  initLock; //locked when uninitialized

  pthread_rwlock_t dataObjLock;

  /* used in redistribution, record which part of the block has not been copied,
   * if all block has been copied, remove this dsm entry.
   */
  pthread_mutex_t ModifyBlankBlockList; 
  std::vector<BlankBlock*> *blankBlockList;
  

private:
  OMPRresult getDataObjOnCPU(Device_Thread threadId, Operation& opr, int loff[], int uoff[],
      int shape[], int dims, DataInfo* pInfo);

public:
  OMPRresult AddCoverBlock(int lb[],int ub[]);
  OMPRresult AddRedistributionBlock(int lb[], int ub[]);
  OMPRresult AddOverrideBlock(int lb[], int ub[]);

public:
  DataObjState(void* abase, int adims, int* ashape, int* alens, int* aoffset):
    gpu1(this,&gpu2),gpu2(this,&gpu1) {
    base = abase; dims = adims;
    memcpy(shape, ashape, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(dimsOffset, aoffset, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(dimsLen, alens, sizeof(int)*DSM_MAX_DIMENSION);

    //modifyState = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_init(&modifyState,NULL);
    initialized = false;
    pthread_mutex_init(&initLock, NULL);
    pthread_mutex_lock(&initLock);
    haloRegionObserverList = new std::vector<HaloState*>();
    cpuState = SValid;

    pthread_rwlock_init(&dataObjLock,NULL);

    blankBlockList = new std::vector<BlankBlock*>();
    int ub[DSM_MAX_DIMENSION];
    for (int i=0; i<dims; i++)
      ub[i] = dimsOffset[i]+shape[i]-1;
    blankBlockList->push_back(new BlankBlock(dimsOffset, ub));

    pGpus[0] = &gpu1; pGpus[1]=&gpu2;
  }

  ~DataObjState() {
    for (std::vector<HaloState*>::iterator itr = haloRegionObserverList->begin();
        itr != haloRegionObserverList->end(); ++itr)
      delete *itr;
    for (std::vector<BlankBlock*>::iterator itr = blankBlockList->begin();
        itr != blankBlockList->end(); ++itr)
      delete *itr;
    delete blankBlockList;
    delete haloRegionObserverList;

    pthread_rwlock_destroy(&dataObjLock);
    pthread_mutex_destroy(&modifyState);
    pthread_mutex_destroy(&initLock);
  }

  bool IsAllRedistributed() { return blankBlockList->empty();}
  bool IsAllOverride() { return blankBlockList->empty(); }


  OMPRresult ReadLock() {
    int re;
    if ((re=pthread_rwlock_rdlock(&dataObjLock))!=0) {
      fprintf(stderr, "rwlock unexpected error at ReadLock!\n");
      assert(0);
    }
    return SUCCESS;
  }

  OMPRresult WriteLock() {
    int re;
    if ((re=pthread_rwlock_wrlock(&dataObjLock))!=0) {
      fprintf(stderr, "rwlock unexpected error at WriteLock!\n");
      assert(0);
    }
    return SUCCESS;
  }

  OMPRresult RWUnlock() {
    int re;
    if ((re=pthread_rwlock_unlock(&dataObjLock))!=0) {
      fprintf(stderr, "rwlock unexpected error at RWUnlock %d!\n",re);
      assert(0);
    }
    return SUCCESS;
  }


  OMPRresult getDataObjOnDevice(Device_Thread threadId, Operation& opr, int loff[], int uoff[],
      int shape[], int dims, DataInfo* pInfo) {
    if (DeviceThread2Type(threadId) == GPU1) 
      return gpu1.getDataObjOnGPU(threadId, opr, loff, uoff, shape, dims, pInfo);
    else if (DeviceThread2Type(threadId) == GPU2)
      return gpu2.getDataObjOnGPU(threadId, opr, loff, uoff, shape, dims, pInfo);
    else
      getDataObjOnCPU(threadId, opr, loff, uoff, shape, dims, pInfo);
  }


  State GPUState(int thread) {
    if (thread == GPU1_THREAD)
      return gpu1.state;
    else if (thread == GPU2_THREAD)
      return gpu2.state;
  }

  void WaitUntilInited() {
    pthread_mutex_lock(&initLock);
    pthread_mutex_unlock(&initLock);
  }

  void SetInitialized() { 
    initialized=true;
    pthread_mutex_unlock(&initLock);
  }
  bool IsInitialized() { return initialized; }

  HRESULT ValidGPU(int threadId) {
    if (threadId == GPU1_THREAD)
      ValidGPU1();
    else if (threadId == GPU2_THREAD)
      ValidGPU2();
    return SUCCESS;
  }

  HRESULT InvalidGPU(int threadId) {
    if (threadId == GPU1_THREAD)
      InvalidGPU1();
    else if (threadId == GPU2_THREAD)
      InvalidGPU2();
    return SUCCESS;
  }

  HRESULT ValidGPU1() {
    pthread_mutex_lock(&modifyState);
    gpu1.state = SValid;
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT ValidCPU() {
    pthread_mutex_lock(&modifyState);
    cpuState = SValid;
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT ValidGPU2() {
    pthread_mutex_lock(&modifyState);
    gpu2.state = SValid;
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT OldCPU() {
    pthread_mutex_lock(&modifyState);
    cpuState = SOld;
    InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT OldGPU(Device_Thread threadId)
  {
    if (threadId == GPU1_THREAD)
      OldGPU1();
    else if (threadId == GPU2_THREAD)
      OldGPU2();
  }

  HRESULT OldGPU1() {
    pthread_mutex_lock(&modifyState);
    gpu1.state = SOld;
    gpu1.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT OldGPU2() {
    pthread_mutex_lock(&modifyState);
    gpu2.state = SOld;
    gpu2.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT WaitGPU(Device_Thread threadId)
  {
    if (threadId == GPU1_THREAD)
      WaitGPU1();
    else if (threadId == GPU2_THREAD)
      WaitGPU2();
  }

  HRESULT WaitGPU1() {
    pthread_mutex_lock(&modifyState);
    gpu1.state = SWaiting;
    gpu1.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT WaitGPU2() {
    pthread_mutex_lock(&modifyState);
    gpu2.state = SWaiting;
    gpu2.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT InvalidCPU() {
    pthread_mutex_lock(&modifyState);
    cpuState = SInvalid;
    InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT InvalidGPU1() {
    pthread_mutex_lock(&modifyState);
    gpu1.state = SInvalid;
    gpu1.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }

  HRESULT InvalidGPU2() {
    pthread_mutex_lock(&modifyState);
    gpu2.state = SInvalid;
    gpu2.InvalidHaloRegionObserver();
    pthread_mutex_unlock(&modifyState);
    return SUCCESS;
  }


};

class BlockKey : public IRectangleBlock{
public:
  void* base; 
  int lb[DSM_MAX_DIMENSION], ub[DSM_MAX_DIMENSION];

  int LB(int idx) { return lb[idx]; }
  int UB(int idx) { return ub[idx]; }

  void SetDimBound(int idx, int aLb, int aUb) {
    lb[idx]=aLb; ub[idx]=aUb;
  }

  BlockKey() {
    memset(lb,0,sizeof(lb));
    memset(ub,0,sizeof(ub));
  }

  BlockKey(BlockKey* key): base(key->base) {
    memcpy(lb, key->lb, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(ub, key->ub, sizeof(int)*DSM_MAX_DIMENSION);
  }

  void SetAllDimBound(int* aLb, int* aUb) {
    memcpy(lb, aLb, sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(ub, aUb, sizeof(int)*DSM_MAX_DIMENSION);
  }

  friend bool operator < (const BlockKey& k1, const BlockKey& k2);

} ;

//typedef std::map<void*, DataObjState*> DataStateType;
typedef std::map<BlockKey, DataObjState*> DataStateType;
//typedef SafeMap<BlockKey, DataObjState*> DataStateType;


/*
typedef struct {
  void* base; //on cpu
  int dims;
  int blockShape[DSM_MAX_DIMENSION];
  
}DataShapeType ;
*/


typedef SafeMap<HaloBlock, HaloState*> HaloBlockListType;

class CUDAInfo {
public:
  CUdevice device;
  CUcontext context;
  pthread_mutex_t contextLock;

public:
  HRESULT LockAndPushContext() {
    pthread_mutex_lock(&contextLock);
    CuSafe(cuCtxPushCurrent(context));
    return SUCCESS;
  }

  HRESULT ReleaseAndPopContext() {
    CuSafe(cuCtxPopCurrent(&context));
    pthread_mutex_unlock(&contextLock);
    return SUCCESS;
  }

  void Empty() {}


} ;

extern CUDAInfo Cudas[GPU_NUM]; //ompr_dsm.cu
extern HaloBlockListType FoldHaloBlocks;  //ompr_dsm.cu
extern cudaStream_t Streams[GPU_NUM*2][DSM_MAX_STREAMS]; //ompr_dsm.cu


#endif



#include "ompr_prefetch.h"

#include "ompr_dsm.h"
#include "ompr_heap.h"

#define  OtherGPUThd(x) ((Device_Thread)(((x)+1)%2))

//goto ompr_dsm.cu
//extern DSMDFTraceManager traceMgr;
//extern DSMPendQueueManger pendQueueMgr;

extern DataStateType DataState;
extern pthread_rwlock_t ModifyDataState;
extern CUDAInfo Cudas[GPU_NUM];
extern cudaStream_t Streams[GPU_NUM*2][DSM_MAX_STREAMS];

extern PortableMemoryManager* PorMemManager;

extern DeviceMemoryManager* DevMemManager[GPU_NUM];

extern CPUMemoryManager* CPUMemManager;
extern HRESULT KernelCopy(void* dest, int* dLen, int* dOffset,
    void* src, int* sLen, int* sOffset,
    int* shape, int dim, ElementType type);

extern void KernelCopyAsync(void* dest, int* dLen, int* dOffset,
    void* src, int* sLen, int* sOffset,
    int* shape, int dim, ElementType type, cudaStream_t &stream);

extern void checkCUDAError(const char *msg);

//from one block transfer to another block
void TransferOnePendEntry(int threadId,cudaStream_t& srcGPUStream,cudaStream_t& destGPUStream,
    ElementType type,int dims,int *shape,
    void *srcAddr,int *sLen,int *sOffset,
    void *destAddr,int *dLen,int* dOffset) 
{
    //kernelcopy to tmp space
    int size = 1;
    for(int i = 0; i < dims; i++){
        size *= shape[i];
    }
    size *= ElementSize(type);

    void* tmpSpace;
    tmpSpace = DevMemManager[threadId]->allocBlock(size);
    Cudas[threadId].LockAndPushContext();

    //alloc space
    //CudaSafe(cudaMalloc(&tmpSpace,size));

    //data to continuous space 
    int tmpOffset[DSM_MAX_DIMENSION];
    memset(tmpOffset,0,sizeof(int)*DSM_MAX_DIMENSION);

    //use async 
    KernelCopyAsync(tmpSpace,shape,tmpOffset,srcAddr,sLen,sOffset,shape,dims,type,srcGPUStream);	
    checkCUDAError("check data reorganization");

    //alloc cpu space
    void* cpuforGPU;// = malloc(size);
    cpuforGPU = PorMemManager->allocBlock(size);
    //CudaSafe(cudaHostAlloc((void**)&cpuforGPU,size,cudaHostAllocPortable));
    //transfer to cpu
    CudaSafe(cudaMemcpyAsync(cpuforGPU,tmpSpace,size,cudaMemcpyDeviceToHost,srcGPUStream));
    //CudaSafe(cudaFree(tmpSpace));
    Cudas[threadId].ReleaseAndPopContext();
    DevMemManager[threadId]->freeBlock(tmpSpace);

    
    tmpSpace = DevMemManager[OtherGPUThd(threadId)]->allocBlock(size);
    //from cpu to another gpu
    Cudas[OtherGPUThd(threadId)].LockAndPushContext();
    //CudaSafe(cudaMalloc(&tmpSpace,size));
    CudaSafe(cudaMemcpyAsync(tmpSpace,cpuforGPU,size,cudaMemcpyHostToDevice,destGPUStream));
    //CudaSafe(cudaFreeHost(cpuforGPU));
    PorMemManager->freeBlock(cpuforGPU);
    
    //copy from continues space to dest
    //use async
    KernelCopyAsync(destAddr,dLen,dOffset,tmpSpace,shape,tmpOffset,shape,dims,type,destGPUStream);
    checkCUDAError("reorganization in dest gpu");
   
    //CudaSafe(cudaFree(tmpSpace));
    Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
    DevMemManager[OtherGPUThd(threadId)]->freeBlock(tmpSpace);
}

//TODO transfer one region
HRESULT startPendPrefetch(int srcPid,PendEntry* pendEntry)
{
  //get DataObjState
  int srcShape[DSM_MAX_DIMENSION];
  int destShape[DSM_MAX_DIMENSION];
  Region*& destRegion = pendEntry->destRegion;
  Region*& srcRegion = pendEntry->srcRegion;
  for(int i = 0; i < destRegion->dims; i++){
    srcShape[i] = srcRegion->key->ub[i] - srcRegion->key->lb[i] + 1;
    destShape[i] = destRegion->key->ub[i] - destRegion->key->lb[i] + 1;
  }

  pthread_rwlock_rdlock(&ModifyDataState);
  DataObjState* srcState = DataState.find(srcRegion->key)->second;
  DataObjState* destState = DataState.find(destRegion->key)->second;
  pthread_rwlock_unlock(&ModifyDataState);

  if(srcState == NULL ) { 
    CheckAndInitDataBlockState(*(srcRegion->key),srcRegion->key->base,srcShape,
        srcRegion->len,srcRegion->key->lb,srcRegion->dims,srcRegion->eleType,
        srcRegion->loff,srcRegion->uoff,true);
    pthread_rwlock_rdlock(&ModifyDataState);
    srcState = DataState.find(srcRegion->key)->second;
    pthread_rwlock_unlock(&ModifyDataState);
  }
  if(destState == NULL) {
    CheckAndInitDataBlockState(*(destRegion->key),destRegion->key->base,destShape,
        destRegion->len,destRegion->key->lb,destRegion->dims,destRegion->eleType,
        destRegion->loff,destRegion->uoff,true);
    pthread_rwlock_rdlock(&ModifyDataState);
    destState = DataState.find(destRegion->key)->second;
    pthread_rwlock_unlock(&ModifyDataState);
  }

  //compute data
  int totalSize = 1;
  int offset = 0;
  int dLen[DSM_MAX_DIMENSION],sLen[DSM_MAX_DIMENSION],
      sOffset[DSM_MAX_DIMENSION],dOffset[DSM_MAX_DIMENSION];
  for(int i = destState->dims -1 ; i >= 0 ; --i){
    //halo block 
    if(destRegion->key->lb[i] == 0) destRegion->loff[i] = 0;
    if(destRegion->key->ub[i] == destRegion->len[i] - 1 ) destRegion->uoff[i] = 0;

    dLen[i] = destState->shape[i] + destRegion->loff[i] + destRegion->uoff[i];
    sLen[i] = srcState->shape[i] + srcState->pGpus[srcPid]->loff[i] + 
      srcState->pGpus[srcPid]->uoff[i];

    sOffset[i] = pendEntry->dimsOffset[i] - srcState->dimsOffset[i] + 
      srcState->pGpus[srcPid]->loff[i];
    dOffset[i] = pendEntry->dimsOffset[i] - destState->dimsOffset[i] + 
      destRegion->loff[i];

    offset = offset*(destState->shape[i]+destRegion->loff[i] +
        destRegion->uoff[i])+destRegion->loff[i];
    totalSize *= dLen[i];
  }

  if(destState->prefetchData.status != START && destState->prefetchData.Addr == NULL){
    destState->prefetchData.clear();
  }
  if(destState->prefetchData.status == START){
    //alloc something
    if(destState->prefetchData.Addr == NULL) {
      printf("prefetch alloc %d \n",totalSize*ElementSize(destState->eleType));
      GPUAlloc(OtherGPUThd(srcPid), &(destState->prefetchData.Addr), 
          totalSize*ElementSize(destState->eleType));
      destState->prefetchData.addr = (char*)destState->prefetchData.Addr + 
        offset*ElementSize(destState->eleType);

      size_t memFree,memTotal;
      Cudas[1-srcPid].LockAndPushContext();
      cuMemGetInfo(&memFree, &memTotal);
      printf("GPU: %d memFree:%ud,memTotal:%ud\n",1-srcPid,memFree,memTotal);
      Cudas[1-srcPid].ReleaseAndPopContext();


    } else {
      assert(0);
    }
  } 

  void* srcAddr = srcState->pGpus[srcPid]->Addr;
  void* destAddr = destState->prefetchData.Addr;
  assert(destAddr);
  assert(srcAddr);

  TransferOnePendEntry(srcPid,pendEntry->srcStream,pendEntry->destStream,
      srcState->eleType,srcState->dims,pendEntry->shape,
      srcAddr,sLen,sOffset,
      destAddr,dLen,dOffset);

  assert(destAddr);
  //add information for this region have prefetch 
  //printf("add info \n");
  destState->prefetchData.pid = 1 - srcPid;
  destState->prefetchData.status = RUNNING;
  destState->prefetchData.stream = pendEntry->destStream;

  RedistributionPoint point;
  memcpy(point.shape,pendEntry->shape,sizeof(int)*DSM_MAX_DIMENSION);
  memcpy(point.offset,pendEntry->dimsOffset,sizeof(int)*DSM_MAX_DIMENSION);
  destState->prefetchData.vecPoint.push_back(point);
  return SUCCESS;
}

#define MAX_PEND_AOT 50
//added by shou baojiang
void* TransferState1Copy(int threadId, cudaStream_t& srcGPUStream, ElementType type, int dims,
    int* shape, void* srcAddr, int *sLen, int* sOffset)
{
  //kernelcopy to tmp space
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= shape[i];
  }
  size *= ElementSize(type);

  void* tmpSpace;
  tmpSpace = DevMemManager[threadId]->allocBlock(size);
  Cudas[threadId].LockAndPushContext();

  //alloc space
  //CudaSafe(cudaMalloc(&tmpSpace,size));

  //data to continuous space 
  int tmpOffset[DSM_MAX_DIMENSION];
  memset(tmpOffset,0,sizeof(int)*DSM_MAX_DIMENSION);

  //use async 
  KernelCopyAsync(tmpSpace,shape,tmpOffset,srcAddr,sLen,sOffset,shape,dims,type,srcGPUStream);	
  checkCUDAError("check data reorganization");

  Cudas[threadId].ReleaseAndPopContext();
  return tmpSpace;
}

void* TransferState2PCIE(int threadId, void* tmpSpace, int*shape, cudaStream_t& srcGPUStream,
    ElementType type, int dims)
{
  //kernelcopy to tmp space
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= shape[i];
  }
  size *= ElementSize(type);

  Cudas[threadId].LockAndPushContext();
  //alloc cpu space
  void* cpuforGPU;// = malloc(size);
  cpuforGPU = PorMemManager->allocBlock(size);
  //CudaSafe(cudaHostAlloc((void**)&cpuforGPU,size,cudaHostAllocPortable));
  //transfer to cpu
  CudaSafe(cudaMemcpyAsync(cpuforGPU,tmpSpace,size,cudaMemcpyDeviceToHost,srcGPUStream));
  //CudaSafe(cudaFree(tmpSpace));
  Cudas[threadId].ReleaseAndPopContext();
  //DevMemManager[threadId]->freeBlock(tmpSpace);

  return cpuforGPU;
}

void* TransferState3PCIE(int threadId, void* cpuforGPU,int* shape, cudaStream_t& destGPUStream,
    ElementType type, int dims)
{
  //kernelcopy to tmp space
  int size = 1;
  for(int i = 0; i < dims; i++){
    size *= shape[i];
  }
  size *= ElementSize(type);

  void* tmpSpace = DevMemManager[OtherGPUThd(threadId)]->allocBlock(size);
  //from cpu to another gpu
  Cudas[OtherGPUThd(threadId)].LockAndPushContext();
  //CudaSafe(cudaMalloc(&tmpSpace,size));
  CudaSafe(cudaMemcpyAsync(tmpSpace,cpuforGPU,size,cudaMemcpyHostToDevice,destGPUStream));
  //CudaSafe(cudaFreeHost(cpuforGPU));
  PorMemManager->freeBlock(cpuforGPU);
  Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
  return tmpSpace;
}

void* TransferState4Copy(int threadId, cudaStream_t& destGPUStream, ElementType type, int dims,
    int* shape, void* destAddr, int *dLen, int* dOffset, void* tmpSpace)
{

  int tmpOffset[DSM_MAX_DIMENSION];
  memset(tmpOffset,0,sizeof(int)*DSM_MAX_DIMENSION);
  Cudas[OtherGPUThd(threadId)].LockAndPushContext();
  //copy from continues space to dest
  //use async
  KernelCopyAsync(destAddr,dLen,dOffset,tmpSpace,shape,tmpOffset,shape,dims,type,destGPUStream);
  checkCUDAError("reorganization in dest gpu");

  //CudaSafe(cudaFree(tmpSpace));
  Cudas[OtherGPUThd(threadId)].ReleaseAndPopContext();
  DevMemManager[OtherGPUThd(threadId)]->freeBlock(tmpSpace);

  return NULL;
}


HRESULT startPendPrefetches(int srcPid, std::vector<PendEntry*> arrivedEntry, 
    sem_t* semWorkerEnd)
{
  ElementType elesType[MAX_PEND_AOT];
  int dims[MAX_PEND_AOT];
  int shapes[MAX_PEND_AOT][DSM_MAX_DIMENSION], 
      sLens[MAX_PEND_AOT][DSM_MAX_DIMENSION], sOffsets[MAX_PEND_AOT][DSM_MAX_DIMENSION],
      dLens[MAX_PEND_AOT][DSM_MAX_DIMENSION], dOffsets[MAX_PEND_AOT][DSM_MAX_DIMENSION],
      dimsOffsets[MAX_PEND_AOT][DSM_MAX_DIMENSION];
  DataObjState* destStates[MAX_PEND_AOT];
  void* srcAddr[MAX_PEND_AOT], *destAddr[MAX_PEND_AOT];
  cudaStream_t* srcStreams[MAX_PEND_AOT], *destStreams[MAX_PEND_AOT];
  int count=0;
  for (std::vector<PendEntry*>::iterator itr= arrivedEntry.begin(); 
      itr!=arrivedEntry.end(); ++itr, count++) {
    PendEntry* pendEntry = *itr;
    //get DataObjState
    int srcShape[DSM_MAX_DIMENSION];
    int destShape[DSM_MAX_DIMENSION];
    Region*& destRegion = pendEntry->destRegion;
    Region*& srcRegion = pendEntry->srcRegion;
    for(int i = 0; i < destRegion->dims; i++){
      srcShape[i] = srcRegion->key->ub[i] - srcRegion->key->lb[i] + 1;
      destShape[i] = destRegion->key->ub[i] - destRegion->key->lb[i] + 1;
    }

    pthread_rwlock_rdlock(&ModifyDataState);
    DataObjState* srcState = DataState.find(srcRegion->key)->second;
    DataObjState* destState = DataState.find(destRegion->key)->second;
    pthread_rwlock_unlock(&ModifyDataState);

    if(srcState == NULL ) { 
      CheckAndInitDataBlockState(*(srcRegion->key),srcRegion->key->base,srcShape,
          srcRegion->len,srcRegion->key->lb,srcRegion->dims,srcRegion->eleType,
          srcRegion->loff,srcRegion->uoff,true);
      pthread_rwlock_rdlock(&ModifyDataState);
      srcState = DataState.find(srcRegion->key)->second;
      pthread_rwlock_unlock(&ModifyDataState);
    }
    if(destState == NULL) {
      CheckAndInitDataBlockState(*(destRegion->key),destRegion->key->base,destShape,
          destRegion->len,destRegion->key->lb,destRegion->dims,destRegion->eleType,
          destRegion->loff,destRegion->uoff,true);
      pthread_rwlock_rdlock(&ModifyDataState);
      destState = DataState.find(destRegion->key)->second;
      pthread_rwlock_unlock(&ModifyDataState);
    }
    destStates[count] = destState;
    memcpy(dimsOffsets[count], pendEntry->dimsOffset, sizeof(int)*DSM_MAX_DIMENSION);

    //compute data
    int totalSize = 1;
    int offset = 0;
    //int dLen[DSM_MAX_DIMENSION],sLen[DSM_MAX_DIMENSION],
    //    sOffset[DSM_MAX_DIMENSION],dOffset[DSM_MAX_DIMENSION];
    int *dLen = dLens[count], *dOffset=dOffsets[count],
        *sLen = sLens[count], *sOffset=sOffsets[count];
    for(int i = destState->dims -1 ; i >= 0 ; --i){
      //halo block 
      if(destRegion->key->lb[i] == 0) destRegion->loff[i] = 0;
      if(destRegion->key->ub[i] == destRegion->len[i] - 1 ) destRegion->uoff[i] = 0;

      dLen[i] = destState->shape[i] + destRegion->loff[i] + destRegion->uoff[i];
      sLen[i] = srcState->shape[i] + srcState->pGpus[srcPid]->loff[i] + 
        srcState->pGpus[srcPid]->uoff[i];

      sOffset[i] = pendEntry->dimsOffset[i] - srcState->dimsOffset[i] + 
        srcState->pGpus[srcPid]->loff[i];
      dOffset[i] = pendEntry->dimsOffset[i] - destState->dimsOffset[i] + 
        destRegion->loff[i];

      offset = offset*(destState->shape[i]+destRegion->loff[i] +
          destRegion->uoff[i])+destRegion->loff[i];
      totalSize *= dLen[i];
    }

    if(destState->prefetchData.status != START && destState->prefetchData.Addr == NULL){
      destState->prefetchData.clear();
    }
    if(destState->prefetchData.status == START){
      //alloc something
      if(destState->prefetchData.Addr == NULL) {
        printf("prefetch alloc %d \n",totalSize*ElementSize(destState->eleType));
        GPUAlloc(OtherGPUThd(srcPid), &(destState->prefetchData.Addr), 
            totalSize*ElementSize(destState->eleType));
        destState->prefetchData.addr = (char*)destState->prefetchData.Addr + 
          offset*ElementSize(destState->eleType);

        size_t memFree,memTotal;
        Cudas[1-srcPid].LockAndPushContext();
        cuMemGetInfo(&memFree, &memTotal);
        printf("GPU: %d memFree:%ud,memTotal:%ud\n",1-srcPid,memFree,memTotal);
        Cudas[1-srcPid].ReleaseAndPopContext();


      } else {
        assert(0);
      }
    } 

    srcAddr[count] = srcState->pGpus[srcPid]->Addr;
    destAddr[count] = destState->prefetchData.Addr;
    assert(destAddr);
    assert(srcAddr);

    memcpy(shapes[count], pendEntry->shape, sizeof(int)*DSM_MAX_DIMENSION);
    dims[count] = srcState->dims;
    elesType[count] = srcState->eleType;
    srcStreams[count] = &(*itr)->srcStream; destStreams[count]=&(*itr)->destStream;
  }

  void* srcGPUTmpSpace[MAX_PEND_AOT];
  for (int i=0; i<count; i++) {
    srcGPUTmpSpace[i] = 
      TransferState1Copy(srcPid, *srcStreams[i], elesType[i], dims[i], shapes[i], srcAddr[i], 
        sLens[i], sOffsets[i]);
  }

  for (int i=0; i<count; i++) {
    DataObjState* destState = destStates[i];
    assert(destAddr);
    //add information for this region have prefetch 
    //printf("add info \n");
    destState->prefetchData.pid = 1 - srcPid;
    destState->prefetchData.status = RUNNING;
    destState->prefetchData.stream = *destStreams[i];

    RedistributionPoint point;
    memcpy(point.shape,shapes[i],sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(point.offset,dimsOffsets[i],sizeof(int)*DSM_MAX_DIMENSION);
    destState->prefetchData.vecPoint.push_back(point);
  }
 
  void* cpuTmpSpace[MAX_PEND_AOT];
  cudaEvent_t transferOK[MAX_PEND_AOT];
  for (int i=0; i<count; i++) {
    cpuTmpSpace[i] =
      TransferState2PCIE(srcPid, srcGPUTmpSpace[i], shapes[i], *srcStreams[i], elesType[i],
          dims[i]);
    Cudas[srcPid].LockAndPushContext();
    CudaSafe(cudaEventCreate(&transferOK[i]));
    CudaSafe(cudaEventRecord(transferOK[i], *srcStreams[i]));
    Cudas[srcPid].ReleaseAndPopContext();
  }

  sem_post(semWorkerEnd);

  //Cudas[srcPid].LockAndPushContext();
  //cudaSetDevice(srcPid);
  for (int i=0; i<count; i++) {
    //CudaSafe(cudaEventSynchronize(transferOK[i]));
    DevMemManager[srcPid]->freeBlock(srcGPUTmpSpace[i]);
  }
  //Cudas[srcPid].ReleaseAndPopContext();

  void* destGPUTmpSpace[MAX_PEND_AOT];
  for (int i=0; i<count; i++) {
    destGPUTmpSpace[i] =
      TransferState3PCIE(srcPid, cpuTmpSpace[i], shapes[i], *destStreams[i], elesType[i], 
          dims[i]);
  }

  for (int i=0; i<count; i++) {
    TransferState4Copy(srcPid, *destStreams[i], elesType[i], dims[i], shapes[i], destAddr[i],
        dLens[i], dOffsets[i], destGPUTmpSpace[i]);
    Cudas[OtherGPUThd(srcPid)].LockAndPushContext();
    CudaSafe(cudaEventCreate(&transferOK[i]));
    CudaSafe(cudaEventRecord(transferOK[i], *destStreams[i]));
    Cudas[OtherGPUThd(srcPid)].ReleaseAndPopContext();
  }

  //cudaSetDevice(OtherGPUThd(srcPid));
  for (int i=0; i<count; i++) {
    //Cudas[OtherGPUThd(srcPid)].LockAndPushContext();
    //CudaSafe(cudaEventSynchronize(transferOK[i]));
    //Cudas[OtherGPUThd(srcPid)].LockAndPushContext();
    DevMemManager[OtherGPUThd(srcPid)]->freeBlock(destGPUTmpSpace[i]);
  }

 
  return SUCCESS;
  
}
//end of add

typedef struct thread_arg{
    DSMDFTraceManager* dfMgr;
    int pid;
}args,*pArgs;

void* prefetch_worker(void* arg)
{

  pArgs threadArgs = (pArgs)arg;
  DSMDFTraceManager* dfMgr = threadArgs->dfMgr;
  int pid = threadArgs->pid;
  delete threadArgs;

  pthread_t self = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(GPU_NUM+pid, &cpuset);
  int result = pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset);
  if (result) assert(0);

  std::vector<PendEntry*> arrivedEntry;
  while(true){
    arrivedEntry.clear();
    sem_wait(&(dfMgr->semAgentBar[pid]));

    int numberId = dfMgr->barrierNumId[1 - pid];
    //recvive data from barrier
    int re = dfMgr->pendQueueMgr.recvAgent(pid,numberId,arrivedEntry);
    printf("pid %d recv %d ",pid,numberId),printf("arriveEntry num %d\n",arrivedEntry.size());

    if(re == SUCCESS && arrivedEntry.size()){
      //add by shou baojiang
      startPendPrefetches(1-pid, arrivedEntry, &(dfMgr->semWorkerEnd[pid]));
      /*
      for(std::vector<PendEntry*>::iterator iter = arrivedEntry.begin(); 
          iter != arrivedEntry.end(); ++iter){
        startPendPrefetch(1 - pid,*iter);
      }
      */
    } else
      sem_post(&(dfMgr->semWorkerEnd[pid]));
  }
}
//end
 

//cpp have

//class Region
Region::Region(BlockKey& key,int dims,ElementType type,int* len,int* loff,int* uoff)
    :dims(dims),eleType(type)
{
    memcpy(this->len,len,sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(this->loff,loff,sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(this->uoff,uoff,sizeof(int)*DSM_MAX_DIMENSION);
    this->key = new BlockKey(&key);
}
//class region end

//class  DFItem
DFItem::DFItem(UDType udType,Region* region)
{
	this->udType = udType;
	this->region = region;
}
//class  DFItem end

//class DFEntry
int DFEntry::entryId[2] = {0,0};

DFEntry::DFEntry(int pid)
{
	//lock
	this->id = DFEntry::entryId[pid];
	++DFEntry::entryId[pid];
    //printf("pid %d entry++\n",pid);
	//unlock
	this->scope = LOCAL;
}


DFEntry::DFEntry(int pid,Scope scope)
{
	//lock
	this->id = DFEntry::entryId[pid];
	++DFEntry::entryId[pid];
    //printf("pid %d entry++\n",pid);
	//unlock
	this->scope = scope;
}

DFEntry::~DFEntry()
{
	if(items.size()){
		//delete;
		for(std::vector<DFItem*>::reverse_iterator iter = items.rbegin(); iter != items.rend(); ++iter){
            if(*iter != NULL) {
			    delete (*iter);
                *iter = NULL;
            }
		}
	}
}

int DFEntry::getId()
{
    return id;
}

void DFEntry::clear()
{
    /*
    for(std::vector<DFItem*>::reverse_iterator iter = items.rbegin(); iter != items.rend(); ++iter){
        if(*iter != NULL) {
		    delete (*iter);
            *iter = NULL;
        }
    }*/
    items.clear();

}

void DFEntry::setScope(Scope scope)
{
    this->scope = scope;
}

Scope DFEntry::getScope()
{
    return this->scope;
}
std::vector<DFItem*>::iterator DFEntry::begin()
{
    return items.begin();
}

std::vector<DFItem*>::iterator DFEntry::end()
{
    return items.end();
}

HRESULT DFEntry::addTraceItem(int pid,UDType udType,Region* region)
{
	DFItem* item = new DFItem(udType,region);
	items.push_back(item);
	return 0;
}

HRESULT DFEntry::addTraceItem(DFItem* item)
{
	items.push_back(item);
	return 0;
}
//class DFEntry end


//class DSMDFTraceManager
DFEntry* DSMDFTraceManager::sumEntry[2] = {NULL,NULL};
DSMDFTraceManager::DSMDFTraceManager():
    num(2)
{
	int re= pthread_barrier_init(&barrierIn, NULL, 2);
	re = pthread_barrier_init(&barrierOut, NULL, 2);

    pthread_rwlock_init(&modifyTrace[0],NULL); 
    pthread_rwlock_init(&modifyTrace[1],NULL); 

    //prefetch worker thread init
    sem_init(&semAgentBar[0],0,0);
    sem_init(&semAgentBar[1],0,0);
    sem_init(&semWorkerEnd[0],0,0);
    sem_init(&semWorkerEnd[1],0,0);
    //int p[2] = {4,5};
    //p[0] = PREFETCH_WORKER;
    //p[1] = PREFETCH_WORKER;
    pArgs threadArgs0 = new args();
    threadArgs0->dfMgr = this;
    threadArgs0->pid = 0;
    re = pthread_create(&pthreads[0], NULL, prefetch_worker, threadArgs0);

    pArgs threadArgs1 = new args();
    threadArgs1->dfMgr = this;
    threadArgs1->pid = 1;
    re = pthread_create(&pthreads[1], NULL, prefetch_worker, threadArgs1);

    this->barrierNumId[0] = 0;
    this->barrierNumId[1] = 0;
}

DSMDFTraceManager::DSMDFTraceManager(int n):num(n)
{
	//init barrier
	int re= pthread_barrier_init(&barrierIn, NULL, n);
	re= pthread_barrier_init(&barrierOut, NULL, n);
}

DSMDFTraceManager::~DSMDFTraceManager()
{
    for(int pid = 0; pid < 2; ++pid)
        for(int i = trace[pid].size() - 1; i >=0 ; --i){ 
            delete trace[pid].at(i);
        }
}
int DSMDFTraceManager::popHead(int pid)
{
    pthread_rwlock_wrlock(&modifyTrace[pid]);
    if(!trace[pid].size()) {
        pthread_rwlock_unlock(&modifyTrace[pid]);
        return -1;
    }
	int id = trace[pid].front()->getId();
	trace[pid].pop_front();
    pthread_rwlock_unlock(&modifyTrace[pid]);
	return id;
}

bool DSMDFTraceManager::pushTail(int pid,DFEntry* entry)
{
    //pthread_rwlock_wrlock(&modifyTrace[pid]);
	trace[pid].push_back(entry);
    //pthread_rwlock_unlock(&modifyTrace[pid]);
	return true;
}

void DSMDFTraceManager::addItem(int pid,UDType udType,Region* region)
{
    pthread_rwlock_wrlock(&modifyTrace[pid]);
    if(!trace[pid].size()) {
        DFEntry *newEntry = new DFEntry(pid,LOCAL);
	    this->pushTail(pid,newEntry);
    }
	trace[pid].back()->addTraceItem(pid, udType, region);
    pthread_rwlock_unlock(&modifyTrace[pid]);
}

//compute two blocks's intersect block
static bool intersect(const Region* a,const Region* b,int *dimsOffset,int *shape) {
    if(a->key->base != b->key->base)
        return false;
    int ub[DSM_MAX_DIMENSION];
    for(int i = 0; i < a->dims; i++) {
        dimsOffset[i] = MAX(a->key->lb[i] , b->key->lb[i]);
        ub[i] = MIN(a->key->ub[i], b->key->ub[i]);
    }
    for(int i = 0; i < a->dims; i++) {
        shape[i] = ub[i] - dimsOffset[i] + 1;
        if(shape[i] <= 0)
            return false;//not intersect
    }
    return true;//
}

//find 
struct Position{
	int offset[DSM_MAX_DIMENSION];
	int shape[DSM_MAX_DIMENSION];
	Position(int offset[],int shape[]){
		memcpy(this->offset,offset,DSM_MAX_DIMENSION*sizeof(int));
		memcpy(this->shape,shape,DSM_MAX_DIMENSION*sizeof(int));
	};
	bool empty(int dims){
		for(int i = 0; i < dims; i++)
			if(!shape[i])
				return true;
		return false;
	};
};
bool intersectPosition(Position& a,Position& b,int dims)
{
	int offset[DSM_MAX_DIMENSION];
	int ub[DSM_MAX_DIMENSION];

	for(int i = 0; i < dims; i++) {
		offset[i] = MAX(a.offset[i] , b.offset[i]);
		ub[i] = MIN(a.offset[i] + a.shape[i] , b.offset[i] + b.shape[i]);
	}
	for(int i = 0; i < dims; i++) {
		if(ub[i] - offset[i] <= 0)
			return false;//not intersect
	}
	for(int i = 1; i < dims; i++) {
		b.shape[i] = offset[i] - b.offset[i];
        //printf("shape change\n");
	}
	return true;//
};
//TODO Pending region for prefetch
HRESULT DSMDFTraceManager::findAndPendIntersect(int pid,cudaStream_t& srcStream,
    cudaStream_t& destStream,Region* region)
{
    int res = 0;

	std::vector<Position> checkPoint;
    checkPoint.clear();

    int distance = 0;
    pthread_rwlock_rdlock(&modifyTrace[pid]);
    std::deque<DFEntry*> tmpTrace(trace[pid].begin(),trace[pid].end());
    pthread_rwlock_unlock(&modifyTrace[pid]);

    int dims = region->dims;
    for(int i = tmpTrace.size() - 1; i >= 0; --i,++distance){ 
        bool isBarrier = (tmpTrace.at(i)->getScope() == GLOBAL);
        int id = tmpTrace.at(i)->getId(); 
        //printf("trace entryid  %d\n",id);
        //one entry
        for(std::vector<DFItem*>::iterator itemIter = tmpTrace.at(i)->begin(); \
            itemIter != tmpTrace.at(i)->end(); ++itemIter){
            if((*itemIter) == NULL)
                continue;
            //printf("hou: %d\n",(*itemIter)->udType);
            if(isBarrier && (*itemIter)->udType == DEF){ 
                int dimsOffset[DSM_MAX_DIMENSION],shape[DSM_MAX_DIMENSION];
                //other GPU's def
                if(intersect(region,(*itemIter)->region,dimsOffset,shape) == true){
					//check();
					Position a(dimsOffset,shape);
                    
					for(std::vector<Position>::iterator iter = checkPoint.begin(); 
                        iter != checkPoint.end(); ++iter){
						intersectPosition(*iter,a,dims);
					}
					if(a.empty(dims)){
						continue;
					}
					checkPoint.push_back(a);//this region can prefetch
                    if(distance > 1) {
                        pendQueueMgr.addEntry(pid,srcStream,destStream,(*itemIter)->region,region,\
                            a.offset,a.shape,id);
                        ++res;
                    }
				} 
			} else if(!isBarrier) { //this GPU's def or use 
                //setDef  check
				int dimsOffset[DSM_MAX_DIMENSION],shape[DSM_MAX_DIMENSION];
				//other GPU's def
				if(intersect(region,(*itemIter)->region,dimsOffset,shape) == true){
					//check();
					Position a(dimsOffset,shape);
					for(std::vector<Position>::iterator iter = checkPoint.begin(); iter != checkPoint.end(); ++iter){
						intersectPosition(*iter,a,dims);
					}
					if(a.empty(dims)){
						continue;
					}
					checkPoint.push_back(a);//this region can prefetch from this GPU,
                    //but in my system ,this should be not handle,because it can't hidden comunication
				} 
            }
        }
	}
    return res;
}

//input use
HRESULT DSMDFTraceManager::prefetchInput(int pid,int streamId,BlockKey& key,int dims,
    ElementType type,int* len,int* loff,int* uoff)
{
    Region* region = new Region(key,dims,type,len,loff,uoff);

    int agentId = getAgentIdFromPrefetchId(pid);

    int peerPid;
    if(pid == 3){
        peerPid = 4;
    } else if(pid == 4){
        peerPid = 3;
    }
    
    if (Streams[pid - 1][streamId] == 0) {
      Cudas[agentId].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[pid - 1][streamId]));
      Cudas[agentId].ReleaseAndPopContext();
    }
    if (Streams[peerPid -1][streamId] == 0) {
      Cudas[OtherGPUThd(agentId)].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[peerPid - 1][streamId]));
      Cudas[OtherGPUThd(agentId)].ReleaseAndPopContext();
    }

    //TODO this just use same streamid for all gpus,some need modify
    cudaStream_t& destStream = Streams[pid-1][streamId];
    cudaStream_t& srcStream = Streams[peerPid - 1][streamId];

    //int re;
    //find this item's pending block
    //if(region->dims != 4) 
    //re = 
    this->findAndPendIntersect(agentId,srcStream,destStream,region);

    //if(re > 0 && region->cpuState == SValid) 
    //    region->PrefetchState();

    //add this item to trace
    this->addItem(agentId,USE,region);
    

    return SUCCESS;

}

//output def
HRESULT DSMDFTraceManager::prefetchOutput(int pid,int streamId,BlockKey& key,int dims,
    ElementType type,int* len,int* loff,int* uoff)
{
    Region* region = new Region(key,dims,type,len,loff,uoff);
    int agentId = getAgentIdFromPrefetchId(pid);

    int peerPid;
    if(pid == 3){
        peerPid = 4;
    } else if(pid == 4){
        peerPid = 3;
    }
    
    if (Streams[pid - 1][streamId] == 0) {
      Cudas[agentId].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[pid - 1][streamId]));
      Cudas[agentId].ReleaseAndPopContext();
    }
    if (Streams[peerPid - 1][streamId] == 0) {
      Cudas[OtherGPUThd(agentId)].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[peerPid - 1][streamId]));
      Cudas[OtherGPUThd(agentId)].ReleaseAndPopContext();
    }

    //TODO this just use same streamid for all gpus,some need modify
    cudaStream_t& destStream = Streams[pid-1][streamId];
    cudaStream_t& srcStream = Streams[peerPid - 1][streamId];
    /*
    if (Streams[agentId][streamId] == 0) {
      Cudas[agentId].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[agentId][streamId]));
      Cudas[agentId].ReleaseAndPopContext();
    }

    if (Streams[OtherGPUThd(agentId)][streamId] == 0) {
      Cudas[OtherGPUThd(agentId)].LockAndPushContext();
      CudaSafe(cudaStreamCreate(&Streams[OtherGPUThd(agentId)][streamId]));
      Cudas[OtherGPUThd(agentId)].ReleaseAndPopContext();
    }

    //TODO this just use same streamid foe all gpus,some need modify
    cudaStream_t& destStream = Streams[agentId][streamId];
    cudaStream_t& srcStream = Streams[OtherGPUThd(agentId)][streamId];
    */

    int re;
    //find this item's pending block
    //if(agentId == 0)
    //if( region->dims != 4 ) 
    //re = 
    this->findAndPendIntersect(agentId,srcStream,destStream,region);
    //if(re > 0) 
    //    region->PrefetchState();

    //add this item to trace
    this->addItem(agentId,DEF,region);
    return SUCCESS;
}

void DSMDFTraceManager::sumThisThread(int pid)
{
    DSMDFTraceManager::sumEntry[pid] = new DFEntry(pid,GLOBAL);

    pthread_rwlock_rdlock(&modifyTrace[pid]);
    for(int i = trace[pid].size() - 1; i >=0 ; --i){ 
        if(trace[pid].at(i)->getScope() == GLOBAL){
            break;
        } else {
            for(std::vector<DFItem*>::iterator itemIter = trace[pid].at(i)->begin();
                itemIter != trace[pid].at(i)->end(); ++itemIter) {
                if((*itemIter)->udType == DEF) {
                    DSMDFTraceManager::sumEntry[pid]->addTraceItem(pid,
                        DEF,(*itemIter)->region);
                }
            }
        }
    }
    pthread_rwlock_unlock(&modifyTrace[pid]);
}

HRESULT DSMDFTraceManager::agentBarrierInput(int pid)
{
  //agent barrier reach
  pthread_rwlock_wrlock(&modifyTrace[pid]);
  //trace[pid].pop_front();
  //this->barrierNumId[pid] = trace[pid].front()->getId() ;
  //trace[pid].pop_front();
  //assert(trace[pid].size());
  while(trace[pid].size()){
    if(trace[pid].front()->getScope() == GLOBAL){
      this->barrierNumId[pid] = trace[pid].front()->getId() ;
      //assert(this->barrierNumId[pid]);
      trace[pid].pop_front();
      break;
    } else { 
      trace[pid].pop_front();
    }
  }
  pthread_rwlock_unlock(&modifyTrace[pid]);
  //this send the other's GPU worker thread a signal,meaning that,other GPU can prefetch
  //before this  barrier
  //send 	
  sem_post(&semAgentBar[1 - pid]);
  //wait the ohter GPU prefetch end
  //recv 
  sem_wait(&semWorkerEnd[1-pid]);

  return SUCCESS;
}

HRESULT DSMDFTraceManager::prefetchBarrierInput(int pid)
{
    int agentId = getAgentIdFromPrefetchId(pid);
    this->sumThisThread(agentId);

    //exchange data
	pthread_barrier_wait(&barrierIn);
	trace[agentId].push_back(DSMDFTraceManager::sumEntry[1-agentId]);
	//this->pushTail(agentId,DSMDFTraceManager::sumEntry[1-agentId]);
	pthread_barrier_wait(&barrierOut);

	//add some handle
    DFEntry *newEntry = new DFEntry(agentId,LOCAL);
	this->pushTail(agentId,newEntry);
    
    return SUCCESS;
}

void DSMDFTraceManager::stopWorkerThreads()
{
    //this->thread_stop = 1;
    pthread_cancel(this->pthreads[0]);
    pthread_cancel(this->pthreads[1]);
}

//class DSMDFTraceManager end

//class PendEntry
PendEntry::PendEntry(Region* srcRegion,Region* destRegion,int *dimsOffset,int *shape,
    int number,cudaStream_t& srcStream,cudaStream_t& destStream):
    srcRegion(srcRegion), destRegion(destRegion), number(number),
    srcStream(srcStream),destStream(destStream)
{
    memcpy(this->dimsOffset,dimsOffset,sizeof(int)*DSM_MAX_DIMENSION);
    memcpy(this->shape,shape,sizeof(int)*DSM_MAX_DIMENSION);
}
//class PendEntry end



//class DSMPendQueueManger
DSMPendQueueManger::DSMPendQueueManger():
    num(2)
{
    pthread_rwlock_init(&modifyQueue[0],NULL); 
    pthread_rwlock_init(&modifyQueue[1],NULL); 
}

DSMPendQueueManger::DSMPendQueueManger(int n):
    num(n)
{

}

//free all memory
DSMPendQueueManger::~DSMPendQueueManger()
{
    for(int i = 0; i < this->num; i++){
        for(std::multimap<int,PendEntry*>::iterator iter = pendQueue[i].begin();
            iter != pendQueue[i].end(); ++iter)
            delete (iter->second);
    }
}

//add entry to the queue tail
HRESULT DSMPendQueueManger::addEntry(int pid,cudaStream_t& srcStream,cudaStream_t& destStream,
    Region* srcRegion,Region* destRegion,int *dimsOffset,int *shape,int number)
{
	PendEntry* pendEntry = new PendEntry(srcRegion,destRegion,dimsOffset,shape,number,srcStream,destStream);
    //add to map
    //printf("pid %d add pend src addr %lld dest addr %lld\n",pid,srcRegion->base,destRegion->base);
    //for(int i = 0; i < srcRegion->dims; i++){
    //    printf("offset[%d] %d shape[%d] %d\n",i,dimsOffset[i],i,shape[i]);
    //}
    printf("pid %d num : %d can prefetch block\n",pid,number);

    pthread_rwlock_wrlock(&modifyQueue[pid]);
	pendQueue[pid].insert(std::make_pair(number,pendEntry));
    pthread_rwlock_unlock(&modifyQueue[pid]);
	return 0;
}

bool DSMPendQueueManger::empty()
{
	return (pendQueue[0].size() || pendQueue[1].size());
}


//if agent thread reach one barrier ,prefetch possible region
HRESULT DSMPendQueueManger::recvAgent(int pid,int number,std::vector<PendEntry*>& arrivedEntry)
{

  pthread_rwlock_rdlock(&modifyQueue[pid]);
  for(std::multimap<int,PendEntry*>::iterator iter = pendQueue[pid].lower_bound(number); 
      iter != pendQueue[pid].upper_bound(number);){

    arrivedEntry.push_back(iter->second);
    //delete iter->second;
    pendQueue[pid].erase(iter++);
  }
  pthread_rwlock_unlock(&modifyQueue[pid]);
  return SUCCESS;
}
//class DSMPendQueueManger end




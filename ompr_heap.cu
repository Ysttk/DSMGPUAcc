#include "ompr_heap.h"
#include "ompr_dsm.h"
#include <cstdio>


#define ENABLE_MEM_POOL

PortableMemoryManager::PortableMemoryManager() {
#ifdef ENABLE_MEM_POOL
  Construct();
#endif
}

PortableMemoryManager::~PortableMemoryManager() {
#ifdef ENABLE_MEM_POOL
  Destruct();
#endif
}

void* PortableMemoryManager::allocMemory(int size) {
  void* base;
  //CudaSafe(cudaHostAlloc(&tmp, size, cudaHostAllocPortable));

#ifdef ENABLE_MEM_POOL
  //1st: check whether block with the same size already in the freeList
  pthread_mutex_lock(&mutex);
  for (std::map<void*,MemoryChunk*>::iterator itr = freeList.begin();
      itr != freeList.end();) {
    if (itr->second->getSize() == size) {
      base = itr->second->getBaseAddr();
      freeList.erase(itr++);
      allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));
      pthread_mutex_unlock(&mutex);
      return base;
    } else ++itr;
  }

  pthread_mutex_unlock(&mutex);
  //2nd: no block with the same size in the freeList, so try to allocate a new one
  cudaError_t result = cudaHostAlloc(&base, size, cudaHostAllocPortable);
  if (result==cudaSuccess) {
    allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));
    return base;
  }

  //3rd: allocate memory failed, try to free block nearby to get a new one, 
  //but if all block in freeList are freed and still can't get a block by cudaMalloc, 
  //then alert.
  int remainSize = size;
  do {
    if (freeList.size() == 0) {
      assert(0);
    }
    int totalSize =0;
    for (std::map<void*,MemoryChunk*>::iterator itr = freeList.begin();
        itr != freeList.end();) {
      CudaSafe(cudaFreeHost(itr->second->getBaseAddr()));
      totalSize += itr->second->getSize();
      delete itr->second;
      freeList.erase(itr++);
      if (totalSize > remainSize) break;
    }
    remainSize = 0;
  } while (cudaHostAlloc(&base, size, cudaHostAllocPortable) != cudaSuccess);

  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error:  %s.\n", cudaGetErrorString( err) );
  }
  allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));

#else
  CudaSafe(cudaHostAlloc(&base, size), cudaHostAllocPortable);
#endif


  return base;
}

void PortableMemoryManager::freeMemory(void* base) {
#ifdef ENABLE_MEM_POOL
  pthread_mutex_lock(&mutex);
  std::map<void*,int>::iterator itr = allocatedBlockSize.find(base);
  if (itr == allocatedBlockSize.end()) {
      //assert(0);
      pthread_mutex_unlock(&mutex);
      return ;
  }

  int size = itr->second;
  allocatedBlockSize.erase(itr);
  //store the block into freeList in order to reuse in the furture
  freeList.insert(std::map<void*,MemoryChunk*>::value_type(base, new MemoryChunk(base, size)));
  pthread_mutex_unlock(&mutex);
#else
  CudaSafe(cudaFreeHost(base));
#endif
}

void* PortableMemoryManager::notEnoughMemorySlot(int size) {
  //printf("not enough portable memory slot\n");
  //assert(0);

  return allocMemory(size);
}


DeviceMemoryManager::DeviceMemoryManager(Device_Thread aThreadId) :
  threadId(aThreadId) {
  Construct(0,0,0);
  pthread_mutex_init(&mutex, NULL);
  sem_init(&memory_sem, 0, 0);
}

DeviceMemoryManager::~DeviceMemoryManager() {
  Destruct();
  pthread_mutex_destroy(&mutex);
}

void* DeviceMemoryManager::allocMemory(int size) {
  void* base;

#ifdef ENABLE_MEM_POOL
  //1st: check whether block with the same size already in the freeList
  pthread_mutex_lock(&mutex);
  for (std::map<void*,MemoryChunk*>::iterator itr = freeList.begin();
      itr != freeList.end();) {
    if (itr->second->getSize() == size) {
      base = itr->second->getBaseAddr();
      freeList.erase(itr++);
      allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));
      pthread_mutex_unlock(&mutex);
      return base;
    } else ++itr;
  }

  pthread_mutex_unlock(&mutex);
  //2nd: no block with the same size in the freeList, so try to allocate a new one
  Cudas[threadId].LockAndPushContext();
  cudaError_t result = cudaMalloc(&base, size);
  if (result==cudaSuccess) {
    allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));
    Cudas[threadId].ReleaseAndPopContext();
    return base;
  }

  //3rd: allocate memory failed, try to free block nearby to get a new one, 
  //but if all block in freeList are freed and still can't get a block by cudaMalloc, 
  //then alert.
  pthread_mutex_lock(&mutex);
  int remainSize = size;
  do {
    if (freeList.size() == 0) {
      //assert(0);
      sem_wait(&memory_sem);
    }
    int totalSize =0;
    for (std::map<void*,MemoryChunk*>::iterator itr = freeList.begin();
        itr != freeList.end();) {
      CudaSafe(cudaFree(itr->second->getBaseAddr()));
      totalSize += itr->second->getSize();
      delete itr->second;
      freeList.erase(itr++);
      if (totalSize > remainSize) break;
    }
    remainSize = 0;
  } while (cudaMalloc(&base, size) != cudaSuccess);
  pthread_mutex_unlock(&mutex);

  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error:  %s.\n", cudaGetErrorString( err) );
  }
  allocatedBlockSize.insert(std::map<void*,int>::value_type(base, size));
  Cudas[threadId].ReleaseAndPopContext();

#else
  Cudas[threadId].LockAndPushContext();
  CudaSafe(cudaMalloc(&base, size));
  Cudas[threadId].ReleaseAndPopContext();
#endif

  return base;
}

void DeviceMemoryManager::freeMemory(void* base) {
#ifdef ENABLE_MEM_POOL
  pthread_mutex_lock(&mutex);
  std::map<void*,int>::iterator itr = allocatedBlockSize.find(base);
  if (itr == allocatedBlockSize.end()) {
      //assert(0);
      pthread_mutex_unlock(&mutex);
      return ;
  }

  int size = itr->second;
  allocatedBlockSize.erase(itr);
  //store the block into freeList in order to reuse in the furture
  freeList.insert(std::map<void*,MemoryChunk*>::value_type(base, new MemoryChunk(base, size)));
  pthread_mutex_unlock(&mutex);
  sem_post(&memory_sem);
#else
  Cudas[threadId].LockAndPushContext();
  CudaSafe(cudaFree(base));
  Cudas[threadId].ReleaseAndPopContext();
#endif
}

void* DeviceMemoryManager::notEnoughMemorySlot(int size) {
  //printf("not enough device memory slot\n");
  //assert(0);
  return allocMemory(size);
}


CPUMemoryManager::CPUMemoryManager() {
  //Construct(2*1024,10000,3);
#ifdef ENABLE_MEM_POOL
  currentBase = new char[MEM_BLOCK_SIZE];
  currentSize = 0;
  additionalMemorySlot.push_back(currentBase);
#endif
}

CPUMemoryManager::~CPUMemoryManager() {
  //Destruct();

#ifdef ENABLE_MEM_POOL
  for (std::vector<void*>::iterator itr = additionalMemorySlot.begin();
      itr != additionalMemorySlot.end(); ++itr) 
    delete (char*)*itr;

  delete (char*)currentBase;
#endif
}

void* CPUMemoryManager::allocMemory(int size) {
  return new char[size];
}

void CPUMemoryManager::freeMemory(void* base) {
  delete (char*)base;
}

void* CPUMemoryManager::notEnoughMemorySlot(int size) {
  char* tmp = new char[size];
  additionalMemorySlot.push_back(tmp);
  return tmp;
}

void* CPUMemoryManager::allocBlock(int size) {
  
#ifdef ENABLE_MEM_POOL
  if (currentSize+size > MEM_BLOCK_SIZE) {
    additionalMemorySlot.push_back(currentBase);
    currentBase = new char[MEM_BLOCK_SIZE];
    currentSize =0;
  }

  void* tmpAddr = (char*)currentBase + currentSize;
  currentSize += size;
#else
  void* tmpAddr = this->allocMemory(size);
#endif

  return tmpAddr;
}

OMPRresult CPUMemoryManager::freeBlock(void* ) {
}

MemoryManagerBase::MemoryManagerBase() {
  pthread_mutex_init(&mutex, NULL);
}

MemoryManagerBase::~MemoryManagerBase() {
  pthread_mutex_destroy(&mutex);
}

OMPRresult MemoryManagerBase::Construct(int maxBlock, int numberPerSize, int sizeLevels) {
#ifdef ENABLE_MEM_POOL
  this->maxMemBlock = maxBlock;
  this->numberPerSize = numberPerSize;
  this->sizeLevel = sizeLevels;

  if (maxBlock ==0
      || numberPerSize ==0
      || sizeLevels==0)
    return SUCCESS;
  
  int size = maxBlock;
  int reMul=1;
  for (int i=0; i<sizeLevels; i++) reMul <<= 1;
  int totalSize = maxBlock*numberPerSize*2/reMul*(reMul-1);
  memBase = this->allocMemory(totalSize);

  char* accumAddr = (char*)memBase;
  baseArray = new void*[numberPerSize*sizeLevels];
  for (int i=0; i<sizeLevels; i++) {
    for (int j=0; j<numberPerSize; j++) {
      baseArray[i*numberPerSize+j] = accumAddr;
      accumAddr += size;
    }
    size >>= 1;
  }

  useFlags =0;

  useFlags= new unsigned int[(numberPerSize*sizeLevels+31)/32];
  memset(useFlags, 0, (numberPerSize*sizeLevels+31)/32*sizeof(int));
#endif
}

OMPRresult MemoryManagerBase::Destruct() {
#ifdef ENABLE_MEM_POOL
  this->freeMemory(memBase);
  delete baseArray;
  delete useFlags;
#endif
}

void* MemoryManagerBase::allocBlock(int size) {
#ifdef ENABLE_MEM_POOL
  if (size <= maxMemBlock) {
    pthread_mutex_lock(&mutex);
    int i=0, lowerSize = maxMemBlock;
    while (size <= lowerSize && i<numberPerSize*sizeLevel) {
      lowerSize >>= 1;
      i += numberPerSize;
    }
    i--;

    while ((useFlags[i/32] & (((int)1) << (i%32))) && i>=0) i--;

    if (i<0) {
      void* tmpR = notEnoughMemorySlot(size);
      pthread_mutex_unlock(&mutex);
      return tmpR;
    }
    
    useFlags[i/32] |= (((unsigned int)1) << (i%32));

    pthread_mutex_unlock(&mutex);
    return baseArray[i];
  } else {
    void* tmp;
    tmp = this->allocMemory(size);
    return tmp;
  }
#else
  return this->allocMemory(size);
#endif
}


OMPRresult MemoryManagerBase::freeBlock(void* base) {
#ifdef ENABLE_MEM_POOL
  pthread_mutex_lock(&mutex);
  for (int i=0; i<numberPerSize*sizeLevel; i++)
    if (baseArray[i] == base) {
      useFlags[i/32] &= ~(((unsigned int)1)<<(i%32));
      pthread_mutex_unlock(&mutex);
      return SUCCESS;
    }

  //CudaSafe(cudaFreeHost(base));
  this->freeMemory(base);
  pthread_mutex_unlock(&mutex);
#else
  this->freeMemory(base);
#endif
  return SUCCESS;
}


#undef ENABLE_MEM_POOL
  

extern DeviceMemoryManager* DevMemManager[GPU_NUM];
static int memorySpace = 0;


HRESULT GPUAlloc(Device_Thread thread, void** Addr, int size) {
  //Cudas[thread].LockAndPushContext();
  //CudaSafe(cudaMalloc(Addr, size));
  *Addr = DevMemManager[thread]->allocBlock(size);
  //Cudas[thread].ReleaseAndPopContext();
  memorySpace += size/(1024*1024);
  //printf("pid %d allocated %d\n",thread,size);
  //printf("pid %d allocated %gKB\n memSpace: %d\n",thread,(float)size/1024,memorySpace);
  return SUCCESS;
}

HRESULT GPUFree(Device_Thread thread, void* Addr) {
  //Cudas[thread].LockAndPushContext();
  //CudaSafe(cudaFree(Addr));
  //Cudas[thread].ReleaseAndPopContext();
  DevMemManager[thread]->freeBlock(Addr);
  
  return SUCCESS;
}





#ifndef OMPR_HEAP_H
#define OMPR_HEAP_H
#include "omp_runtime_global.h"
#include "ompr_dsm.h"
#include <pthread.h>
#include <semaphore.h>


HRESULT GPUAlloc(Device_Thread thread, void** Addr, int size);
HRESULT GPUFree(Device_Thread thread, void* Addr);

#define MAX_MEM_BLOCK       64*1024*1024
#define NUMBER_PER_SIZE     2
#define SIZE_LEVEL          4

class MemoryManagerBase {

  private:
    void* memBase;
    void** baseArray;
    unsigned int *useFlags; //the last bit stands for baseArray[0], and so on...
    int maxMemBlock, numberPerSize, sizeLevel;

    pthread_mutex_t mutex;

  protected:
    virtual void* allocMemory(int size) =0;
    virtual void freeMemory(void* base) =0;
    virtual void* notEnoughMemorySlot(int size) = 0; 

    OMPRresult Construct(int maxBlock=MAX_MEM_BLOCK, int numberPerSize=NUMBER_PER_SIZE, 
        int sizeLevels=SIZE_LEVEL);
    OMPRresult Destruct();

  public:
    MemoryManagerBase();
    ~MemoryManagerBase();

    void* allocBlock(int size);
    OMPRresult freeBlock(void*);

};


class MemoryChunk {
  private:
    void* base;
    int size;

  public:
    MemoryChunk(void* addr, int aSize) :
      base(addr), size(aSize) {}

    void* getBaseAddr() {return base;}
    int getSize() { return size; }
};

class PortableMemoryManager : 
  public MemoryManagerBase {

  private:
    std::map<void*,MemoryChunk*> freeList;
    std::map<void*, int> allocatedBlockSize;
    pthread_mutex_t mutex;

  protected:
    virtual void* allocMemory(int size);
    virtual void freeMemory(void* base);
    virtual void* notEnoughMemorySlot(int size);

  public:
    PortableMemoryManager();
    ~PortableMemoryManager();

};

class DeviceMemoryManager :
  public MemoryManagerBase {

  private:
    Device_Thread threadId;
    
    std::map<void*,MemoryChunk*> freeList;
    std::map<void*, int> allocatedBlockSize;
    pthread_mutex_t mutex;

    sem_t memory_sem;

  protected:
    virtual void* allocMemory(int size);
    virtual void freeMemory(void* base);
    virtual void* notEnoughMemorySlot(int size);

  public:
    DeviceMemoryManager(Device_Thread);
    ~DeviceMemoryManager();

};

class CPUMemoryManager :
  public MemoryManagerBase {

  private:
    std::vector<void*> additionalMemorySlot;
    void* currentBase;
    int currentSize;
    static const int MEM_BLOCK_SIZE=1024*1024;

  protected:
    virtual void* allocMemory(int size);
    virtual void freeMemory(void* base);
    virtual void* notEnoughMemorySlot(int size);


  public:
    CPUMemoryManager();
    ~CPUMemoryManager();

    void* allocBlock(int size);
    OMPRresult freeBlock(void* base);
};

#endif

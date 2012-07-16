#ifndef OMPRPREFETCH_H
#define OMPRPREFETCH_H
#include <iostream>
#include <vector>
#include <queue> 
#include <map>
#include <semaphore.h>

#include "omp_runtime_global.h"

//#include <pthread>

//Prefetch data shape
class BlockKey;
typedef struct Region {
    //to find DataObjState
    BlockKey *key;
    //other info;
    int dims;
    ElementType eleType;
    int len[DSM_MAX_DIMENSION];
    int loff[DSM_MAX_DIMENSION];
    int uoff[DSM_MAX_DIMENSION];
    Region(BlockKey& key,int dims,ElementType type,int* len,int* loff,int* uoff);
};

//void* prefetch_worker(void* arg);

typedef enum {GLOBAL = 0,LOCAL}Scope;
typedef enum {USE = 0,DEF}UDType ;

class DFItem {
public:
	UDType udType;
	Region* region;
	DFItem(UDType udType,Region* region);
};

class DFEntry {
	static int entryId[2];
private:
	Scope scope;
	std::vector<DFItem*> items;
	pthread_mutex_t lock;
	unsigned int id;

public:
	DFEntry(int pid);
	DFEntry(int pid,Scope scope);
	~DFEntry();
	HRESULT addTraceItem(int pid,UDType udType,Region* region);
	HRESULT addTraceItem(DFItem* item);
	//DFEntry(const DFEntry& entry);
	void setScope(Scope scope);
	Scope getScope();
	int getId();
    void clear();
	std::vector<DFItem*>::iterator begin();
	std::vector<DFItem*>::iterator end();
};

//one GPU/core's Data Flow Trace
typedef std::deque<DFEntry*> DFTrace;

//pend for prefetch worker thread prefetch
class PendEntry{
public:
	//src datablock to dest datablock
	Region *srcRegion,*destRegion;

	//intersect shape
	int shape[DSM_MAX_DIMENSION];
	int dimsOffset[DSM_MAX_DIMENSION];
    cudaStream_t srcStream;
    cudaStream_t destStream;

	int number;
	PendEntry(Region *srcRegion,Region *destRegion,int *dimsOffset,int *shape,int number,
            cudaStream_t& srcStream,cudaStream_t& destStream);
};

//one GPU/core's data prefetch queue
//use muitl_map to handle find
typedef std::multimap<int,PendEntry*> PendQueue;
//typedef std::queue<PendEntry*> PendQueue;

//a class manage n pengQueue logic
//when a new intersect region can prefetch new a pendentry
class DSMPendQueueManger{
private:
	const int num;
	PendQueue pendQueue[2];
    pthread_rwlock_t modifyQueue[2];
public :
	DSMPendQueueManger();
	//assume to n GPU
	DSMPendQueueManger(int n);
	~DSMPendQueueManger();

	//when compute a new region to prefetch add it to queue
	HRESULT addEntry(int pid,cudaStream_t& srcStream,cudaStream_t& destStream,
            Region* srcRegion,Region* destRegion,
            int *dimsOffset,int *shape,int number);

	//if agent thread reach one barrier ,prefetch possible region
	HRESULT recvAgent(int pid,int number,std::vector<PendEntry*>& arrivedEntry);
	//pendqueue is empty?
	bool empty();

	//protected:
	//    HRESULT startPendPrefetch(int pid,PendEntry *pendEntry);
};


//a class manage n DFtrace logic
//when a new omp for reach add a new Entry in trace
//when a barrier reach add a new Entry in trace ,and get the UD data from other GPU thread
class DSMDFTraceManager{
	static DFEntry* sumEntry[2];
private:
	enum LoopState{
		LOOPIN =0,LOOPOUT
	}loopState;
	const int num;

    //manager two trace
	DFTrace trace[2];
    pthread_rwlock_t modifyTrace[2];

	DSMPendQueueManger pendQueueMgr;

	pthread_barrier_t barrierIn;
	pthread_barrier_t barrierOut;

	//sem for barrier reach
	sem_t semAgentBar[2];
    sem_t semWorkerEnd[2];

	//after sem_t,worker need this numid to start prefetch 
	volatile int barrierNumId[2];
	volatile int threadId;

protected:
	//use for sum all entry this thread between two barrier
	void sumThisThread(int pid);

public:
    pthread_t pthreads[2];

	DSMDFTraceManager();
	~DSMDFTraceManager();
	DSMDFTraceManager(int n);

    //stop worker thread
    void stopWorkerThreads();
	//trace head-tail op
	//public bool pushHead(int pid,DFEntry);
	bool pushTail(int pid,DFEntry* dfEntry);
	int popHead(int pid);
	//public bool popTail(int pid);
	//
	//add one item in tail entry
	void addItem(int pid,UDType udType,Region* region);

	//compute intersect shape and pend prefetch region
	HRESULT findAndPendIntersect(int pid,cudaStream_t& srcStream,cudaStream_t& destStream,Region* region);

	//two threads enter a barrier, exchange Use-Def data with others 
	//HRESULT addBarrier(int pid);//two thread must in and out  at the same time

	//interface with dsm runtime,when a DSM input occur , it calls it 
	HRESULT prefetchInput(int pid,int streamId,BlockKey& key,int dims,ElementType type,
        int* len,int* loff,int* uoff); 
	HRESULT prefetchOutput(int pid,int streamId,BlockKey& key,int dims,ElementType type,
        int* len,int* loff,int* uoff); 
	//HRESULT loopEnd();
	HRESULT agentBarrierInput(int pid); 
	HRESULT prefetchBarrierInput(int pid); 

	friend void* prefetch_worker(void* arg);    
};


#endif

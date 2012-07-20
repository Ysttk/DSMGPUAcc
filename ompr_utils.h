#ifndef OMPR_UTILS_H
#define OMPR_UTILS_H

#include "ompr_dsm.h"
#include "omp_runtime_global.h"
#include <assert.h>
#include <vector>
#include <map>

typedef char  BOOL;

#define TRUE  1
#define FALSE 0

template <class K, class V>
class SafeMapIterator{
  typename std::map<K,V>::iterator itr;
  pthread_rwlock_t* pLock;

  public:
  typedef typename std::map<K,V>::iterator StdMapItr;

  SafeMapIterator(StdMapItr aItr, pthread_rwlock_t* aPLock): 
    itr(aItr),pLock(aPLock) {}

  const typename std::map<K,V>::iterator& getStdMapItr() const { return itr; }

  pthread_rwlock_t* getPLock() const { return pLock; }

  SafeMapIterator(const SafeMapIterator& aItr): itr(aItr.getStdMapItr()),
    pLock(aItr.getPLock()) {}

  void operator= (SafeMapIterator& aItr) {
    itr= aItr.getStdMapItr();
    pLock = aItr.getPLock();
  }

  void operator++ () {
    pthread_rwlock_rdlock(pLock);
    ++itr;
    pthread_rwlock_unlock(pLock);
  }

  typename std::map<K,V>::value_type*
    operator->() {
      return &(*itr);
    }


  friend inline bool operator!= (const SafeMapIterator& itr1,
      const SafeMapIterator& itr2) {
    /*
    pthread_rwlock_rdlock(itr1.pLock);
    pthread_rwlock_rdlock(itr2.pLock);
    */
    bool result = itr1.itr!=itr2.itr;
    /*
    pthread_rwlock_unlock(itr2.pLock);
    pthread_rwlock_unlock(itr1.pLock);
    */
    return result;
  }

};


template <class K, class V>
class SafeMap {
  private:
    std::map<K,V>* kvContainer;
    pthread_rwlock_t lock;
    std::vector<SafeMapIterator<K,V>*> itrs;

  public:
    typedef std::pair<const K, V> value_type;
    typedef SafeMapIterator<K,V> iterator;

    SafeMap() {
      kvContainer = new std::map<K,V>();
      pthread_rwlock_init(&lock, NULL);
    }

    iterator begin() {
      pthread_rwlock_rdlock(&lock);
      iterator itr(kvContainer->begin(), &lock);
      pthread_rwlock_unlock(&lock);
      return itr;
    }

    iterator end() {
      pthread_rwlock_rdlock(&lock);
      iterator itr(kvContainer->end(), &lock);
      pthread_rwlock_unlock(&lock);
      return itr;
    }

    void insert(const value_type& v) {
      pthread_rwlock_wrlock(&lock);
      kvContainer->insert(v);
      pthread_rwlock_unlock(&lock);
    }

    void erase(iterator& itr) {
      pthread_rwlock_wrlock(&lock);
      kvContainer->erase(itr.getStdMapItr());
      pthread_rwlock_unlock(&lock);
    }


    ~SafeMap() {
      delete kvContainer;
      pthread_rwlock_destroy(&lock);
    }
};



  

/*
//#define CuSafe(x) (assert( (x) == CUDA_SUCCESS))
#define CuSafe(x)   { \
		      pthread_mutex_lock(&cudaFuncLock); \
		      CUresult re = x; \
		      assert( re == CUDA_SUCCESS); \
		      pthread_mutex_unlock(&cudaFuncLock); \
		    } 
//#define CudaSafe(x) (assert( x == cudaSuccess))
#define CudaSafe(x) { \
		      pthread_mutex_lock(&cudaFuncLock);\
		      cudaError_t re = x; \
		      assert( re == cudaSuccess); \
		      pthread_mutex_unlock(&cudaFuncLock); \
		    }
        */
#define CuSafe(x)   { \
		      CUresult re = x; \
		      assert( re == CUDA_SUCCESS); \
		    }
#define CudaSafe(x) { \
		      cudaError_t re = x; \
		      assert( re == cudaSuccess); \
		    }
#define UP_CEIL(x,y)  (((x)+(y-1))/(y))
#define LOW_CEIL(x,y) ((x)/(y))
#define MAX(x,y)    ((x)>(y)?(x):(y))
#define MIN(x,y)    ((x)>(y)?(y):(x))

int ElementSize(ElementType type);

class BlockKey;
HRESULT CheckAndInitDataBlockState(BlockKey& key,void* base,int* shape,int* lens,
    int* offset,int dims,ElementType eType,int* loff,int* uoff,bool initialize );

/*
 * if rec1 equal to rec2, return EQUAL,
 * else return NOT_EQUAL
 */
template <typename RecType>
HRESULT RecEqual(RecType* rec1, RecType* rec2, int dim) {
  for (int i=dim-1; i>=0; i--) {
    if (rec2->LB(i)!=rec1->LB(i) ||
        rec2->UB(i)!=rec1->UB(i))
      return NOT_EQUAL;
  }
  return SUCCESS;
}

/*
 * In this function, rec2 will make intersection with rec1 and rec1 will be splited by rec2
 *
 * RecType must have five functions, one is LB(idx) for return the lower bound of the 
 * pointed dimention; another UB(idx) for the upper bound of the pointed dimention;
 * the third one is constructor RecType(RecType*); the fourth one is SetDimBound(idx,lb,ub);
 * the fifth is SetAllDimBound(int lb[DSM_MAX_DIMENSION], int ub[DSM_MAX_DIMENSION]).
 *
 * This function can be devide into two parts:
 * first step, select all of the avalible line for each dimention;
 * second step, make a full combination of all the avalible line from each dimention
 * to get all rectangles.
 */
//step 1 function
template <typename RecType>
HRESULT IntersectAndSplitRec(RecType* rec1, RecType* rec2, int dim, int* lineSegN,
    int lineSegment[3][2][DSM_MAX_DIMENSION]);

//step 1 and 2 function
template <typename RecType>
HRESULT IntersectAndSplitRec(RecType* rec1, RecType* rec2, int dim, 
    RecType* intersectRec, std::vector<RecType*>* splitList) {
  int lineSegN[DSM_MAX_DIMENSION];
  int lineSegment[3][2][DSM_MAX_DIMENSION]; 
    //the last dimention: 0 for lower bound, 1 for upper bound
    //the first dimention: 0 for intersection rectangle

  //first step
  if (IntersectAndSplitRec<RecType>(rec1, rec2, dim, lineSegN, lineSegment)
      ==NO_UNION_BETWEEN_RECS)
    return NO_UNION_BETWEEN_RECS;

  //second step
  intersectRec->SetAllDimBound(&lineSegment[0][0][0], &lineSegment[0][1][0]);
  for (int j=0; j<lineSegN[0]; j++) {
    RecType* rec = new RecType();
    rec->SetDimBound(0, lineSegment[j][0][0], lineSegment[j][1][0]);
    splitList->push_back(rec);
  }

  for (int i=1; i<dim; i++) {
    int currentListSize = splitList->size();
    for (int j=0; j<currentListSize; j++) 
      (*splitList)[j]->SetDimBound(i, lineSegment[0][0][i], lineSegment[0][1][i]);

    int j=1;
    while (j<lineSegN[i]) {
      for (int k=0; k<currentListSize; k++) {
        RecType* tmpRec = new RecType((*splitList)[k]);
        tmpRec->SetDimBound(i, lineSegment[j][0][i], lineSegment[j][1][i]);
        splitList->push_back(tmpRec);
      }
      j++;
    }
  }

  for (typename std::vector<RecType*>::iterator itr=splitList->begin(); 
      itr!=splitList->end(); ++itr) {
    if (RecEqual<RecType>((*itr), intersectRec, dim)==SUCCESS) {
      splitList->erase(itr);
      break;
    }
  }
  
  return SUCCESS;

}

template <typename RecType>
HRESULT IntersectAndSplitRec(RecType* rec1, RecType* rec2, int dim, int* lineSegN,
    int lineSegment[3][2][DSM_MAX_DIMENSION]) {
  
  for (int i=0; i<dim; i++) {
    if (rec1->LB(i)>rec2->UB(i) || rec2->LB(i)>rec1->UB(i)) {
      return NO_UNION_BETWEEN_RECS;
    }

    if (rec1->LB(i)<rec2->LB(i) && rec1->UB(i)<=rec2->UB(i)) {
      lineSegN[i]=2;
      lineSegment[0][0][i]=rec2->LB(i); lineSegment[0][1][i]=rec1->UB(i);
      lineSegment[1][0][i]=rec1->LB(i); lineSegment[1][1][i]=rec2->LB(i)-1;
    } else if (rec1->LB(i)<rec2->LB(i) && rec1->UB(i)>rec2->UB(i)) {
      lineSegN[i]=3;
      lineSegment[0][0][i]=rec2->LB(i); lineSegment[0][1][i]=rec2->UB(i);
      lineSegment[1][0][i]=rec1->LB(i); lineSegment[1][1][i]=rec2->LB(i)-1;
      lineSegment[2][0][i]=rec2->UB(i)+1; lineSegment[2][1][i]=rec1->UB(i);
    } else if (rec1->LB(i)>=rec2->LB(i) && rec1->UB(i)>rec2->UB(i)) {
      lineSegN[i]=2;
      lineSegment[0][0][i]=rec1->LB(i); lineSegment[0][1][i]=rec2->UB(i);
      lineSegment[1][0][i]=rec2->UB(i)+1; lineSegment[1][1][i]=rec1->UB(i);
    } else if (rec1->LB(i)>=rec2->LB(i) && rec1->UB(i)<=rec2->UB(i)) {
      lineSegN[i]=1;
      lineSegment[0][0][i]=rec1->LB(i); lineSegment[0][1][i]=rec1->UB(i);
    }

  }

  return SUCCESS;
}

/*
 * This function return SUCCESS if rec1 and rec2 has intersection,
 * return NO_UNION_BETWEEN_RECS if rec1 and rec2 has no intersection
 */
template <typename RecType>
HRESULT IsIntersect(RecType* rec1, RecType* rec2, int dim) {
  for (int i=dim-1; i>=0; i--) {
    if (rec1->LB(i)>rec2->UB(i) || rec1->UB(i)<rec2->LB(i))
      return NO_UNION_BETWEEN_RECS;
  }

  return SUCCESS;
}


/*
 * if rec1 is contained by rec2, return CONTAINED, 
 * else return NOT_CONTAINED
 */
template <typename RecType>
HRESULT RecContain(RecType* rec1, RecType* rec2,int dim) {
  for (int i=dim-1; i>=0; i--) {
    if (rec2->LB(i)>rec1->LB(i) ||
        rec2->UB(i)<rec1->UB(i))
      return NOT_CONTAINED;
  }
  return CONTAINED;
}


void checkCUDAError(const char* msg);
double gettime();

pid_t getPrefetchIdFromAgentId(pid_t pid);

pid_t getAgentIdFromPrefetchId(pid_t pid);

pid_t getPrefetchIdFromWorkId(pid_t pid);
bool isAgentId(pid_t threadId);
bool isPrefetchId(pid_t threadId);

#endif


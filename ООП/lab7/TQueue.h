#ifndef TQUEUE_H
#define	TQUEUE_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include "TQueueItem.h"
#include "TIterator.h"
#include "IRemoveCriteria.h"
#include <memory>

template <class T, class TT> class TQueue {
public:
    TQueue();

    void PushSubitem(std::shared_ptr<TT> figure);
    void DeleteSubitem(IRemoveCriteria<TT> * criteria);
    void Push(std::shared_ptr<TT> figure);
    void Pop();
    bool Empty();

    template <class A, class AA> friend std::ostream& operator<<(std::ostream& os,const TQueue<A,AA>& queue);

    TIterator<TQueueItem<T,TT>,T> begin();
    TIterator<TQueueItem<T,TT>,T> end();

    std::shared_ptr<TQueueItem<T,TT>> head;
    std::shared_ptr<TQueueItem<T,TT>> tail;

    virtual ~TQueue();
private:
    
  
};

#define TQUEUE_FUNCTIONS
#include "TQueue.cpp"

#endif	
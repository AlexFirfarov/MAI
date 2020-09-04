#ifndef TQUEUE_H
#define	TQUEUE_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include "TQueueItem.h"
#include <memory>

template <class T> class TQueue {
public:
    TQueue();

    void push(std::shared_ptr<T> &&figure);
    std::shared_ptr<T> pop();
    bool empty();
    template <class A> friend std::ostream& operator<<(std::ostream& os,const TQueue<A>& queue);
    virtual ~TQueue();
private:
    
    std::shared_ptr<TQueueItem<T>> head;
    std::shared_ptr<TQueueItem<T>> tail;
};

#define TQUEUE_FUNCTIONS
#include "TQueue.cpp"

#endif	
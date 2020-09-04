#ifndef TQUEUEITEM_H
#define	TQUEUEITEM_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include "Block.h"
#include <memory>

template <class T> class TQueueItem {
public:
    TQueueItem(const std::shared_ptr<T>& figure);
    template <class A> friend std::ostream& operator<<(std::ostream& os, const TQueueItem<A>& obj);
    
    void SetNext(std::shared_ptr<TQueueItem> &next);
    std::shared_ptr<TQueueItem<T>> GetNext();
    std::shared_ptr<T> GetFigure() const;

    void * operator new (size_t size);
    void operator delete(void *p);

    virtual ~TQueueItem();
private:
    std::shared_ptr<T> figure;
    std::shared_ptr<TQueueItem<T>> next;

    static TAllocationBlock queueItem_allocator;
};


#define TQUEUEITEM_FUNCTION
#include "TQueueItem.cpp"
#endif
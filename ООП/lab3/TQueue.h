#ifndef TQUEUE_H
#define	TQUEUE_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include "TQueueItem.h"
#include <memory>

class TQueue {
public:
    TQueue();

    void push(std::shared_ptr<Figure> &&figure);
    std::shared_ptr<Figure> pop();
    bool empty();
    friend std::ostream& operator<<(std::ostream& os,const TQueue& queue);
    virtual ~TQueue();
private:
    
    std::shared_ptr<TQueueItem> head;
    std::shared_ptr<TQueueItem> tail;
};

#endif	
#ifndef TQUEUE_H
#define	TQUEUE_H

#include "Rectangle.h"
#include "TQueueItem.h"

class TQueue {
public:
    TQueue();

    void push(Rectangle &&rectangle);
    Rectangle pop();
    bool empty();
    friend std::ostream& operator<<(std::ostream& os,const TQueue& queue);
    ~TQueue();
private:
    
    TQueueItem *head;
    TQueueItem *tail;
};

#endif	
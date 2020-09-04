#ifndef TQUEUEITEM_H
#define	TQUEUEITEM_H

#include "Rectangle.h"
class TQueueItem {
public:
    TQueueItem(const Rectangle& rectangle);
    friend std::ostream& operator<<(std::ostream& os, const TQueueItem& obj);
    
    TQueueItem* SetNext(TQueueItem* next);
    TQueueItem* GetNext();
    Rectangle GetRectangle() const;

    virtual ~TQueueItem();
private:
    Rectangle rectangle;
    TQueueItem *next;
};

#endif
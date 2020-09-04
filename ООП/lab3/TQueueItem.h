#ifndef TQUEUEITEM_H
#define	TQUEUEITEM_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include <memory>


class TQueueItem {
public:
    TQueueItem(const std::shared_ptr<Figure>& figure);
    friend std::ostream& operator<<(std::ostream& os, const TQueueItem& obj);
    
    void SetNext(std::shared_ptr<TQueueItem> &next);
    std::shared_ptr<TQueueItem> GetNext();
    std::shared_ptr<Figure> GetFigure() const;

    virtual ~TQueueItem();
private:
    std::shared_ptr<Figure> figure;
    std::shared_ptr<TQueueItem> next;
};

#endif
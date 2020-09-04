#include "TQueueItem.h"
#include <iostream>

TQueueItem::TQueueItem(const Rectangle& rectangle) {
    this->rectangle = rectangle;
    this->next = nullptr;
}

TQueueItem* TQueueItem::SetNext(TQueueItem* next) {
    this->next = next;
    return this;
}

Rectangle TQueueItem::GetRectangle() const {
    return this->rectangle;
}

TQueueItem* TQueueItem::GetNext() {
    return this->next;
}

std::ostream& operator<<(std::ostream& os, const TQueueItem& obj) {
    os << obj.rectangle << std::endl;
    return os;
}

TQueueItem::~TQueueItem() {
    next = nullptr;
}
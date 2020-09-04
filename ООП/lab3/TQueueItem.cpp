#include "TQueueItem.h"
#include <iostream>

TQueueItem::TQueueItem(const std::shared_ptr<Figure>& figure) {
    this->figure = figure;
    this->next = nullptr;
}

void TQueueItem::SetNext(std::shared_ptr<TQueueItem>& next) {
    this->next = next;
    return;
}

std::shared_ptr<Figure> TQueueItem::GetFigure() const {
    return this->figure;
}

std::shared_ptr<TQueueItem> TQueueItem::GetNext() {
    return this->next;
}

std::ostream& operator<<(std::ostream& os, const TQueueItem& obj) {
    std::shared_ptr<Figure> figure = obj.figure;
    figure->Print();
    return os;
}

TQueueItem::~TQueueItem() {
}
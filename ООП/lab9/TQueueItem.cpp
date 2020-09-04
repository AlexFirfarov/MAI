#ifndef TQUEUEITEM_FUNCTION
#include "TQueueItem.h"
#include <iostream>

#else

template <class T> TQueueItem<T>::TQueueItem(const std::shared_ptr<T>& figure) {
    this->figure = figure;
    this->next = nullptr;
}

template <class T> void TQueueItem<T>::SetNext(std::shared_ptr<TQueueItem<T>>& next) {
    this->next = next;
    return;
}

template <class T> std::shared_ptr<T> TQueueItem<T>::GetFigure() const {
    return this->figure;
}

template <class T> std::shared_ptr<TQueueItem<T>> TQueueItem<T>::GetNext() {
    return this->next;
}

template <class A> std::ostream& operator<<(std::ostream& os, const TQueueItem<A>& obj) {
    os << *obj.figure << std::endl;
    return os;
}

template <class T> TQueueItem<T>::~TQueueItem() {
}

#endif
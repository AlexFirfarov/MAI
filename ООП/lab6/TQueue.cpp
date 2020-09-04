#ifndef TQUEUE_FUNCTIONS
#include "TQueue.h"

#else

template <class T> TQueue<T>::TQueue() {
    tail = nullptr;
    head = nullptr;
}

template <class T> std::ostream& operator<<(std::ostream& os, const TQueue<T>& queue) {

    std::shared_ptr<TQueueItem<T>> item = queue.head;
    
    while(item != nullptr) {

        std::shared_ptr<Figure> t = item->GetFigure();
        t->Print();
        std::cout << '\n';
        item = item->GetNext();
    }
    
    return os;
}

template <class T> void TQueue<T>::push(std::shared_ptr<T> &&figure) {

    std::shared_ptr<TQueueItem<T>> other(new TQueueItem<T>(figure)); 
    if (empty()) {
        head = other;
        tail = other;
        std::shared_ptr<TQueueItem<T>> empty = nullptr;
        other->SetNext(empty);
        return;
    }

    tail->SetNext(other);
    tail = other;
    std::shared_ptr<TQueueItem<T>> empty = nullptr;
    tail->SetNext(empty);
}

template <class T> bool TQueue<T>::empty() {
    return head == nullptr;
}

template <class T> std::shared_ptr<T> TQueue<T>::pop() {

    std::shared_ptr<T> result;
    if (!empty()) {
        std::shared_ptr<TQueueItem<T>> old_head = head;
        head = head->GetNext();
        result = old_head->GetFigure();
        std::shared_ptr<TQueueItem<T>> empty = nullptr;
        old_head->SetNext(empty);
        return result;
    }
    else {
        std::cout << "Очередь пуста " << std::endl;
        return nullptr;
    }  
}

template <class T> TIterator<TQueueItem<T>,T> TQueue<T>::begin() {
    return TIterator<TQueueItem<T>,T>(head);
}

template <class T> TIterator<TQueueItem<T>,T> TQueue<T>::end() {
    return TIterator<TQueueItem<T>,T>(nullptr);
}

template <class T> TQueue<T>::~TQueue() {
    tail = nullptr;
    head = nullptr;
    std::cout << "Очередь удалена" << std::endl; 
}

#endif
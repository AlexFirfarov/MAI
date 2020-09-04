#ifndef TQUEUE_FUNCTIONS
#include "TQueue.h"
#include <exception>

#else

template <class T> TQueue<T>::TQueue() {
    tail = nullptr;
    head = nullptr;
}

template <class T> std::shared_ptr<T> TQueue<T>::operator[] (size_t ind) {
    if (ind > size() - 1) throw std::invalid_argument("Индекс больше размера очереди");
    size_t i = 0;
    for (auto a: *this) {
        if (i == ind) return a;
        ++i;
    }
    return std::shared_ptr<T>(nullptr);
}

template <class T> size_t TQueue<T>::size() {
    int result = 0;
    for (auto a: *this) ++result;
    return result;
}

template <class T> void TQueue<T>::sort() {
    if (size() > 1) {
        std::shared_ptr<T> middle = pop();
        TQueue<T> left, right;

        while (!empty()) {
            std::shared_ptr<T> item = pop();
            if (*item < *middle) {
                left.push(std::move(item));
            }
            else {
                right.push(std::move(item));
            }
        }
        left.sort();
        right.sort();

        while (!left.empty()) push(std::move(left.pop()));
        push(std::move(middle));
        while (!right.empty()) push(std::move(right.pop()));
    }
}

template <class T> std::future<void> TQueue<T>::sort_in_background() {
    std::packaged_task<void(void)>
    task(std::bind(std::mem_fn(&TQueue<T>::sort_parallel), this));
    std::future<void> res(task.get_future());
    std::thread th(std::move(task));
    th.detach();
    return res;
}

template <class T> void TQueue<T>::sort_parallel() {
    if (size() > 1) {
        std::shared_ptr<T> middle = pop();
        TQueue<T> left, right;

        while(!empty()) {
            std::shared_ptr<T> item = pop();
            if (*item < *middle) {
                left.push(std::move(item));
            }
            else {
                right.push(std::move(item));
            }
        }
        std::future<void> left_res = left.sort_in_background();
        std::future<void> right_res = right.sort_in_background();

        left_res.get();
        while (!left.empty()) push(std::move(left.pop()));
        push(std::move(middle));
        right_res.get();
        while (!right.empty()) push(std::move(right.pop()));
    }
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
    while (!empty()) {
        std::shared_ptr<TQueueItem<T>> temp = head;
        head = head->GetNext();
    }
    tail = nullptr;
    head = nullptr;
}

#endif
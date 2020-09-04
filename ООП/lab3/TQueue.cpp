#include "TQueue.h"

TQueue::TQueue() {
    tail = nullptr;
    head = nullptr;
}

std::ostream& operator<<(std::ostream& os, const TQueue& queue) {

    std::shared_ptr<TQueueItem> item = queue.head;
    
    while(item != nullptr) {

        std::shared_ptr<Figure> t = item->GetFigure();
        t->Print();
        std::cout << '\n';
        item = item->GetNext();
    }
    
    return os;
}

void TQueue::push(std::shared_ptr<Figure> &&figure) {

    std::shared_ptr<TQueueItem> other(new TQueueItem(figure)); 
    if (empty()) {
        head = other;
        tail = other;
        std::shared_ptr<TQueueItem> empty = nullptr;
        other->SetNext(empty);
        return;
    }

    tail->SetNext(other);
    tail = other;
    std::shared_ptr<TQueueItem> empty = nullptr;
    tail->SetNext(empty);
}

bool TQueue::empty() {
    return head == nullptr;
}

std::shared_ptr<Figure> TQueue::pop() {

    std::shared_ptr<Figure> result;
    if (!empty()) {
        std::shared_ptr<TQueueItem> old_head = head;
        head = head->GetNext();
        result = old_head->GetFigure();
        std::shared_ptr<TQueueItem> empty = nullptr;
        old_head->SetNext(empty);
        return result;
    }
    else {
        std::cout << "Очередь пуста " << std::endl;
        return nullptr;
    }  
}

TQueue::~TQueue() {
    while (!empty()) {
        std::shared_ptr<TQueueItem> temp = head;
        head = head->GetNext();
    }
    tail = nullptr;
    head = nullptr;
    std::cout << "Очередь удалена" << std::endl; 
}
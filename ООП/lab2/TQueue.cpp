#include "TQueue.h"

TQueue::TQueue() {
    tail = nullptr;
    head = nullptr;
}

std::ostream& operator<<(std::ostream& os, const TQueue& queue) {

    TQueueItem *item = queue.head;
    
    while(item != nullptr) {

        Rectangle t = item->GetRectangle();
        os << t << std::endl;
        item = item->GetNext();
    }
    
    return os;
}

void TQueue::push(Rectangle &&rectangle) {
    TQueueItem *other = new TQueueItem(rectangle);
    if (empty()) {
        head = other;
        tail = other;
        other->SetNext(nullptr);
        return;
    }
    tail->SetNext(other);
    tail = other;
    tail->SetNext(nullptr);
}

bool TQueue::empty() {
    return head == nullptr;
}

Rectangle TQueue::pop() {
    Rectangle result;
    if (!empty()) {
        TQueueItem *old_head = head;
        head = head->GetNext();
        result = old_head->GetRectangle();
        old_head->SetNext(nullptr);
        delete old_head;
        return result;
    }
    else {
        std::cout << "Очередь пуста " << std::endl;
    }  
}

TQueue::~TQueue() {
    while (!empty()) {
        TQueueItem *temp = head;
        head = head->GetNext();
        delete temp;
    }
    tail = nullptr;
    head = nullptr;
    std::cout << "Очередь удалена" << std::endl;
}
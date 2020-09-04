#ifndef TQUEUE_FUNCTIONS
#include "TQueue.h"

#else

template <class T, class TT> TQueue<T, TT>::TQueue() {
    tail = nullptr;
    head = nullptr;
}

template <class A, class AA> std::ostream& operator<<(std::ostream& os, const TQueue<A, AA>& queue) {
     for (auto i: queue) {
        i->PrintTree(); 
        std::cout << '\n';
    }
}

template <class T, class TT> void TQueue<T,TT>::Push(std::shared_ptr<TT> figure) {
    std::shared_ptr<TQueueItem<T,TT>> other(new TQueueItem<T,TT>()); 
    std::shared_ptr<TBinaryTree<TT>> newTree(new TBinaryTree<TT>(figure));
    other->SetTree(newTree);
    if (Empty()) {
        head = other;
        tail = other;
        std::shared_ptr<TQueueItem<T,TT>> empty = nullptr;
        other->SetNext(empty);
        return;
    }

    tail->SetNext(other);
    tail = other;
    std::shared_ptr<TQueueItem<T,TT>> empty = nullptr;
    tail->SetNext(empty);
}

template <class T, class TT> void TQueue<T,TT>::Pop() { 
    if (Empty()) {
        std::cout << "Очередь пуста" << std::endl;
        return;
    }
    else {
        std::shared_ptr<TQueueItem<T,TT>> old_head = head;
        head = head->GetNext();
        std::shared_ptr<TQueueItem<T,TT>> empty = nullptr;
        old_head->SetNext(empty);
        return;
    }
}

template <class T, class TT> void TQueue<T,TT>::PushSubitem(std::shared_ptr<TT> figure) {
    if (Empty()) {
        Push(figure);
    }
    
    else if (tail->TGetSize() < 5) {
        tail->GetTree()->Insert(figure);
    }
    else {
        Push(figure);
    }
    return;    
}

template <class T, class TT> void TQueue<T,TT>::DeleteSubitem(IRemoveCriteria<TT> * criteria) {
        if (Empty()) {
            std::cout << "Очередь пуста" << std::endl;
            return;
        }

       int size = head->TGetSize();
       int count = 0;
       std::shared_ptr<TT> *temp = new std::shared_ptr<TT>[size];
       head->Inorder(criteria, temp, count);

        if (count == 0) {
            Pop();
            return;
        }

       std::shared_ptr<TBinaryTree<TT>> newTree(new TBinaryTree<TT>(temp[0]));
       for (int j = 1; j < count; ++j) {
           newTree->Insert(temp[j]);
       }
       head->Delete();
       head->SetTree(newTree);

    return;
}

template <class T, class TT> bool TQueue<T,TT>::Empty() {
    return head == nullptr;
}

template <class T, class TT> TIterator<TQueueItem<T,TT>,T> TQueue<T,TT>::begin() {
    return TIterator<TQueueItem<T,TT>,T>(head);
}

template <class T, class TT> TIterator<TQueueItem<T,TT>,T> TQueue<T,TT>::end() {
    return TIterator<TQueueItem<T,TT>,T>(nullptr);
}

template <class T, class TT> TQueue<T,TT>::~TQueue() {
    tail = nullptr;
    head = nullptr;
    std::cout << "Очередь удалена" << std::endl; 
}

#endif
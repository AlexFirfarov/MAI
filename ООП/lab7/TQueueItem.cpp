#ifndef TQUEUEITEM_FUNCTION
#include "TQueueItem.h"
#include <iostream>

#else

template <class T, class TT> TQueueItem<T,TT>::TQueueItem() {
    this->next = nullptr;
}

template <class T, class TT> void TQueueItem<T,TT>::SetNext(std::shared_ptr<TQueueItem<T,TT>>& next) {
    this->next = next;
    return;
}

template <class T,class TT> std::shared_ptr<TQueueItem<T,TT>> TQueueItem<T,TT>::GetNext() {
    return this->next;
}

template <class T,class TT> void TQueueItem<T,TT>::SetTree(std::shared_ptr<TBinaryTree<TT>> newTree) {
    this->tree = newTree;
}

template <class T, class TT> std::shared_ptr<TBinaryTree<TT>> TQueueItem<T,TT>::GetTree() {
    return this->tree;
}

template <class T, class TT> int TQueueItem<T,TT>::TGetSize() {
    return tree->GetSize();
}

template <class T, class TT> void TQueueItem<T,TT>::Inorder(IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i) {
    tree->Inorder(criteria, temp, i);
}

template <class T, class TT> void TQueueItem<T,TT>::TPrintTree() {
    if (tree->GetSize() == 0) {
        return;
    }
    tree->PrintTree();
}

template <class T, class TT> void TQueueItem<T,TT>::Delete() {
    tree->~TBinaryTree();
}

template <class A, class AA> std::ostream& operator<<(std::ostream& os, const TQueueItem<A,AA>& obj) {
    os << obj->PrintTree() << std::endl;
    return os;
}

template <class T, class TT> TQueueItem<T,TT>::~TQueueItem() {
}

#endif
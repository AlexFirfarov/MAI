#ifndef TQUEUEITEM_H
#define	TQUEUEITEM_H

#include "Rectangle.h"
#include "Rhomb.h"
#include "Trapeze.h"
#include "TBinaryTree.h"
#include <memory>
#include <cstdlib>

template <class T, class TT> class TQueueItem {
public:
    TQueueItem();
    
    void SetNext(std::shared_ptr<TQueueItem<T,TT>>& next);
    std::shared_ptr<TQueueItem<T,TT>> GetNext();
    std::shared_ptr<TBinaryTree<TT>> GetTree();
    void SetTree(std::shared_ptr<TBinaryTree<TT>> newTree);
    void Inorder(IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i);
    int TGetSize();
    void TPrintTree();
    void Delete();

    template <class A, class AA> friend std::ostream& operator<<(std::ostream& os, const TQueueItem<A,AA>& obj);

    virtual ~TQueueItem();
private:
    std::shared_ptr<TQueueItem<T,TT>> next;
    std::shared_ptr<TBinaryTree<TT>> tree;

};


#define TQUEUEITEM_FUNCTION
#include "TQueueItem.cpp"
#endif
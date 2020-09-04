#ifndef TBINARYTREE_H
#define TBINARYTREE_H
#include "TreeNode.h"
#include <iostream>
#include "IRemoveCriteriaAll.h"
#include "IRemoveCriteriaBySquare.h"

template <class TT> class TBinaryTree {
public:
    TBinaryTree(std::shared_ptr<TT> _figure);
    void Insert(std::shared_ptr<TT> figure);
    void PrintTree();
    int GetSize();
    std::shared_ptr<TreeNode<TT>> GetRoot();
    void Inorder(IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i);
    bool Empty();

   
    ~TBinaryTree();
private:
    void Insert(std::shared_ptr<TreeNode<TT>> node, std::shared_ptr<TT> figure);
    void PrintTree(std::shared_ptr<TreeNode<TT>> node, int tab);
    void Inorder(std::shared_ptr<TreeNode<TT>> node, IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i);

    std::shared_ptr<TreeNode<TT>> root;
    int size;
    
};


#define TREE_FUNCTIONS
#include "TBinaryTree.cpp"

#endif
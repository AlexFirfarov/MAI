#ifndef TBINARYTREE_H
#define TBINARYTREE_H
#include "TreeNode.h"
#include <iostream>


template <class TT> class TBinaryTree {
public:
    TBinaryTree();
    void Insert(std::shared_ptr<TT> command);
    void Inorder();

    ~TBinaryTree();
private:
    void Insert(std::shared_ptr<TreeNode<TT>> node, std::shared_ptr<TT> command);
    void Inorder(std::shared_ptr<TreeNode<TT>> node);

    std::shared_ptr<TreeNode<TT>> root;
    
};


#define TREE_FUNCTIONS
#include "TBinaryTree.cpp"

#endif
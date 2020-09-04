#ifndef TREE_FUNCTIONS
#include "TBinaryTree.h"

#else

template <class TT> TBinaryTree<TT>::TBinaryTree() {
    root.reset();
}

template <class TT> void TBinaryTree<TT>::Insert(std::shared_ptr<TT> command) {
    if (root == nullptr) {
        root.reset(new TreeNode<TT>(command));
        return;
    }
    Insert(root, command);
    return;
}

template <class TT> void TBinaryTree<TT>::Insert(std::shared_ptr<TreeNode<TT>> node, std::shared_ptr<TT> command) {

    if (node->right == nullptr) {
        node->right.reset(new TreeNode<TT>(command));
        return;
    }
    Insert(node->right, command);
    return;
}


template <class TT> void TBinaryTree<TT>::Inorder() {
    Inorder(root);
    return;
}

template <class TT> void TBinaryTree<TT>::Inorder(std::shared_ptr<TreeNode<TT>> node) {
    if (node == nullptr) {
        return;
    }
    Inorder(node->left);
    std::shared_ptr<TT> cmd = node->command;
    (*cmd)();
    Inorder(node->right);
}

template <class TT> TBinaryTree<TT>::~TBinaryTree() {
    root.reset();
}

#endif
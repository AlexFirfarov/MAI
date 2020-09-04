#ifndef TREE_FUNCTIONS
#include "TBinaryTree.h"

#else

template <class TT> TBinaryTree<TT>::TBinaryTree(std::shared_ptr<TT> _figure) {
    root.reset(new TreeNode<TT>(_figure));
    size = 1;
}

template <class TT> void TBinaryTree<TT>::Insert(std::shared_ptr<TT> figure) {
    Insert(root, figure);
    return;
}

template <class TT> void TBinaryTree<TT>::Insert(std::shared_ptr<TreeNode<TT>> node, std::shared_ptr<TT> figure) {

    if (node->figure->Square() < figure->Square() && node->right == nullptr) {
        node->right.reset(new TreeNode<TT>(figure));
        ++size;
        return;
    }
    if (node->figure->Square() >= figure->Square() && node->left == nullptr) {
        node->left.reset(new TreeNode<TT>(figure));
        ++size;
        return;
    }
    if (node->figure->Square() < figure->Square()) {
        Insert(node->right, figure);
    }
    if (node->figure->Square() >= figure->Square()) {
        Insert(node->left, figure);
    }
    return;
}

template <class TT> int TBinaryTree<TT>::GetSize() {
    return size;
}

template <class TT> bool TBinaryTree<TT>::Empty() {
    return size == 0;
}

template <class TT> void TBinaryTree<TT>::PrintTree() {
    puts("---------------------------------------\n");
    PrintTree(root,0);
    return;
}

template <class TT> void TBinaryTree<TT>::PrintTree(std::shared_ptr<TreeNode<TT>> node,int tab) {
    if (node == nullptr) {
        return;
    }
    PrintTree(node->left, tab + 0);
    for (int i = 0; i < tab; ++i){
        putchar(' ');
    }
    node->figure->Print();
    std::cout << std::endl;
    PrintTree(node->right, tab + 0);
}

template <class TT> void TBinaryTree<TT>::Inorder(IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i) {
    Inorder(root, criteria, temp, i);
    return;
}

template <class TT> void TBinaryTree<TT>::Inorder(std::shared_ptr<TreeNode<TT>> node, IRemoveCriteria<TT> * criteria, std::shared_ptr<TT> temp[], int &i) {
    if (node == nullptr) {
        return;
    }
    Inorder(node->left, criteria, temp, i);
    if (!criteria->isIt(node->figure)) {
        temp[i] = node->figure;
        ++i;
    }
    Inorder(node->right, criteria, temp, i);
}

template <class TT> std::shared_ptr<TreeNode<TT>> TBinaryTree<TT>::GetRoot() {
    return root;
}

template <class TT> TBinaryTree<TT>::~TBinaryTree() {
    
}

#endif
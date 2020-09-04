#ifndef TREENODE_H
#define TREENODE_H
#include <cstdlib>
#include <memory>

template <class TT> class TreeNode {
public:
    TreeNode(std::shared_ptr<TT> _figure) : figure(_figure), left(nullptr), right(nullptr) {};

    std::shared_ptr<TreeNode<TT>> left;
    std::shared_ptr<TreeNode<TT>> right;
    std::shared_ptr<TT> figure;

};

#endif
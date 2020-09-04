#ifndef BLOCK_H
#define BLOCK_H
#include <cstdlib>
#include "TreeNode.h"

class TAllocationBlock {
public:
    TAllocationBlock(size_t _size, size_t _count);
    void *allocate();
    void deallocate(void *pointer);
    bool has_free_blocks();

    Tree* Insert(Tree* tree, size_t key);
    Tree* Search(Tree* tree, size_t key);
    Tree* CreateTree(size_t left_border, size_t right_border, Tree* root);
    void DeleteTree(Tree* tree);
     
    virtual ~TAllocationBlock();

private:
    size_t size;
    size_t count;

    char *used_block;
    Tree *root;
    int first;
    int last;

    size_t count_free;
};

#endif
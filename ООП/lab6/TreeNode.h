#ifndef TREENODE_H
#define TREENODE_H
#include <cstdlib>

typedef struct  TFreeBlock {
    void *free_block;
    struct TFreeBlock *left;
    struct TFreeBlock *right;
    size_t key;
} Tree;

#endif
#include "Block.h"
#include <iostream>
#include <math.h>

TAllocationBlock::TAllocationBlock(size_t _size, size_t _count):size(_size), count(_count) {
    used_block = (char*)malloc(size * count);
    root = nullptr;
    root = CreateTree(1,count,root);
    count_free = count;
    first = 0;
    last = 1;
}

void *TAllocationBlock::allocate() {
    void *result = nullptr;
    if (count_free > 0) {
        if (count_free == count) {
            Tree* tree = Search(root,1);
            result = tree->free_block;
            --count_free;
            first = 1;
            last = 2;
            std::cout << "Память выделена" << std::endl;
        }
        else {
            Tree* tree = Search(root,last);
            result = tree->free_block;
            --count_free;
            last = (last % count) + 1;
             std::cout << "Память выделена" << std::endl;
        }
    }
    else {
        std::cout << "Памяти больше нет" << std::endl;
        throw std::bad_alloc();
    }

    return result;
}

void TAllocationBlock::deallocate(void* pointer) {
    Tree* tree = Search(root, first);
    tree->free_block = pointer;
    ++count_free;
    first = (first % count) + 1;
     std::cout << "Память освобождена" << std::endl;
}

bool TAllocationBlock::has_free_blocks() {
    return count_free > 0;
}

TAllocationBlock::~TAllocationBlock() {
    DeleteTree(root);
    free(used_block);
}

Tree* TAllocationBlock::Insert(Tree* tree, size_t key) {
    if(tree == nullptr) {
        Tree* newTree = (Tree*)malloc(sizeof(Tree));
        newTree->key = key;
        newTree->left = nullptr;
        newTree->right = nullptr;
        newTree->free_block = used_block + (key - 1) * size;
        return newTree;
    }
    if(tree->key == key) {
        return tree;
    }
    if(key < tree->key) {
        tree->left = Insert(tree->left, key);
    }else {
        tree->right = Insert(tree->right, key);
    }
    return tree;
}

Tree* TAllocationBlock::CreateTree(size_t left_border, size_t right_border, Tree* tree) {
    if (left_border == right_border) {
        tree = Insert(tree,left_border);
        return tree;
    }
    int average = 0;
    int con = 0;
    for (int i = left_border; i <= right_border; ++i) {
        average += i;
        ++con;
    }
    if (con == 0 ) {
        return tree;
    }
    average = ceil(average / con);
    tree = Insert(tree,average);
    CreateTree(left_border, average - 1, tree);
    CreateTree(average + 1, right_border, tree);
    return tree;
}

Tree* TAllocationBlock::Search(Tree* tree, size_t key) {
    if(tree == nullptr) {
        return nullptr;
    }
    if(tree->key == key) {
        return tree;
    }
    if(key < tree->key) {
        Search(tree->left, key);
    }else {
        Search(tree->right, key);
    }
}

void TAllocationBlock::DeleteTree(Tree* tree) {

    if (tree != nullptr) {
        DeleteTree(tree->left);
        DeleteTree(tree->right);
        free(tree);
    }
}


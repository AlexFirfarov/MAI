#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

const int SIZE_WORD = 256;

struct Node {
    char* word;
    unsigned long long value;
    Node* left;
    Node* right;
    int height;

    Node (char* word, unsigned long long value) {
        int size = strlen(word);
        this->word = new char[size + 1];
        strcpy(this->word, word);
        this->value = value;
        this->height = 1;
        this->left = nullptr;
        this->right = nullptr;
    }
};

class AVL {
public:
    AVL();
    void Insert(char* word, unsigned long long & value);
    void Search(char* word);
    void Delete(char* word);
    void Save(FILE* file);
    void Load(FILE* file);
    void Destroy();

private:
    void Search(char* word, Node* node);
    Node* Insert(char* word, unsigned long long & value, Node* node);
    Node* Delete(char* word, Node* node);
    Node* RotateLeft(Node* node);
    Node* RotateRight(Node* node);
    int Height(Node* node);
    int BalanceFactor(Node* node);
    void ChangeHeight(Node* node);
    Node* UpdateTree(Node* node, int & balance);
    Node* FindMax(Node* node);
    Node* DeleteMax(Node* node);
    void Save(Node* node, FILE* file);
    Node* LoadTree(FILE* file);
    int CompStr(char* str1, char* str2);
    void Destroy(Node* node);

    Node* root;
};
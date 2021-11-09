#ifndef TSUFFTREE_H
#define TSUFFTREE_H

#include <iostream>
#include <algorithm>
#include <map>

struct TNode {
    std::map<char, TNode*> child;
    TNode *suffLink;

    int start;
    int *end;

};

class Stat;

class TSuffixTree {

public:
    TSuffixTree(const std::string& str);
    void DelTSuffixTree();
    friend Stat;

private:
    std::string str;
    int strSize = -1;
    TNode *root = nullptr;
    TNode *needSuffLink = nullptr;

    TNode *curNode = nullptr;
    int curEdge = -1;
    int curLength = 0;

    int countSuff = 0;
    int leafEnd = -1;
    std::vector<int*> split;
    int *rootEnd = nullptr;

    TNode* NewNode(int start, int* end);
    int EdgeLength(TNode *node);
    bool Walk(TNode *cur);
    void ExtendTree(int pos);
    void DelTSuffixTree(TNode *node);

};

class Stat {

public:
    Stat(const std::string& text, int sizePat);
    void FindStat(const std::string& text, TSuffixTree& tree, const std::string& pattern);

private:
    int curLength = 0;
    int curPos = 0;
    std::vector<int> stat;
    TNode *curNode = nullptr;

    int PosStat(const std::string& text, const std::string& pattern, int i);

};

#endif
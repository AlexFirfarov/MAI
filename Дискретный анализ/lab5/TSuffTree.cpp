#include "TSuffTree.h"

TSuffixTree::TSuffixTree(const std::string& str) {
    this->str = str;
    strSize = str.size();
    rootEnd = new int;
    *rootEnd = -1;

    root = NewNode(-1, rootEnd);
    curNode = root;

    for (int i = 0; i < strSize; ++i) {
        ExtendTree(i);
    }

}

TNode* TSuffixTree::NewNode(int start, int* end) {
    TNode *node = new TNode;
    node->suffLink = root;
    node->start = start;
    node->end = end;

    return node;
}

int TSuffixTree::EdgeLength(TNode *node) {
    return *(node->end) - node->start + 1;
}

bool TSuffixTree::Walk(TNode* cur) {
    if (curLength >= EdgeLength(cur)) {
        curEdge += EdgeLength(cur);
        curLength -= EdgeLength(cur);
        curNode = cur;
        return true;
    }
    return false;
}

void TSuffixTree::ExtendTree(int pos) {
    leafEnd = pos;
    ++countSuff;
    needSuffLink = nullptr;

    while (countSuff != 0) {
        if (curLength == 0) {
            curEdge = pos;
        }
        if (curNode->child.find(str[curEdge]) == curNode->child.end()) {
            curNode->child[str[curEdge]] = NewNode(pos, &leafEnd);

            if (needSuffLink != nullptr) {
                needSuffLink->suffLink = curNode;
                needSuffLink = nullptr;
            }
        }
        else {
            TNode *next = curNode->child[str[curEdge]];
            if (Walk(next)) {
                continue;
            }
            if (str[next->start + curLength] == str[pos]) {
                if (needSuffLink != nullptr && curNode != root) {
                    needSuffLink->suffLink = curNode;
                    needSuffLink = nullptr;
                }
                ++curLength;
                break;
            }

            split.push_back(new int);
            *(split.back()) = next->start + curLength - 1;

            TNode *splNode = NewNode(next->start, split.back());
            curNode->child[str[curEdge]] = splNode;

            splNode->child[str[pos]] = NewNode(pos, &leafEnd);
            next->start += curLength;
            splNode->child[str[next->start]] = next;

            if (needSuffLink != nullptr) {
                needSuffLink->suffLink = splNode;
            }
            needSuffLink = splNode;
        }
        --countSuff;
        if (curNode == root && curLength > 0) {
            --curLength;
            curEdge = pos - countSuff + 1;
        }
        else if (curNode != root) {
            curNode = curNode->suffLink;
        }
    }
}

void TSuffixTree::DelTSuffixTree() {
    DelTSuffixTree(root);

    for (int i = 0; i < split.size(); ++i) {
        delete split[i];
    }
    delete rootEnd;

}

void TSuffixTree::DelTSuffixTree(TNode *node) {
    if (node == nullptr) {
        return;
    }

    for (std::map<char, TNode*>::iterator iter = node->child.begin(); iter != node->child.end(); ++iter) {
        DelTSuffixTree(iter->second);
    }

    delete node;
}

Stat::Stat(const std::string& text, int sizePat) {
    stat.resize(text.size() - sizePat + 1);
}

void Stat::FindStat(const std::string& text, TSuffixTree& tree, const std::string& pattern) {
    curNode = tree.root;
    stat[0] = PosStat(text, pattern, 0);

    for (int i = 1; i < stat.size(); ++i) {
        if (curPos == 0) {
            if (curNode != tree.root) {
                curNode = curNode->suffLink;
                stat[i] = PosStat(text, pattern, i);
            }
            else {
                stat[i] = PosStat(text, pattern, i);
            }
        }
        else {
            int ind = curLength - curPos + i - 1;
            if (curNode != tree.root) {
                curNode = curNode->suffLink;
            }
            else {
                ind = i;
                --curPos;
                if (curPos == 0) {
                    stat[i] = PosStat(text, pattern, i);
                    continue;
                }
            }

            for(;;) {
                if (curNode->child.find(text[ind]) == curNode->child.end()) {
                    return ;
                }

                TNode *next = curNode->child[text[ind]];
                int length = tree.EdgeLength(next);
                if (length < curPos) {
                    curNode = next;
                    curPos -= length;
                    ind += length;
                }
                else if (length == curPos) {
                    curPos = 0;
                    curNode = next;
                    break;
                }
                else {
                    break;
                }
            }
            stat[i] = PosStat(text, pattern, i);  
        }
    }
    for (int i = 0; i < stat.size() - 1; ++i) {
        if (stat[i] == pattern.size() - 1) {
            std::cout << i + 1 << '\n';
        }
    }
    if (stat[stat.size() - 1] == pattern.size()) {
        std::cout << stat.size() << '\n';
    }

}

int Stat::PosStat(const std::string& text, const std::string& pattern, int i) {
    int position;
    int j = 0;
    bool fromEdge = true;

    if (curLength == 0) {
        position = i;
    }
    else {
        position = i + curLength - 1;
        --curLength;
    }

    if (position > text.size() - 1)  {
        return curLength;
    } 

    if (curPos > 0) {
        TNode *next = curNode->child[text[position - curPos]];

        int edge = *(next->end) - next->start + 1;
        while ((curPos != edge) && (text[position + j] == pattern[next->start + curPos])) {
            ++curLength;
            ++curPos;
            ++j;
        }
        if (curPos == edge) {
            curPos = 0;
            curNode = next;
            fromEdge = false;
        }
    }
    else {
        fromEdge = false;
    }

    if (fromEdge) {
        return curLength;
    }
    

    while (curNode->child.find(text[position + j]) != curNode->child.end()) {
      
        TNode *next = curNode->child[text[position + j]];
        int edge = *(next->end) - next->start + 1;
    
        while ((curPos != edge) && (text[position + j] == pattern[next->start + curPos])) {
            ++curLength;
            ++curPos;
            ++j;
        }
        if (curPos == edge) {
            curPos = 0;
            curNode = next;
        }
        else {
            break;
        }
    }
    return curLength;   
}

#include "TSuffTree.h"

int main() {
    std::string pattern;
    std::string strt;

    std::cin >> pattern;
    if (pattern.empty()) {
        return 0;
    }

    std::cin >> strt;
    if (strt.empty()) {
        return 0;
    }

    if (pattern.size() > strt.size()) {
        return 0;
    }

    TSuffixTree tree(pattern + '$');
    Stat st(strt + '$', pattern.size() + 1);
    st.FindStat(strt + '$', tree, pattern + '$');

    tree.DelTSuffixTree();

    return 0;
}
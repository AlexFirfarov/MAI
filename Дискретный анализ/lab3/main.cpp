#include "lab2.h"

const int PATH_LENGTH = 255;

int main() {
    
    char symb;
    unsigned long long value = 0;
    AVL tree;

    while (std::cin >> symb) {
        if (symb == '+') {
            char word[SIZE_WORD + 1];
            scanf("%s", word);
            scanf("%llu", &value);
            tree.Insert(word, value); 
        }
        else if (symb == '-') {
            char word[SIZE_WORD + 1];
            scanf("%s", word);
            tree.Delete(word);
        }
        else if (symb == '!') {
            char word[SIZE_WORD + 1];
            scanf("%s", word);
            if (!strcmp(word, "Save")) {
                char path[PATH_LENGTH];
                scanf("%s", path);
                FILE* file = fopen(path, "wb+");

                if (!file) { 
                    printf("ERROR: Couldn't create file\n");
                }
                else {
                    tree.Save(file);
                    fclose(file);
                }
            }
            else {
                char path[PATH_LENGTH];
                scanf("%s", path);
                FILE* file = fopen(path, "rb+");

                if (!file) {
                    printf("ERROR: No such file\n");
                }
                else {
                    tree.Destroy();
                    tree.Load(file);
                    fclose(file);
                }
            }
        }
        else  {
            char word[SIZE_WORD + 1];
            char sym[SIZE_WORD + 1];
            if(std::cin.peek() != '\n') {
                scanf("%s", word);
                sprintf(sym, "%c%s", symb, word);
                tree.Search(sym);
            }
            else {
                sprintf(sym, "%c", symb);
                tree.Search(sym);
            }     
        }
    }

    tree.Destroy();
    
    return 0;
}
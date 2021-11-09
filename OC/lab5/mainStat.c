#include "vector.h"

int main() {
    Vector *vector = (Vector*)malloc(sizeof(Vector));    
    vector->size = 0;
    vector->capacity = 0;

    int choice = 0;
    char string[256];

    puts("---------------------MENU-------------------\n");
    puts("        1 - Просмотреть 1 элемент           \n");
    puts("     2 - Просмотреть последний элемент      \n");
    puts("           3 - Добавить в конец             \n");
    puts("             4 - Извлечь из конца           \n");
    puts("           5 - Просмотреть вектор           \n");
    puts("                  7 - выход                 \n");
    puts("--------------------------------------------\n");

    do {
        printf("\n");
        scanf("%d",&choice);
  
        switch(choice) {
            case 1: {
                printf("%s\n",Front(vector));
                break;
            }
            case 2: {
                printf("%s\n",Back(vector));
                break;
            }
            case 3: {
                scanf("%s",string);
                PushBack(vector,string);
                break;
            }
            case 4: {
                PopBack(vector);
                break;
            }
            case 5: {
                Print(vector);
                break;
            }
            case 7: {
                break;
            }
            default: {
                printf("Нет варианта");
                break;
            }
        }
    } while (choice != 7);
    
    DeleteVector(vector);
    free(vector);
    return 0;


}
#include "vector.h"
#include <stdio.h>
#include <dlfcn.h>

int main() {

    int (*Capacity)(Vector *vector);
    int (*Size)(Vector *vector);
    char* (*Front)(Vector *vector);
    char* (*Back)(Vector *vector);
    void (*PushBack)(Vector *vector, char* string);
    void (*PopBack)(Vector *vector);
    bool (*Empty)(Vector *vector);
    void (*Print)(Vector *vector);
    void (*DeleteVector)(Vector *vector);

    void *lib;
    char* err;
    int check;
    lib = dlopen("libvector.so", RTLD_LAZY);
    if (!lib) {
        printf("Не удалось подключить библиотеку\n");
        exit(EXIT_FAILURE);
    }

    Capacity = dlsym(lib, "Capacity");
    Size = dlsym(lib, "Size");
    Front = dlsym(lib, "Front");
    Back = dlsym(lib, "Back");
    PushBack = dlsym(lib, "PushBack");
    PopBack = dlsym(lib, "PopBack");
    Empty = dlsym(lib, "Empty");
    Print = dlsym(lib, "Print");
    DeleteVector = dlsym(lib, "DeleteVector");

    err = dlerror();
    if (err != NULL) {
        printf("Не удалось подключить библиотеку\n");
        exit(EXIT_FAILURE);
    }

    Vector *vector = (Vector*)malloc(sizeof(Vector));    
    vector->size = 0;
    vector->capacity = 0;

    int choice;
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
                printf("%s\n",(*Front)(vector));
                break;
            }
            case 2: {
                printf("%s\n",(*Back)(vector));
                break;
            }
            case 3: {
                scanf("%s",string);
                (*PushBack)(vector,string);
                break;
            }
            case 4: {
                (*PopBack)(vector);
                break;
            }
            case 5: {
                (*Print)(vector);
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

    (*DeleteVector)(vector);
    free(vector);
    if((check = dlclose(lib)) != 0) {
        printf("Не удалось выгрузить библиотеку\n");
        exit(-1);
    }
    return 0;


}
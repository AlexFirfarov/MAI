#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <ctype.h>
#include <signal.h>
#include <semaphore.h>
#include <string.h>

#define FILE_NAME 256

sem_t *ready;

int main() {
    int countProcess;
    int limit;
    int resultFork;

    printf("Введите кол-во дочерних процессов: \n");
    scanf("%d", &countProcess);

    if (countProcess <= 0) {
        printf("Количество дочерних процесов должно быть больше 0\n");
        exit(-1);
    }

    printf("Введите предел последовательности: \n");
    scanf("%d", &limit);

    if (limit <= 0) {
        printf("Предел последовательности должен быть больше 0\n");
        exit(-1);
    }

    int fn = open("file", O_RDWR | O_CREAT);
    if (fn == -1) {
        printf("Не удалось открыть файл для отображения\n");
        exit(-1);
    }

    int sizeFile = FILE_NAME + limit * sizeof(int) + 1;

    int check_ftruncate = ftruncate(fn,sizeFile);
    if (check_ftruncate == -1) {
        printf("Не удалось установить размер файла");
    }

    char *file_in_memory = mmap(NULL, sizeFile, PROT_READ | PROT_WRITE, MAP_SHARED, fn, 0);
    if (file_in_memory == MAP_FAILED) {
        printf("Не удалось отобразить файл\n");
        exit(-1);
    }

    ready = sem_open("semaphore", O_CREAT, 0777, 1);
    if (ready == SEM_FAILED) {
        printf("Не удалось создать семафор\n");
        exit(-1);
    }

    close(fn);

    for (int i = 0; i < countProcess; ++i) {

        if ((resultFork = fork()) < 0) {
            printf("Не удалось создать дочерний процесс\n");
            exit(-1);
        }
        else if (resultFork > 0) {
            int status;
            char fileName[256];
            int check = 0;
            printf("Введите имя файла: ");
            scanf("%s",fileName);
            if ((check = sprintf(file_in_memory,"%s",fileName)) < 0) {
                printf("Не удалось записать данные\n");
                exit(-1);
            }
            for (int j = 1; j <= limit ; ++j) {
              if ((check = sprintf(file_in_memory + FILE_NAME + sizeof(int) * j, "%d", j)) < 0) {
                printf("Не удалось записать данные\n");
                exit(-1);
              }
            }
            if ((check = sem_post(ready)) == -1) {
                printf("Не удалось разблокировать семафор\n");
                exit(-1);
            }
            if ((check = wait(&status)) == -1) {
                printf("Не удалось дождаться завершения дочернего процесса\n");
                exit(-1);
            }

            if (status != 0) {
                printf("Дочерний процесс завершился с ошибкой\n");
                exit(-1);
            }
        }
        else {
            int check = 0;
            if ((check = sem_wait(ready)) == -1) {
                printf("Не удалось передать управление дочернему процессу\n");
                exit(-1);
            }
            char name[256];
            int shift = 1;
            int num = 0;
            int check_k = 0;
            int check_p = 0;
            if ((check_k = sscanf(file_in_memory , "%s", name)) == -1) {
                printf("Не удалось считать имя файла для записи\n");
                exit(-1);
            }
            FILE* fp = fopen(name,"a");
            if (fp == NULL) {
                printf("Не удалось открыть файл для записи\n");
                exit(-1);
            }
            int check_s = 0;
            while (check_s = sscanf(file_in_memory + FILE_NAME + shift * sizeof(int), "%d", &num)) {
                if (check_s == -1) {
                    printf("Не удалось считать последовательность чисел\n");
                    exit(-1);
                }
                if ((check_p = fprintf(fp, "%d ", num)) < 0) {
                    printf("Не удалось записать в файл\n");
                    exit(-1);
                }
                if (num == limit) {
                    break;
                }
                ++shift;
            }
            if ((check = fclose(fp)) != 0) {
                printf("Не удалось закрыть файл\n");
                exit(-1);
            }
            exit(0);
        }
    }
    if (munmap(file_in_memory, sizeFile)) {
        printf("Не удалось освободить память\n");
        exit(-1);
    }
    int check_sem = 0;
    if ((check_sem = sem_close(ready)) != 0) {
        printf("Не удалось закрыть семафор\n");
        exit(-1);
    }
    return 0;
}
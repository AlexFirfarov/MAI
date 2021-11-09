#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main() {

    int n;
    int m; 
    int pipefd[2];
    int resultFork;

    printf("Кол-во дочерних процессов: \n");
    scanf("%d", &n);

    printf("Введите предел последовательности: \n");
    scanf("%d", &m);


    for (int i = 0; i < n; ++i) {

        if (pipe(pipefd) == -1) {
            printf("Не удалось создать pipe\n");
            exit(-1);
        }

        resultFork = fork();

        if (resultFork < 0) {
           printf("Не удалось создать дочерний процесс\n");
           exit(-1);
        }
        else if (resultFork > 0) {

            char fileName[20];

            close(pipefd[0]);
            printf("Введите имя файла: ");
            scanf("%s", fileName);

            write(pipefd[1], fileName, 20);

            for (int j = 1; j < m + 1; ++j) {
                write(pipefd[1], &j, sizeof(int));
            }
            close(pipefd[1]);
            
        }
        else if (resultFork == 0) {

            char name[20];
            int num = 0;
            close(pipefd[1]);
            read(pipefd[0], name, 20);
            FILE* fp = fopen(name,"a");

            while (read(pipefd[0], &num, sizeof(int))) {
                fprintf(fp, "%d ", num);
            }

            fclose(fp);
            close(pipefd[0]);
            exit(0);
        }
    }

    return 0;
}
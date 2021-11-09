#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include <math.h>

#define SIZE_NUM 32
 
typedef unsigned __int128 int128_t;

int128_t Atonum(char* bufer);
void *ThreadFunction(void *tmpPtr);
void PrintResult(int128_t result);

typedef struct {
    long long countNumbers;
    int128_t localResult;
    __off_t startPos;
    long long localCountNumbers;
} ThreadParam;

long long threadCount;
long long memoryCount;
char filename[256];

int main(int argc, char **argv)
{
    int fn;
    unsigned int localCountNumber;
    unsigned int localCountNumbersLast;

    if (argc != 3) {
        printf("Недопустимое количество параметров\n");
        exit(-1);
    }

    memoryCount = atoi(argv[1]);
    if (memoryCount <= 0) {
        printf("Количество памяти должно быть больше 0\n");
        exit(-1);
    }
    threadCount = atoi(argv[2]);
        if (threadCount <= 0) {
        printf("Количество потоков должно быть больше 0\n");
        exit(-1);
    }

    if ((threadCount * (sizeof(ThreadParam) + sizeof(pthread_t))) > memoryCount) {
        printf("Недостаточно памяти для заданного числа потоков\n");
        exit(-1);
    }

    ThreadParam *threadParam;
    if (!(threadParam = (ThreadParam*)malloc(sizeof(ThreadParam) * threadCount))) {
        printf("Не удалось выделить память\n");
        exit(-1);
    }
    pthread_t *threads;
    if (!(threads = (pthread_t*)malloc(sizeof(pthread_t) * threadCount))) {
        printf("Не удалось выделить память\n");
        exit(-1);
    }
    
    printf("Введите имя файла с числами\n ");
    scanf("%s",filename);

    if ((fn = open(filename, O_RDONLY)) == -1) {
        printf("Не удалось открыть файл\n");
        exit(-1);
    }

    long long size = lseek(fn, 0, SEEK_END);
    close(fn);
    if (size == -1) {
        printf("Не удалось сменить позицию в файле\n");
        exit(-1);
    }
    if (size == 0) {
        printf("Файл пуст\n");
        exit(-1);
    }
    long long numCount = size % (SIZE_NUM + 1);
    if (numCount != 0 ) {
        printf("Неверный формат файла\n");
        exit(-1);
    }
    numCount = size / (SIZE_NUM + 1); 

    if (threadCount > numCount) {
        threadCount = numCount;
    }

    localCountNumbersLast = numCount % threadCount;
    if (localCountNumbersLast != 0) {
        localCountNumber = numCount/ threadCount;
        localCountNumbersLast = numCount - localCountNumber * (threadCount - 1);
    }
    else {
        localCountNumber = numCount / threadCount;
        localCountNumbersLast = localCountNumber;
    }

    threadParam[0].countNumbers = numCount;
    threadParam[0].localResult = 0;
    threadParam[0].startPos = 0;
    threadParam[0].localCountNumbers = localCountNumber;
    for (int i = 1; i < threadCount; ++i) {
        threadParam[i].countNumbers = numCount;
        threadParam[i].localResult = 0;
        threadParam[i].startPos = i * (threadParam[i - 1].localCountNumbers * (SIZE_NUM + 1));
        threadParam[i].localCountNumbers = localCountNumber;
    }
    threadParam[threadCount - 1].localCountNumbers = localCountNumbersLast;

    for (int i = 0; i < threadCount; ++i) {
        int check = 0;
        if ((check = pthread_create(&threads[i], NULL, ThreadFunction, (void *) &threadParam[i])) != 0) {
            printf("Не удалось создать поток\n");
            exit(-1);
        } 
    }

    for (int i = 0; i < threadCount; ++i) {
        int check = 0;
        if ((check = pthread_join(threads[i], NULL)) != 0) {
            printf("Не удалось дождаться завершения работы потока\n");
            exit(-1);
        } 
    }

    int128_t result = 0;
    for (int i = 0; i < threadCount; ++i) {
        result += threadParam[i].localResult;
    }
    PrintResult(result);
    return 0;
}

int128_t Atonum(char* bufer) {
    int128_t num = 0;
    while(*bufer) {
        if (isalpha(*bufer)) {
           num = num * 16 + (*bufer - '7');
        }
        if (isdigit(*bufer)) {
            num = num * 16 + (*bufer - '0');
        }
        ++bufer;
    }
    return num;
}

void *ThreadFunction(void *tmpPtr)
{
    ThreadParam *threadParam = (ThreadParam *)tmpPtr;
    char bufer[SIZE_NUM + 1];
    char ch = 0;
    int fn = 0;
    int check = 0;

    if ((fn = open(filename, O_RDONLY)) == -1) {
        printf("Не удалось открыть файл\n");
        exit(-1);
    }

    if ((check = lseek(fn, threadParam->startPos, SEEK_SET)) == -1) {
        printf("Не удалось сменить позицию в файле\n");
        exit(-1);
    }
    for (int i = 0; i < threadParam->localCountNumbers; ++i)
    {
        if ((check = read(fn,bufer,SIZE_NUM)) == -1) {
            printf("Ошибка чтения\n");
            exit(-1);
        }
        bufer[SIZE_NUM] = '\0';
        int128_t number;
        number = Atonum(bufer);
        number /= threadParam->countNumbers;
        threadParam->localResult += number;

        if ((check = read(fn, &ch, 1)) == -1) {
            printf("Ошибка чтения\n");
            exit(-1);
        }

        if ((ch != '\n') && (ch != '\0')) {
            printf("Неверный формат файла\n");
            exit(-1);
        }
    }
    close(fn);
    return 0; 
}

void PrintResult(int128_t result) {
    double length = 128 * log10(2);
    int len = (int)(length) + 2;
    int i = 0, k = len - 1;

    int num[len];
    
    for (int j = 0; j < len; ++j) {
        num[j] = 0;
    }

    while (result != 0) {
        int temp = result % 10;
        num[i] = temp;
        result = result / 10;
        ++i;
    }

    while (num[k] == 0 && k != 0)  {
        --k;
    }

    for (int j = k; j > 0; --j) {
        printf("%d",num[j]);
    }
    printf("%d\n",num[0]);
}



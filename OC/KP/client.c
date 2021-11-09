#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <assert.h>
#include "zmq.h"

#define ESC "\033"

char name[64];
void *soc;
void *con;

int room = -1;
int time_s = 0;
int cur = -1;
int fieldSize = 0;
int ind = -1;

int flag = 0;

typedef struct coord {
    int coord1;
    int coord2;
    int figure;
} Coord;

typedef struct dataCoord {
    Coord* coords;
    int size;
    int capacity;
} DataCoord;

typedef struct MD {
    char clientName[64];
    char opponentName[64];
    short action;
    int codeResp;
    int coord1;
    int coord2;
    int fieldSize;
    int room;
    short cur;
    short ind;
    short time_s;
    void *client;
} MessageData;

int SendRecv(MessageData* ms);
int FindOpponent(MessageData* ms);
int CheckMove(MessageData* ms, DataCoord* data);
void DrawField(DataCoord *data);
int CheckCoord(DataCoord *data, int coord1, int coord2);
void ExitFunc(int signum);
void FirstMenu();
void Menu();

int main() {

    signal(SIGINT, ExitFunc);
    signal(SIGTSTP, ExitFunc);
    
    int resp = 0;
    int action = 0;

    printf("Введите имя игрока\n");
    scanf("%s", name);

    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REQ);
    zmq_connect(socket, "tcp://localhost:4040");

    soc = socket;
    con = context;

    MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
    ms->action = 1;
    ms->client = socket;
    strcpy(ms->clientName , name);

    resp = SendRecv(ms);
    switch(resp) {
        case 10: {
            printf("Сервер переполнен\n");
            return 0;
        }
        case 1: {
            printf("Вы подключены\n");
            break;
        }
    }
    free(ms);
    FirstMenu();
    resp = 0;

    flag = 1;
    
    do {
        printf("\n");
        scanf("%d", &action);

        MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
        ms->client = socket;
        ms->room = room;
        strcpy(ms->clientName , name);

        switch(action) {
            case 1: {
                ms->action = 2;
                resp = FindOpponent(ms);
                if (resp == 2000) {
                    printf("Истекло время ожидания соперника\n");
                    printf("Вы можете завершить игру или попытаться заново\n");
                }
                else if (resp == 200) {
                    printf("Все игроки заняты\n");
                    printf("Вы можете завершить игру или попытаться заново\n");
                }
                break;
            }
            case 0: {
                printf("Выход...\n");
                ms->action = 0;
                resp = SendRecv(ms);

                free(ms);
                zmq_close(socket);
                zmq_ctx_destroy(context);
                return 0;
            }
            default: {
                printf("Нет вариантов\n");
                break;
            }
        }
        free(ms);
    } while (resp != 2);

    DataCoord data;
    data.size = 0;
    data.capacity = 1;
    data.coords = (Coord*)malloc(sizeof(Coord));

    Menu();
    int check = 0;
    resp = 0;

    flag = 2;

    DrawField(&data);

    do {
        MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
        ms->client = socket;
        ms->room = room;
        ms->cur = cur;
        ms->ind = ind;
        strcpy(ms->clientName , name);

        if (cur == 0) {
            printf("Ход противника...\n");
            ms->action = 3;
            resp = CheckMove(ms, &data);

            if (resp == 300) {
                printf("Вы выйграли. Противник разорвал соединение\n");
                free(data.coords);
                free(ms);
                zmq_close(socket);
                zmq_ctx_destroy(context);
                return 0;
            }

            DrawField(&data);
        }

        if (resp == 30) {
            printf("Вы проиграли!\n");
            free(data.coords);
            free(ms);
            zmq_close(socket);
            zmq_ctx_destroy(context);
            return 0;
        }

        printf("Ваш ход: \n");
        scanf("%d", &action);

        switch(action) {
            case 1: {
                ms->action = 4;
                do {
                    printf("Введите координаты: ");
                    scanf("%d", &(ms->coord1));
                    scanf("%d", &(ms->coord2));
                } while ((ms->coord1 < 0) || (ms->coord2 < 0));

                resp = SendRecv(ms);
                if (resp == 4000) {
                    data.coords[data.size].coord1 = ms->coord1;
                    data.coords[data.size].coord2 = ms->coord2;
                    if (ind == 1) {
                        data.coords[data.size].figure = 1;
                    }
                    else {
                        data.coords[data.size].figure = 2;
                    }
                    ++(data.size);

                    if (data.size == data.capacity) {
                        data.coords = (Coord*)realloc(data.coords, sizeof(Coord) * (data.capacity) * 2);
                        data.capacity *= 2;
                    }

                    int max;
                    if (ms->coord1 >= ms->coord2) {
                        max = ms->coord1;
                    }
                    else {
                        max = ms->coord2;
                    }

                    if (max > fieldSize) {
                        fieldSize = max;
                    }

                    printf(ESC "c");
                    DrawField(&data);

                    printf("Вы победили!\n"); 
                }
                else if (resp == 40) {
                    printf("Клетка уже занята, попробуйте еще раз\n");
                }
                else if (resp == 4) {
                    
                    data.coords[data.size].coord1 = ms->coord1;
                    data.coords[data.size].coord2 = ms->coord2;
                    if (ind == 1) {
                        data.coords[data.size].figure = 1;
                    }
                    else {
                        data.coords[data.size].figure = 2;
                    }
                    ++(data.size);

                    if (data.size == data.capacity) {
                        data.coords = (Coord*)realloc(data.coords, sizeof(Coord) * (data.capacity) * 2);
                        data.capacity *= 2;
                    }

                    cur = 0;

                    int max;
                    if (ms->coord1 >= ms->coord2) {
                        max = ms->coord1;
                    }
                    else {
                        max = ms->coord2;
                    }

                    if (max > fieldSize) {
                        fieldSize = max;
                    }

                    printf(ESC "c");
                    DrawField(&data);
                }
                break;
            }
            case 0: {
                printf("Выход...\n");

                ms->action = 10;
                resp = SendRecv(ms);

                free(data.coords);
                free(ms);
                zmq_close(socket);
                zmq_ctx_destroy(context);
                return 0;
            }
            default: {
                printf("Нет вариантов\n");
                break;
            }
        }
    free(ms);
    } while (resp != 4000);

    free(data.coords);
    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}

int SendRecv(MessageData* ms) {
    zmq_msg_t zmqMessage;
    zmq_msg_init_size(&zmqMessage, sizeof(MessageData));
    memcpy(zmq_msg_data(&zmqMessage), ms, sizeof(MessageData));
    zmq_msg_send(&zmqMessage, ms->client, 0);
    zmq_msg_close(&zmqMessage);

    zmq_msg_init(&zmqMessage);
    zmq_msg_recv(&zmqMessage, ms->client, 0);
    ms = (MessageData*)zmq_msg_data(&zmqMessage);
    zmq_msg_close(&zmqMessage);
    return ms->codeResp;
}

void FirstMenu() {
    puts("---------------------MENU-------------------\n");
    puts("               1 - Найти игру               \n");
    puts("                  0 - Выход                 \n");
    puts("--------------------------------------------\n");
    return;
}

void Menu() {
    puts("---------------------MENU-------------------\n");
    puts("          1 - Ввести координаты поля        \n");
    puts("                  0 - Выход                 \n");
    puts("--------------------------------------------\n");
    return;
}

int FindOpponent(MessageData* ms) {

    do {
        ms->time_s = time_s;
        ms->room = room;
        zmq_msg_t zmqMessage;
        zmq_msg_init_size(&zmqMessage, sizeof(MessageData));
        memcpy(zmq_msg_data(&zmqMessage), ms, sizeof(MessageData));
        zmq_msg_send(&zmqMessage, ms->client, 0);
        zmq_msg_close(&zmqMessage);

        zmq_msg_init(&zmqMessage);
        zmq_msg_recv(&zmqMessage, ms->client, 0);
        ms = (MessageData*)zmq_msg_data(&zmqMessage);
        zmq_msg_close(&zmqMessage);
        int res = ms->codeResp;
        room = ms->room;

        if (res == 20) {
            sleep(5);
            time_s += 5;
        }
        else if (res == 2) {
            printf("Соперник найден: %s\n", ms->opponentName);
            cur = ms->cur;
            fieldSize = ms->fieldSize;
            ind = ms->ind;
            time_s = 0;
            return res;
        }
        else {
            room = -1;
            time_s = 0;
            return res;
        }
    } while(time_s <= 45);

    room = -1;
    ind = -1;
    time_s = 0;

}

int CheckMove(MessageData* ms, DataCoord *data) {
    int coord1;
    int coord2;
    int resp = 0;

    do {
        zmq_msg_t zmqMessage;
        zmq_msg_init_size(&zmqMessage, sizeof(MessageData));
        memcpy(zmq_msg_data(&zmqMessage), ms, sizeof(MessageData));
        zmq_msg_send(&zmqMessage, ms->client, 0);
        zmq_msg_close(&zmqMessage);

        zmq_msg_init(&zmqMessage);
        zmq_msg_recv(&zmqMessage, ms->client, 0);
        ms = (MessageData*)zmq_msg_data(&zmqMessage);
        zmq_msg_close(&zmqMessage);
        cur = ms->cur;

        if (cur == 0) {
            sleep(5);
        }
        else {
            coord1 = ms->coord1;
            coord2 = ms->coord2;
            fieldSize = ms->fieldSize;
            resp = ms->codeResp;

            if (resp == 300) {
                return resp;
            }
        }

    } while(cur != 1);

    data->coords[data->size].coord1 = coord1;
    data->coords[data->size].coord2 = coord2;
    if (ind == 1) {
        data->coords[data->size].figure = 2;
    }
    else {
        data->coords[data->size].figure = 1;
    }
    ++(data->size);

    if (data->size == data->capacity) {
        data->coords = (Coord*)realloc(data->coords, sizeof(Coord) * (data->capacity) * 2);
        data->capacity *= 2;
    }

    printf(ESC "c");
    printf("Противник сделал ход: %d %d\n", coord1, coord2);
    return resp;
}

void DrawField(DataCoord *data) {
    int col = 1;
    int str = 1;
    int figure = 0;

    char x = 'X';
    char o = 'O';
    char p = '-'; 

    printf("\n");
    
    for (int i = 0; i < 4; ++i) {
        printf(" ");
    }
    for (int i = 0; i < fieldSize; ++i) {
        printf("%3d ", i + 1);
    }
    printf("\n");

    while (str <= fieldSize) {
        printf("%3d |", str);
        for (col = 1; col <= fieldSize; ++col) {
            for (int i = 0; i < data->size; ++i) {
                if ((data->coords[i].coord1 == col) && (data->coords[i].coord2 == str)) {
                    figure = data->coords[i].figure;
                    break;
                }
            }
            if (figure == 1) {
                printf("%2c |", x);
                figure = 0;
            }
            else if (figure == 2) {
                printf("%2c |", o);
                figure = 0;
            }
            else {
                printf("%2c |", p);
            }
        }
        ++str;
        col = 1;
        printf("\n");
    }

    printf("\n");
    return;
}

void ExitFunc(int signum) {

    if (flag == 0) {
        raise(SIGKILL);
    }
    else if (flag == 1) {
        int resp;
        MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
        ms->action = 0;
        ms->client = soc;
        strcpy(ms->clientName, name);
        resp = SendRecv(ms);
        free(ms);
        zmq_close(soc);
        zmq_ctx_destroy(con);
        raise(SIGINT);  
    }
    else {
        int resp;
        MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
        ms->action = 10;
        ms->client = soc;
        ms->ind = ind;
        ms->room = room;
        resp = SendRecv(ms);
        free(ms);
        zmq_close(soc);
        zmq_ctx_destroy(con);
        raise(SIGINT);
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "zmq.h"

int players = 0;

typedef struct coord {
    int coord1;
    int coord2;
    int figure;
} Coord;

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

typedef struct person {
    char name[64];
} Person;

typedef struct room {
    Person per1;
    Person per2;
    int cur;
    int fieldSize;
    int count;
    Coord* coords;
    int size;
    int capacity;
    int win;
} Room;

typedef struct listItem {
    Person player;
    struct listItem* next;
}ListItem;

typedef struct List {
    ListItem *head;
    ListItem *tail;
    int size;
}List;

typedef struct rooms {
    Room *room;
    int size;
    int firstFree;
} Rooms;

void Add(List *list, char* clientName);
void Delete(List *list, char* clientName);

int CheckWin(Room *room, int ind);

int main() {
    int count = 0;

    do {
        printf("Введите максимальное количество игроков больше 0\n");
        scanf("%d", &count);
    } while (count <= 0);

    Rooms rooms;
    rooms.room = (Room*)calloc(count / 2, sizeof(Room));
    rooms.size = 0;
    rooms.firstFree = 0;

    List* data = (List*)malloc(sizeof(List));
    data->head = NULL;
    data->tail = NULL;
    data->size = 0;

    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);
    zmq_bind(socket, "tcp://*:4040");

    while(1) {
        zmq_msg_t message;
        zmq_msg_init(&message);
        zmq_msg_recv(&message, socket, 0);
        MessageData *md = (MessageData*)malloc(sizeof(MessageData));
        memcpy(md, zmq_msg_data(&message), sizeof(MessageData));
        zmq_msg_close(&message);

        switch(md->action) {
            case 1: {
                if (players == count) {
                    zmq_msg_init_size(&message, sizeof(MessageData));
                    md->codeResp = 10;
                    memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                    zmq_msg_send(&message, socket, 0);
                    zmq_msg_close(&message);
                }
                else {
                    Add(data, md->clientName);
                    ++players;
                    zmq_msg_init_size(&message, sizeof(MessageData));
                    md->codeResp = 1;
                    memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                    zmq_msg_send(&message, socket, 0);
                    zmq_msg_close(&message);
                }
                break;  
            }
            case 2: {
                if (md->room == -1) {
                    if (rooms.size == count / 2) {
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        md->codeResp = 200;
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                    }
                    else {
                        int pos = rooms.firstFree;
                        if (rooms.room[pos].count == 0) {
                            strcpy(rooms.room[pos].per1.name, md->clientName);
                            ++rooms.room[pos].count;
                            Delete(data, md->clientName);
                            zmq_msg_init_size(&message, sizeof(MessageData));
                            md->codeResp = 20;
                            md->room = rooms.firstFree;
                            memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                            zmq_msg_send(&message, socket, 0);
                            zmq_msg_close(&message);
                        }
                        else if (rooms.room[pos].count == 1) {
                            strcpy(rooms.room[pos].per2.name, md->clientName);
                            ++(rooms.room[pos].count);
                            Delete(data, md->clientName);
                            rooms.room[pos].cur = 1;
                            rooms.room[pos].fieldSize = 19;
                            ++(rooms.size);

                            zmq_msg_init_size(&message, sizeof(MessageData));
                            md->codeResp = 2;
                            md->room = rooms.firstFree;
                            md->cur = 0;
                            md->fieldSize = 19;
                            md->ind = 2;

                            for (int i = 0; i < count / 2; ++i) {
                                if (rooms.room[i].count != 2) {
                                    rooms.firstFree = i;
                                    break;
                                }
                            }

                            strcpy(md->opponentName, rooms.room[md->room].per1.name);
                            memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                            zmq_msg_send(&message, socket, 0);
                            zmq_msg_close(&message);
                        }
                    }
                }
                else if (md->room != -1) {
                    if (rooms.room[md->room].count == 1) {
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        if (md->time_s == 45) {
                            md->codeResp = 2000;
                            --rooms.room[md->room].count;
                            --rooms.size;

                            for (int i = 0; i < count / 2; ++i) {
                                if (rooms.room[i].count != 2) {
                                    rooms.firstFree = i;
                                    break;
                                }
                            }

                            Add(data, md->clientName);
                        }
                        else {
                            md->codeResp = 20;
                        }
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                    }
                    else if (rooms.room[md->room].count == 2) {
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        md->codeResp = 2;
                        md->cur = 1;
                        md->fieldSize = 19;
                        md->ind = 1;

                        rooms.room[md->room].size = 0;
                        rooms.room[md->room].capacity = 1;
                        rooms.room[md->room].coords = (Coord*)malloc(sizeof(Coord));
                        rooms.room[md->room].win = 0;

                        strcpy(md->opponentName, rooms.room[md->room].per2.name);
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                    }
                }
                break;
            }
            case 3: {
                if (md->ind != rooms.room[md->room].cur) {
                    zmq_msg_init_size(&message, sizeof(MessageData));
                    md->cur = 0;
                    memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                    zmq_msg_send(&message, socket, 0);
                    zmq_msg_close(&message);
                }
                else {
                    
                    zmq_msg_init_size(&message, sizeof(MessageData));
                    Room *room_t = &(rooms.room[md->room]);

                    md->cur = 1;
                    if (room_t->size != 0) {
                        md->coord1 = room_t->coords[room_t->size - 1].coord1;
                        md->coord2 = room_t->coords[room_t->size - 1].coord2;
                        md->fieldSize = room_t->fieldSize;
                    }
                    
                    if (room_t->win != 0) {
                        if (room_t->win != md->ind) {
                            md->codeResp = 30;
                            players = players - 2;
                            free(room_t->coords);
                            room_t->count = 0;
                            room_t->win = 0;
                            --(rooms.size);

                            for (int i = 0; i < count / 2; ++i) {
                                if (rooms.room[i].count != 2) {
                                    rooms.firstFree = i;
                                    break;
                                }
                            }
                        }
                        else {
                            md->codeResp = 300;
                            players = players - 2;
                            free(room_t->coords);
                            room_t->count = 0;
                            room_t->win = 0;
                            --(rooms.size);

                            for (int i = 0; i < count / 2; ++i) {
                                if (rooms.room[i].count != 2) {
                                    rooms.firstFree = i;
                                    break;
                                }
                            }
                        }
                    }
                    else {
                        md->codeResp = 3;
                    }
                    memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                    zmq_msg_send(&message, socket, 0);
                    zmq_msg_close(&message);
                }
                break;
            }
            case 4: {
                int check = 1;
                Room *room_t = &(rooms.room[md->room]);
                for (int i = 0; i < room_t->size; ++i) {
                    if ((md->coord1 == room_t->coords[i].coord1) && (md->coord2 == room_t->coords[i].coord2)) {
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        md->codeResp = 40;
                        check = 0;
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                        break;
                    }
                }
                if (check != 0) {
                    room_t->coords[room_t->size].coord1 = md->coord1;
                    room_t->coords[room_t->size].coord2 = md->coord2;
                    room_t->coords[room_t->size].figure = md->ind;
                    room_t->size = room_t->size + 1;

                    if (room_t->size == room_t->capacity) {
                        room_t->coords = (Coord*)realloc(room_t->coords, sizeof(Coord) * room_t->capacity * 2);
                        room_t->capacity *= 2;
                    }

                    int max;
                    if (md->coord1 >= md->coord2) {
                        max = md->coord1;
                    }
                    else {
                        max = md->coord2;
                    }

                    if (max > room_t->fieldSize) {
                        room_t->fieldSize = max;
                    }

                    if (CheckWin(room_t, md->ind)) {
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        md->codeResp = 4000;
                        room_t->win = md->ind;
                        room_t->cur = 3 - (md->ind);
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                    }
                    else {
                        room_t->cur = 3 - md->ind;
                        zmq_msg_init_size(&message, sizeof(MessageData));
                        md->fieldSize = room_t->fieldSize;
                        md->codeResp = 4;
                        memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                        zmq_msg_send(&message, socket, 0);
                        zmq_msg_close(&message);
                    }
                }
            break;
            }
            case 0: {
                Delete(data, md->clientName);
                --players;

                zmq_msg_init_size(&message, sizeof(MessageData));
                md->codeResp = 0;
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);

                break;
            }
            case 10: {
                rooms.room[md->room].cur = 3 - md->ind;
                rooms.room[md->room].win = 3 - md->ind;

                zmq_msg_init_size(&message, sizeof(MessageData));
                md->codeResp = 10;
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);

                break;
            }
        }
        free(md);
    }

    return 0;
}

void Add(List *list, char* clientName) {
    ListItem *temp = (ListItem*)malloc(sizeof(ListItem));
    strcpy(temp->player.name, clientName);
    temp->next = NULL;

    if (list->size == 0) {
        list->head = temp;
        list->tail = temp;
    }
    else {
        list->tail->next = temp;
        list->tail = temp;
    }
    ++(list->size); 
}

void Delete(List *list, char* clientName) {

    ListItem *temp = list->head;
    int check;

    if ((check = strcmp(list->head->player.name, clientName)) == 0) {
        list->head = list->head->next;
        free(temp);
        --(list->size);
        return;
    }
    else {
        while ((check = strcmp(temp->next->player.name, clientName)) != 0) {
            temp = temp->next;
        }
        ListItem *t = temp->next;
        temp->next = temp->next->next;

        if (t == list->tail) {
            list->tail = temp;
        }
        
        free(t);
        --(list->size);
        return;
    }
}

int CheckWin(Room *data, int ind) {
    for (int i = 0; i < data->size; ++i) {
        if (data->coords[i].figure == ind) {
            Coord temp = data->coords[i];

            int col1, col2;
            int str1, str2;
            int dm1 = 0, dm2 = 0;
            int dp1 = 0, dp2 = 0;

            if ((temp.coord2 + 4) > data->fieldSize) {
                col1 = -1;
                dm1 = -1;
                dp2 = -1;
            }
            else {
                col1 = 0;
            }

            if ((temp.coord2 - 4) < 1) {
                col2 = -1;
                dm2 = -1;
                dp1 = -1;
            }
            else {
                col2 = 0;
            }

            if ((temp.coord1 + 4) > data->fieldSize) {
                str1 = -1;
                dm1 = -1;
                dp1 = -1;
            }
            else {
                str1 = 0;
            }

            if ((temp.coord1 - 4) < 1) {
                str2 = -1;
                dm2 = -1;
                dp2 = -1;
            }
            else {
                str2 = 0;
            }

            for (int j = 0; j < data->size; ++j) {
                Coord tmp = data->coords[j];
                if (col1 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord2 == (temp.coord2 + k)) && (tmp.coord1 == temp.coord1) && (tmp.figure == temp.figure)) {
                            ++col1;
                            break;
                        }
                    }
                }
                if (col2 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord2 == (temp.coord2 - k)) && (tmp.coord1 == temp.coord1) && (tmp.figure == temp.figure)) {
                            ++col2;
                            break;
                        }
                    }
                }
                if (str1 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 + k)) && (tmp.coord2 == temp.coord2) && (tmp.figure == temp.figure)) {
                            ++str1;
                            break;
                        }
                    }
                }
                if (str2 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 - k)) && (tmp.coord2 == temp.coord2) && (tmp.figure == temp.figure)) {
                            ++str2;
                            break;
                        }
                    }
                }
                if (dm1 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 + k)) && (tmp.coord2 == (temp.coord2 + k)) && (tmp.figure == temp.figure)) {
                            ++dm1;
                            break;
                        }
                    }
                }
                if (dm2 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 - k)) && (tmp.coord2 == (temp.coord2 - k)) && (tmp.figure == temp.figure)) {
                            ++dm2;
                            break;
                        }
                    }
                }
                if (dp1 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 + k)) && (tmp.coord2 == (temp.coord2 - k)) && (tmp.figure == temp.figure)) {
                            ++dp1;
                            break;
                        }
                    }
                }
                if (dp2 != -1) {
                    for (int k = 1; k <= 4; ++k) {
                        if ((tmp.coord1 == (temp.coord1 - k)) && (tmp.coord2 == (temp.coord2 + k)) && (tmp.figure == temp.figure)) {
                            ++dp2;
                            break;
                        }
                    }
                }

                if ((col1 == 4) || (col2 == 4) || (str1 == 4) || (str2 == 4) || (dm1 == 4) || (dm2 == 4) || (dp1 == 4) || (dp2 == 4)) {
                    return 1;
                }
            }
            
        }
    }
    return 0;
}

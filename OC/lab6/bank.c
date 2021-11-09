#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "zmq.h"

int limit = 10000;

typedef struct MD {
    char clientId[400];
    int action;
    int sum;
    int credit;
    char anotherClientId[400];
    char recvMessage[256];
    void *client;
} MessageData;

typedef struct _client {
    char clientId[400];
    int sum;
    int credit;
    struct _client *next;
} Client;

typedef struct client_list {
    Client* head;
    Client* tail;
    int size;
} ClientList;

void Add(ClientList *list, char* clientId);
Client* Find(ClientList *list, char* clientId);
void Destroy(ClientList *list);

char* ClientIncreaseDeb(ClientList *list, char* cl, int sum);
char* ClientIncreaseCred(ClientList *list, char* cl, int sum);
char* ClientDecreaseDeb(ClientList *list, char* cl, int sum);
int ClientDecreaseCred(Client* cur, int sum);
char* SendMoney(ClientList *list, char* anotherClientId, char* client, int sum);
int CheckDeb(ClientList *list, char* cl);
int CheckCred(ClientList *list, char* cl);

void SendRecv(MessageData* ms);

int main() {
    ClientList* data = (ClientList*)malloc(sizeof(ClientList));
    data->head = NULL;
    data->tail = NULL;
    data->size = 0;
    int action = 0;

    void *context = zmq_ctx_new();
    void *socket = zmq_socket(context, ZMQ_REP);

    int bank;
    char adress[256];
    printf("Введите адрес банка\n");
    scanf("%d",&bank);
    sprintf(adress, "%s%d", "tcp://*:", bank);
    zmq_bind(socket, adress);
    

    while(1) {
        char info[256];

        zmq_msg_t message;
        zmq_msg_init(&message);
        zmq_msg_recv(&message, socket, 0);
        MessageData *md = (MessageData*)zmq_msg_data(&message);
        zmq_msg_close(&message);

        switch (md->action) {
            case 1: {
                strncpy(info, ClientIncreaseDeb(data, md->clientId, md->sum), 256);
                memcpy(md->recvMessage, info, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }
            case 2: {
                strncpy(info, ClientIncreaseCred(data, md->clientId, md->sum), 256);
                memcpy(md->recvMessage, info, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }  
            case 3: {
                strncpy(info, SendMoney(data,md->anotherClientId ,md->clientId, md->sum), 256);
                memcpy(md->recvMessage, info, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }  
            case 4: {
                int info;
                char inf_m[256];
                if ((info = CheckDeb(data, md->clientId)) == -1) {
                    strcpy(inf_m, "Вы не клиент банка\n");
                }
                else {
                    sprintf(inf_m, "%s%d", "Дебетовый счет: ", info);
                }
                memcpy(md->recvMessage, inf_m, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }
            case 5: {
                int info;
                char inf_m[256];
                if ((info = CheckCred(data, md->clientId)) == -1) {
                    strcpy(inf_m, "Вы не клиент банка\n");
                }
                else {
                    sprintf(inf_m, "%s%d", "Задолженность: ", info);
                }
                memcpy(md->recvMessage, inf_m, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }
            case 6: {
                strncpy(info, ClientDecreaseDeb(data, md->clientId, md->sum), 256);
                memcpy(md->recvMessage, info, 256);
                zmq_msg_init_size(&message, sizeof(MessageData));
                memcpy(zmq_msg_data(&message), md, sizeof(MessageData));
                zmq_msg_send(&message, socket, 0);
                zmq_msg_close(&message);
                break;
            }
        }
    }

}

void Add(ClientList *list, char* clientId) {
    Client *cl = (Client*)malloc(sizeof(Client));
    strncpy(cl->clientId, clientId, 400);
    cl->sum = 0;
    cl->credit = 0;
    cl->next = NULL;

    if (list->size == 0) {
        list->head = cl;
        list->tail = cl;
    }
    else {
        list->tail->next = cl;
        list->tail = cl;
    }
    ++(list->size); 
}

Client* Find(ClientList *list, char* clientId) {
    Client *cur = list->head;
    int check = 0;
    while(cur != NULL) {
        if ((check = strcmp(cur->clientId, clientId)) == 0) {
            return cur;
        }
        cur = cur->next;
    }
    return NULL;
}

char* ClientIncreaseDeb(ClientList *list, char* cl, int sum) {
    int add = 0;
    Client *cur = Find(list, cl);
    if (cur == NULL) {
        Add(list, cl);
        add = 1;
    }
    cur = Find(list, cl);
    cur->sum = cur->sum + sum;
    if (add == 1) {
        return "Клиент добавлен. Деньги зачислены на дебетовый счет\n";
    }
    else {
        return "Деньги зачислены на дебетовый счет\n";
    }
}

char* ClientIncreaseCred(ClientList *list, char* cl, int sum) {
    Client *cur = Find(list, cl);
    if (cur == NULL) {
        return "Не является клиентом банка\n";
    }
    if (cur->credit == 0) {
        return "Нет кредита для погашения\n";
    }
    if (sum > cur->credit) {
        return "Сумма больше текущего кредита\n";
    }
    cur->credit = cur->credit - sum;
    if (cur->credit == 0) {
        return "Кредит погашен\n";
    }
    return "Часть кредита погашена\n";
}

char* SendMoney(ClientList *list, char* anotherClientId, char* client, int sum) {
    char info[256];
    Client *client_an = Find(list, anotherClientId);
    if (client_an == NULL) {
        return "Получатель не является клиентом банка\n";
    }
    Client *cur = Find(list, client);
    if (cur == NULL) {
        return "Отправитель не является клиентом банка\n";
    }
    if (cur->sum < sum) {
        return "Недостаточно средств для перевода\n";
    }
    cur->sum = cur->sum - sum;
    client_an->sum = client_an->sum + sum;
    return "Деньги переведены\n";
}

int CheckDeb(ClientList *list, char* cl) {
    Client *cur = Find(list, cl);
    if (cur == NULL) {
        return -1;
    }
    return cur->sum;
}

int CheckCred(ClientList *list, char* cl) {
    Client *cur = Find(list, cl);
    if (cur == NULL) {
        return -1;
    }
    return cur->credit;
}

char* ClientDecreaseDeb(ClientList *list, char* cl, int sum) {
    Client *cur = Find(list, cl);
    if (cur == NULL) {
        return "Вы не являетесь клиентом банка\n";
    }
    if (cur->sum < sum) {
        int check = 0;
        int credit = sum - cur->sum;
        cur->sum = 0;
        check = ClientDecreaseCred(cur, credit);
        if (check == -1) {
            return "Превышен лимит\n";
        }
        if (check == 0) {
            return "Деньги сняты. Был взят кредит\n";
        } 
    }
    else {
        cur->sum = cur->sum - sum;
        return "Деньги сняты\n";
    }
}

int ClientDecreaseCred(Client* cur, int sum) {
    if ((cur->credit + sum) > limit) {
        return -1;
    }
    cur->credit = cur->credit + sum;
    return 0;
}
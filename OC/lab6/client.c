#include <time.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "zmq.h"

typedef struct MD {
    char clientId[400];
    int action;
    int sum;
    int credit;
    char anotherClientId[400];
    char recvMessage[256];
    void *client;
} MessageData;

int SendRecv(MessageData* ms) {
    int check = 0;
    zmq_msg_t zmqMessage;
    zmq_msg_init_size(&zmqMessage, sizeof(MessageData));
    memcpy(zmq_msg_data(&zmqMessage), ms, sizeof(MessageData));
    zmq_msg_send(&zmqMessage, ms->client, 0);
    zmq_msg_close(&zmqMessage);

    zmq_msg_init(&zmqMessage);
    zmq_msg_recv(&zmqMessage, ms->client, 0);
    ms = (MessageData*)zmq_msg_data(&zmqMessage);
    if ((check = strlen(ms->recvMessage)) < 2) {
        zmq_msg_close(&zmqMessage);
        return -1;
    }
    printf("%s\n", ms->recvMessage);
    zmq_msg_close(&zmqMessage);
    return 0;
}

int main() {
    char clientId[400];
    int bankId;
    char bankAdress[256];

    void *context = zmq_ctx_new();
    printf("Введите логин клиента\n");
    scanf("%s",clientId);
    printf("Введите адрес банка\n");
    scanf("%d",&bankId);

    sprintf(bankAdress, "%s%d", "tcp://localhost:", bankId);

    void *socket = zmq_socket(context, ZMQ_REQ);
    zmq_connect(socket, bankAdress);

    int action = 0;

    puts("---------------------MENU-------------------\n");
    puts("        1 - Положить деньги на счет         \n");
    puts("             2 - Погасить кредит            \n");
    puts("        3 - Перевести другому клиенту       \n");
    puts("          4 - Запросить дебетовый счет      \n");
    puts("     5 - Запросить задолженность по кредиту \n");
    puts("               6 - Снять деньги             \n");
    puts("                  7 - Выход                 \n");
    puts("--------------------------------------------\n");

    do {
        printf("\n");
        scanf("%d",&action);

        MessageData* ms = (MessageData*)malloc(sizeof(MessageData));
        ms->action = action;
        ms->client = socket;
        strncpy(ms->clientId , clientId, 400);

        switch(action) {
            case 1: {
                int check = 0;
                printf("Введите сумму: ");
                scanf("%d", &(ms->sum));
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 2: {
                int check = 0;
                printf("Введите сумму: ");
                scanf("%d", &(ms->sum));
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 3: {
                int check = 0;
                printf("Введите сумму: ");
                scanf("%d", &(ms->sum));
                printf("Введите ID клиента: ");
                scanf("%s", (ms->anotherClientId));
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 4: {
                int check = 0;
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 5: {
                int check = 0;
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 6: {
                int check = 0;
                printf("Введите сумму: ");
                scanf("%d", &(ms->sum));
                if ((check = SendRecv(ms)) == -1) {
                    SendRecv(ms);
                }
                break;
            }
            case 7: {
                break;
            }
            default: {
                printf("Нет вариантов\n");
            }
        }
    } while(action != 7);

    zmq_close(socket);
    zmq_ctx_destroy(context);

    return 0;
}
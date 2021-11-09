#ifndef BANK_H
#define BANK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "zmq.h"

typedef struct _client {
    char clientId[1000];
    int sum;
    int credit;
    Client* next;
} Client;

typedef struct client_list {
    Client* head;
    Client* tail;
} ClientList;

ClientList data;

ClientList Init();
void Add(char* clientId);
Client Find(char* clientId);
void Destroy();

void ClientIncreaseDeb(char* cl, int sum);
void ClientIncreaseCred(char* cl, int sum);
void ClientDecreaseDeb(char* cl, int sum);
void ClientDecreaseCred(char* cl, int sum);
int SendMoney(char* anotherClientId, char* client, int sum);
int CheckDeb(char* cl);
int CheckCred(char* cl);


#endif
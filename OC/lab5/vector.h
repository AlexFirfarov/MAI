#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    char string[256];
}Element;

typedef struct {
    Element *data;
    int capacity;
    int size;
}Vector; 

extern int Capacity(Vector *vector);
extern int Size(Vector *vector);
extern char* Front(Vector *vector);
extern char* Back(Vector *vector);
extern void PushBack(Vector *vector, char* string);
extern void PopBack(Vector *vector);
extern bool Empty(Vector *vector);
extern void Print(Vector *vector);
extern void DeleteVector(Vector *vector);

#endif
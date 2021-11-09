#include "vector.h"

int Capacity(Vector *vector) {
    return vector->capacity;
}

int Size(Vector *vector) {
    return vector->size;
}

char* Front(Vector *vector) {
    if (Empty(vector)) {
        return "Вектор пуст";
    }
    else {
        return vector->data[0].string;
    }
}

char* Back(Vector *vector) {
    if (Empty(vector)) {
        return "Вектор пуст";
    }
    else {
        return vector->data[vector->size-1].string;
    }
}

void PushBack(Vector *vector, char* string) {
    if (vector->capacity <= vector->size) {
        vector->data = (Element*)realloc(vector->data, sizeof(Element) * vector->capacity * 2 + 1 * sizeof(Element));
        vector->capacity = vector->capacity * 2 + 1;
    }
    strncpy(vector->data[vector->size].string, string, 256);
    ++vector->size;
    return;
}

void PopBack(Vector *vector) {
    if (Empty(vector)) {
        printf("Вектор пуст\n");
        return;
    }
    --vector->size;
}

bool Empty(Vector *vector) {
    if (vector->size == 0) {
        return true;
    }
    return false;
}

void Print(Vector *vector) {
    if (Empty(vector)) {
        printf("Вектор пуст\n");
        return;
    }
    int pos = 0;
    do {
        printf("%s\n",vector->data[pos].string);
        ++pos;
    } while(pos != vector->size);
    return;
}

void DeleteVector(Vector *vector) {
    free(vector->data);
}
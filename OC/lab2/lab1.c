#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    fork();
    fork();
    fork();

    return 0;
}
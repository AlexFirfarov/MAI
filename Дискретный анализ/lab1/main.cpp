#include <cstdlib>
#include <iostream>
#include <cmath>

const int SIZE_KEY = 3;
const int MAX_BYTE = 256;
const int BYTE = 8;

struct TData {
    int key[SIZE_KEY];
    unsigned long long value;
};

void Sort(TData data[], int i, int count, int temp[], TData result[]);

int main(int argc, char* argv[]) {

    //TData data[50];
    TData *data = (TData*)malloc(sizeof(TData));
    std::istream &is = std::cin;
    int count = 0;
    int size_array = 1;

    
    while (!is.eof()) {

        is >> data[count].key[0];
        is >> data[count].key[1];
        is >> data[count].key[2];
        is >> data[count].value;

        data[count].key[1] *= (-1);
        data[count].key[2] *= (-1);

        count++;

        if (count == size_array) {
            data = (TData*)realloc(data,sizeof(TData) * size_array * 2);
            size_array *= 2;
        } 
    } 

    //delete is;

    if (count < size_array) {
        data = (TData*)realloc(data,sizeof(TData) * count);
        size_array = count;
    } 

    int *temp = new int[MAX_BYTE](); 
    TData *result = new TData[count]();

    for (int i = SIZE_KEY - 1; i >= 0; i--) {
        Sort(data,i,count,temp,result);
    } 

    delete [] temp;
    delete [] result;

    for (int i = 0; i < count; i++) {
        std::cout << data[i].key[0] << '-' << data[i].key[1] << '-' << data[i].key[2] << '\t';
        std::cout << data[i].value << std::endl;
    }

    free(data);
    data = nullptr;

    return 0;
}

void Sort(TData data[], int i, int count, int temp[], TData result[]) {
    int digit = sizeof(int);

    for (int j = 0; j < digit; j++) {

        for (int k = 0; k < count; k++) {
            temp[((data[k].key[i] >> (j * BYTE)) & (MAX_BYTE - 1))]++;
        }
        for (int k = 1; k < MAX_BYTE; k++) {
            temp[k] += temp[k-1];
        } 
        for (int k = count - 1; k >= 0; k--) {
            result[temp[((data[k].key[i] >> (j * BYTE)) & (MAX_BYTE - 1))] - 1] = data[k];
            temp[((data[k].key[i] >> (j * BYTE)) & (MAX_BYTE - 1))]--;
        } 
        for (int k = 0; k < count; k++) {
            data[k] = result[k];
        }
        for (int k = 0; k < MAX_BYTE; k++) {
            temp[k] = 0;
        }

    } 

} 


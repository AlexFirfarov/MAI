#include <cstdlib>
#include <iostream>
#include "Rectangle.h"
#include "TQueueItem.h"
#include "TQueue.h"

int main(int argc, char** argv) {

    char choice;
    TQueue queue;

    puts("---------------------MENU-------------------\n");
    puts("            1 - Добавить фигуру             \n");
    puts("             2 - Извлечь фигуру             \n");
    puts("           3 - Просмотреть очередь          \n");
    puts("             4 - Удалить очередь            \n");
    puts("                  x - выход                 \n");
    puts("--------------------------------------------\n");

    do {
        std::cout << '\n';
        std::cin >> choice;
  
        switch(choice) {
            case '1': {
                Rectangle t;
                std::cin >> t;
                queue.push(std::move(t));
                std::cout << "Добавлено " << std::endl;
                break;
            }
            case '2': {
                Rectangle t;
                t = queue.pop();
                std::cout << t;
                break;
            }
            case '3': {
                std::cout << queue;
                break;
            }
             case '4': {
                queue.~TQueue();
                break;
            }
              case 'x':{
                break;
            }
            default:{
                std::cout << "Введите 1,2,3,4 или x" << std::endl;
                break;
            }
        }
    } while (choice != 'x'); 

    return 0;
}

#include <cstdlib>
#include <iostream>
#include "Figure.h"
#include "TQueueItem.h"
#include "TQueue.h"
#include <memory>

int main(int argc, char** argv) {

    char choice;
    TQueue queue;

    puts("---------------------MENU-------------------\n");
    puts("        r - Добавить прямоугольник          \n");
    puts("             b - Добавить ромб              \n");
    puts("           t - Добавить трапецию            \n");
    puts("             2 - Извлечь фигуру             \n");
    puts("           3 - Просмотреть очередь          \n");
    puts("             4 - Удалить очередь            \n");
    puts("                  x - выход                 \n");
    puts("--------------------------------------------\n");

    do {
        std::cout << '\n';
        std::cin >> choice;
  
        switch(choice) {
            case 'r': {
                Figure *t = new Rectangle(std::cin);
                queue.push(std::shared_ptr<Figure>(t));
                std::cout << "Добавлен прямоугольник " << std::endl;
                break;
            }
            case 'b': {
                Figure *t = new Rhomb(std::cin);
                queue.push(std::shared_ptr<Figure>(t));
                std::cout << "Добавлен ромб " << std::endl;
                break;
            }
            case 't': {
                Figure *t = new Trapeze(std::cin);
                queue.push(std::shared_ptr<Figure>(t));
                std::cout << "Добавлена трапеция " << std::endl;
                break;
            }
            case '2': {
                std::shared_ptr<Figure> t;
                t = queue.pop();
                if (t == nullptr) {
                    break;
                }
                t->Print();
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
                std::cout << "Нет варианта" << std::endl;
                break;
            }
        }
    } while (choice != 'x');

    return 0;
}


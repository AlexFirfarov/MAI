#include <cstdlib>
#include <iostream>
#include "TQueueItem.h"
#include "TQueue.h"
#include "Block.h"
#include <memory>

int main(int argc, char** argv) {

    char choice;
    TQueue<Figure> queue;

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
                queue.push(std::shared_ptr<Rectangle>(new Rectangle(std::cin)));
                std::cout << "Добавлен прямоугольник " << std::endl;
                break;
            }
            case 'b': {
                queue.push(std::shared_ptr<Rhomb>(new Rhomb(std::cin)));
                std::cout << "Добавлен ромб " << std::endl;
                break;
            }
            case 't': {
                queue.push(std::shared_ptr<Trapeze>(new Trapeze(std::cin)));
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
                for (auto i: queue) {
                    i->Print();
                    std::cout << '\n';
                }
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


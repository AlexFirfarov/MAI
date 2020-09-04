#include <cstdlib>
#include <iostream>
#include "TQueueItem.h"
#include "TQueue.h"
#include <memory>
#include "IRemoveCriteriaAll.h"
#include "IRemoveCriteriaBySquare.h"


int main(int argc, char** argv) {

    char choice;
    TQueue<TreeNode<Figure>,Figure> queue;

    puts("---------------------MENU-------------------\n");
    puts("        r - Добавить прямоугольник          \n");
    puts("             b - Добавить ромб              \n");
    puts("           t - Добавить трапецию            \n");
    puts("             2 - Извлечь голову             \n");
    puts("           3 - Просмотреть очередь          \n");
    puts("             4 - Удалить очередь            \n");
    puts("       5 - Удаление элементов по площади    \n");
    puts("         6 - Удаление элементов по типу     \n");
    puts("                  x - выход                 \n");
    puts("--------------------------------------------\n");

    do {
        std::cout << '\n';
        std::cin >> choice;
  
        switch(choice) {
            case 'r': {
                queue.PushSubitem(std::shared_ptr<Rectangle>(new Rectangle(std::cin)));
                std::cout << "Добавлен прямоугольник " << std::endl;
                break;
            }
            case 'b': {
                queue.PushSubitem(std::shared_ptr<Rhomb>(new Rhomb(std::cin)));
                std::cout << "Добавлен ромб " << std::endl;
                break;
            }
            case 't': {
                queue.PushSubitem(std::shared_ptr<Trapeze>(new Trapeze(std::cin)));
                std::cout << "Добавлена трапеция " << std::endl;
                break;
            } 
            case '2': {
                queue.Pop();
                break;
            }
            case '3': {
                for (auto i: queue) {
                    i->TPrintTree();
                    std::cout << '\n';
                }
                std::cout << '\n';
                break;
            }
            case '4': {
                queue.~TQueue();
                break;
            }
            case '5': {
                char symbol;
                double square;
                std::cout << "Введите площадь и символ сравнения\n";
                std::cin >> square >> symbol;
                IRemoveCriteriaBySquare<Figure> criteria(square, symbol);
                queue.DeleteSubitem(&criteria);
                break;
            } 
            case '6': {
                int type;
                std::cout << "Введите тип фигуры: 1 - Прямоугольник, 2 - Трапеция, 3 - Ромб\n";
                std::cin >> type;
                IRemoveCriteriaAll<Figure> criteria(type);
                queue.DeleteSubitem(&criteria);
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


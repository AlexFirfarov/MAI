#include <cstdlib>
#include <iostream>
#include "TQueueItem.h"
#include "TQueue.h"
#include "TBinaryTree.h"
#include <memory>
#include <functional>
#include <random>

int main(int argc, char** argv) {

    char choice;
    TQueue<Figure> queue;
    typedef std::function<void (void)> command;
    TBinaryTree <command> tree_com;

    command insert = [&]() {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(1,1000);

        for (int i = 0; i < 5; ++i) {
            int figure = distribution(generator);
            if (figure % 2 == 0) {
                int length = distribution(generator);
                int height = distribution(generator);
                queue.push(std::shared_ptr<Rectangle>(new Rectangle(length, height)));
            }
            else if (figure % 3 == 1) {
                int up_base = distribution(generator);
                int low_base = distribution(generator);
                int height = distribution(generator);
                queue.push(std::shared_ptr<Trapeze>(new Trapeze(up_base, low_base, height)));
            }
            else {
                int rhomb_side = distribution(generator);
                int rhomb_height = distribution(generator);
                queue.push(std::shared_ptr<Rhomb>(new Rhomb(rhomb_side, rhomb_height)));
            }
        }
    };

    command print = [&]() {
        puts("----------------------------------------------\n");
        for (auto i: queue) {
            i->Print();
            std::cout << '\n';
        }
    };

    command del = [&]() {
        double square = 0;
        std::cout << "Введите значение площади" << std::endl;
        std::cin >> square;
        TQueue<Figure> temp;
        while(!queue.empty()) {
            std::shared_ptr<Figure> figure = queue.pop();
            if (figure->Square() >= square) {
                temp.push(std::move(figure));
            }
        }
        while(!temp.empty()) {
            queue.push(temp.pop());
        }

    };
    

    puts("---------------------MENU---------------------\n");
    puts("r - Добавить 5 фигур со случайными параметрами\n");
    puts("           p - Просмотреть очередь            \n");
    puts("            s - Удаление по площади           \n");
    puts("              g - Запустить команды           \n");
    puts("             4 - Удалить очередь              \n");
    puts("                  x - выход                   \n");
    puts("----------------------------------------------\n");

    do {
        std::cout << '\n';
        std::cin >> choice;
  
        switch(choice) {
            case 'r': {
                tree_com.Insert(std::shared_ptr<command> (&insert, [](command*) {}));
                break;
            }
            case 'p':{
                tree_com.Insert(std::shared_ptr<command> (&print, [](command*) {}));
                break;
            }
            case 's':{
                tree_com.Insert(std::shared_ptr<command> (&del, [](command*) {}));
                break;
            }
            case 'g':{
                tree_com.Inorder();
                tree_com.~TBinaryTree();
                break;
            }
            case '4':{
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


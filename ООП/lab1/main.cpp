#include <cstdlib>
#include <iostream>
#include "Rectangle.h"
#include "Trapeze.h"
#include "Rhomb.h"


int main(int argc, char** argv) {
    char choice;

    puts("---------------------MENU-------------------\n");
    puts("              1 - прямоугольник             \n");
    puts("                2 - трапеция                \n");
    puts("                  3 - ромб                  \n");
    puts("                  x - выход                 \n");
    puts("--------------------------------------------\n");

    do {
        std::cout << '\n';
        std::cin >> choice;
  
        switch(choice) {
            case '1':{
                Figure *ptr = new Rectangle(std::cin);
                ptr->Print();
                std::cout << "Площадь: " <<  ptr->Square() << std::endl;
                delete ptr;
                break;
            }
            case '2':{
                Figure *ptr = new Trapeze(std::cin);
                ptr->Print();
                std::cout << "Площадь: " <<  ptr->Square() << std::endl;
                delete ptr;
                break;
            }
            case '3':{
                Figure *ptr = new Rhomb(std::cin);
                ptr->Print();
                std::cout << "Площадь: " <<  ptr->Square() << std::endl;
                delete ptr;
                break;
            }
              case 'x':{
                break;
            }
            default:{
                std::cout << "Введите 1,2,3 или x" << std::endl;
                break;
            }
        }
    } while (choice != 'x');

return 0;
}


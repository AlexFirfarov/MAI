#include "Trapeze.h"
#include <iostream>
#include <cmath>

Trapeze::Trapeze(double a,double b,double c,double d) : up_base(a), low_base(b), left_side(c), right_side(d) {
    std::cout << "Trapeze created: " << std::endl;
}

Trapeze::Trapeze(std::istream &is) {
    std::cout << "Введите верхнее основание трапеции: ";
    is >> up_base;
    std::cout << '\n';
    std::cout << "Введите нижнее основание трапеции: ";
    is >> low_base;
    std::cout << '\n';
    std::cout << "Введите боковые стороны через пробел: ";
    is >> left_side;
    is >> right_side;
    std::cout << '\n';
}

double Trapeze::Square() {
  double p = (up_base + low_base) / 2.0;
  double temp = ((low_base - up_base) * (low_base - up_base) + left_side * left_side - right_side * right_side)/(2*(low_base - up_base));  
  return p*sqrt(left_side*left_side - temp * temp);
}

void Trapeze::Print() {
    std::cout << "Тип: трапеция" << std::endl;
    std::cout << "Верхнее основание: " <<  up_base << std::endl;
    std::cout << "Нижнее основание: " << low_base << std::endl;
    std::cout << "Левая сторона: " << left_side << std::endl;
    std::cout << "Правая сторона: " << right_side << std::endl;
}

Trapeze::~Trapeze() {
    std::cout << "Trapeze deleted" << std::endl;
}
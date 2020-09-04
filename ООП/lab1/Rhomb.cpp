#include "Rhomb.h"
#include <iostream>
#include <cmath>

Rhomb::Rhomb(double a) : rhomb_side(a) {
    std::cout << "Rhomb created: " << std::endl;
}

Rhomb::Rhomb(std::istream &is) {
    std::cout << "Введите сторону ромба: ";
    is >> rhomb_side;
    std::cout << '\n';
    std::cout << "Введите высоту ромба: ";
    is >> rhomb_height;
    std::cout << '\n';
}

double Rhomb::Square() {
    return rhomb_side * rhomb_height;
}

void Rhomb::Print() {
    std::cout << "Тип: ромб" << std::endl;
    std::cout << "Сторона ромба: " << rhomb_side << std::endl;
    std::cout << "Высота: " << rhomb_height << std::endl;
}

Rhomb::~Rhomb() {
    std::cout << "Rhomb deleted" << std::endl;
}
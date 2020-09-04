#include "Rectangle.h"
#include <iostream>
#include <cmath>

Rectangle::Rectangle(double a,double b) : len_rec(a), height_rec(b) {
    std::cout << "Rectangle created: " << std::endl;
}

Rectangle::Rectangle(std::istream &is) {
    std::cout << "Введите длину прямоугольника: ";
    is >> len_rec;
    std::cout << '\n';
    std::cout << "Введите высоту прямоугольника: ";
    is >> height_rec;
    std::cout << '\n';
}

double Rectangle::Square() {
    return len_rec * height_rec;
}

void Rectangle::Print() {
    std::cout << "Тип: прямоугольник" << std::endl;
    std::cout << "Длина: " << len_rec << std::endl;
    std::cout << "Высота: " << height_rec << std::endl;
}

Rectangle::~Rectangle() {
    std::cout << "Rectangle deleted" << std::endl;
}
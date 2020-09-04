#include "Rectangle.h"
#include <iostream>

Rectangle::Rectangle() : Rectangle(0,0) {
}

Rectangle::Rectangle(double a, double b) : length(a), height(b) {
}

double Rectangle::Square() {
    return length * height;
}

Rectangle& Rectangle::operator=(const Rectangle& right) {

    length = right.length;
    height = right.height;

    return *this;
}

bool Rectangle::operator==(const Rectangle& right) {
    if ((length == right.length) && (height == right.height)) {
        return true;
    }
    return false;
}

std::ostream& operator<<(std::ostream& os, Rectangle& obj) {

    os << "Длина: " << obj.length << ", Высота: " << obj.height << ", Площадь: " << obj.Square() << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Rectangle& obj) {

    do {
        std::cout << "Введите длину: " << std::endl;
        is >> obj.length;
    } while (obj.length < 0);

    do {
        std::cout << "Введите ширину: " << std::endl;
        is >> obj.height;
    } while (obj.height < 0);
    

    return is;
}

Rectangle::~Rectangle() {
}
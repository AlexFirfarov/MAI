#include "Rectangle.h"
#include <iostream>

Rectangle::Rectangle() : Rectangle(0,0) {
}

Rectangle::Rectangle(double a, double b) : length(a), height(b) {
}

Rectangle::Rectangle(std::istream &is) {

    do {
        std::cout << "Введите длину: " << std::endl;
        is >> length;
    } while (length < 0);

    do {
        std::cout << "Введите ширину: " << std::endl;
        is >> height;
    } while (height < 0);

}

void Rectangle::Print() {
    std::cout << "Прямоугольник" << std::endl;
    std::cout << "Длина: " << length << std::endl;
    std::cout << "Высота: " << height << std::endl;
    std::cout << "Площадь: " << Square() << std::endl;
}

double Rectangle::Square() {
    return length * height;
}

Rectangle& Rectangle::operator=(const Rectangle& right) {

    length = right.length;
    height = right.height;

    return *this;
}

Rectangle::operator double () const {
    return length * height;
}

bool Rectangle::operator==(const Figure& right) {
    return (double)(*this) == (double)(right);
}

bool Rectangle::operator<(const Figure& right) {
    return (double)(*this) < (double)(right);
}

bool Rectangle::operator>(const Figure& right) {
    return (double)(*this) > (double)(right);
}

bool Rectangle::operator<=(const Figure& right) {
    return (double)(*this) <= (double)(right);
}

bool Rectangle::operator>=(const Figure& right) {
    return (double)(*this) >= (double)(right);
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
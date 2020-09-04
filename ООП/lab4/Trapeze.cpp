#include "Trapeze.h"
#include <iostream>
#include <cmath>

Trapeze::Trapeze() : Trapeze(0,0,0) {
}

Trapeze::Trapeze(double a, double b, double c) : up_base(a), low_base(b), height(c) {
}

Trapeze::Trapeze(std::istream &is) {
    do {
        std::cout << "Введите верх. основание.: " << std::endl;
        is >> up_base;
    } while (up_base < 0);

    do {
        std::cout << "Введите нижн. основание: " << std::endl;
        is >> low_base;
    } while (low_base < 0);

    do {
        std::cout << "Введите высоту: " << std::endl;
        is >> height;
    } while (height < 0);

}

void Trapeze::Print() {
    std::cout << "Трапеция" << std::endl;
    std::cout << "Верхнее основание: " <<  up_base << std::endl;
    std::cout << "Нижнее основание: " << low_base << std::endl;
    std::cout << "Высота: " << height << std::endl;
    std::cout << "Площадь: " << Square() << std::endl;
}

double Trapeze::Square() {
    return (up_base + low_base) * 0.5 * height;
}

Trapeze& Trapeze::operator=(const Trapeze& right) {

    up_base = right.up_base;
    low_base = right.low_base;
    height = right.height;

    return *this;
}

bool Trapeze::operator==(const Trapeze& right) {
    if ((up_base == right.up_base) && (low_base == right.low_base) && (height == right.height)) {
        return true;
    }
    return false;
}

std::ostream& operator<<(std::ostream& os, Trapeze& obj) {

    os << "Верх. осн: " << obj.up_base << ", Нижн. осн: " << obj.low_base << ", Высота: " << obj.height << ", Площадь: " << obj.Square() << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Trapeze& obj) {

    do {
        std::cout << "Введите верх. основание.: " << std::endl;
        is >> obj.up_base;
    } while (obj.up_base < 0);

    do {
        std::cout << "Введите нижн. основание: " << std::endl;
        is >> obj.low_base;
    } while (obj.low_base < 0);

    do {
        std::cout << "Введите высоту: " << std::endl;
        is >> obj.height;
    } while (obj.height < 0);
    

    return is;
}

Trapeze::~Trapeze() {
}
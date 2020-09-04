#include "Rhomb.h"
#include <iostream>
#include <cmath>

Rhomb::Rhomb() : Rhomb(0,0) {
}

Rhomb::Rhomb(double a, double b) : rhomb_side(a), rhomb_height(b) {
}

Rhomb::Rhomb(std::istream &is) {
    do {
        std::cout << "Введите сторону: " << std::endl;
        is >> rhomb_side;
    } while (rhomb_side < 0);

    do {
        std::cout << "Введите высоту: " << std::endl;
        is >> rhomb_height;
    } while (rhomb_height < 0);

}

void Rhomb::Print() {
    std::cout << "Ромб" << std::endl;
    std::cout << "Сторона ромба: " << rhomb_side << std::endl;
    std::cout << "Высота: " << rhomb_height << std::endl;
    std::cout << "Площадь: " << Square() << std::endl;
}

double Rhomb::Square() {
    return rhomb_side * rhomb_height;
}

Rhomb& Rhomb::operator=(const Rhomb& right) {

    rhomb_side = right.rhomb_side;
    rhomb_height = right.rhomb_height;

    return *this;
}

bool Rhomb::operator==(const Rhomb& right) {
    if ((rhomb_side == right.rhomb_side) && (rhomb_height == right.rhomb_height)) {
        return true;
    }
    return false;
}

std::ostream& operator<<(std::ostream& os, Rhomb& obj) {

    os << "Сторона: " << obj.rhomb_side << ", Высота: " << obj.rhomb_height << ", Площадь: " << obj.Square() << std::endl;
    return os;
}

std::istream& operator>>(std::istream& is, Rhomb& obj) {

    do {
        std::cout << "Введите сторону: " << std::endl;
        is >> obj.rhomb_side;
    } while (obj.rhomb_side < 0);

    do {
        std::cout << "Введите высоту: " << std::endl;
        is >> obj.rhomb_height;
    } while (obj.rhomb_height < 0);
    

    return is;
}

Rhomb::~Rhomb() {
}
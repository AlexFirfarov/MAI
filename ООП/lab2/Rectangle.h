#ifndef RECTANGLE_H
#define	RECTANGLE_H
#include <cstdlib>
#include <iostream>

class Rectangle {
public:

    Rectangle();
    Rectangle(double a,double b);

    double Square();

    friend std::ostream& operator<<(std::ostream& os, Rectangle& obj);
    friend std::istream& operator>>(std::istream& is,  Rectangle& obj);
    Rectangle& operator=(const Rectangle& right);
    bool operator==(const Rectangle& right);

    ~Rectangle();

private:
    double length;
    double height;
};

#endif
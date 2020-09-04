#ifndef RECTANGLE_H
#define	RECTANGLE_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Rectangle: public Figure {
public:

    Rectangle();
    Rectangle(double a,double b);
    Rectangle(std::istream &is);

    double Square() override;
    void Print() override;
    int GetType() override;

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
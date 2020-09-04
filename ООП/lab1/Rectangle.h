#ifndef RECTANGLE_H
#define	RECTANGLE_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Rectangle : public Figure {
public:
    Rectangle(double a = 0, double b = 0);
    Rectangle(std::istream &is);

    double Square() override;
    void   Print() override;

    virtual ~Rectangle();
private:
    double len_rec;
    double height_rec;
};

#endif
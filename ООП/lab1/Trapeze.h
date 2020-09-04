#ifndef TRAPEZE_H
#define	TRAPEZE_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Trapeze : public Figure {
public:
    Trapeze(double a = 0, double b = 0, double c = 0, double d = 0);
    Trapeze(std::istream &is);

    double Square() override;
    void   Print() override;

    virtual ~Trapeze();
private:
    double up_base;
    double low_base;
    double left_side;
    double right_side;
};

#endif
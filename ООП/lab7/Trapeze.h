#ifndef TRAPEZE_H
#define	TRAPEZE_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Trapeze : public Figure {
public:

    Trapeze();
    Trapeze(double a,double b,double c);
    Trapeze(std::istream &is);

    double Square() override;
    void Print() override;
    int GetType() override;

    friend std::ostream& operator<<(std::ostream& os, Trapeze& obj);
    friend std::istream& operator>>(std::istream& is,  Trapeze& obj);
    Trapeze& operator=(const Trapeze& right);
    bool operator==(const Trapeze& right);

    ~Trapeze();

private:
    double up_base;
    double low_base;
    double height;
};

#endif
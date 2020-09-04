#ifndef RHOMB_H
#define	RHOMB_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Rhomb : public Figure {
public:
    Rhomb(double a = 0);
    Rhomb(std::istream &is);

    double Square() override;
    void   Print() override;

    virtual ~Rhomb();
private:
    double rhomb_side;
    double rhomb_height;
};

#endif
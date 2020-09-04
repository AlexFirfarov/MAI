#ifndef RHOMB_H
#define	RHOMB_H
#include <cstdlib>
#include <iostream>
#include "Figure.h"

class Rhomb : public Figure {
public:

    Rhomb();
    Rhomb(double a,double b);
    Rhomb(std::istream &is);    

    double Square() override;
    void Print() override;

    friend std::ostream& operator<<(std::ostream& os, Rhomb& obj);
    friend std::istream& operator>>(std::istream& is,  Rhomb& obj);
    Rhomb& operator=(const Rhomb& right);
    bool operator==(const Rhomb& right);

    ~Rhomb();
    
private:
    double rhomb_side;
    double rhomb_height;
};

#endif
#ifndef TBIGINT_H
#define TBIGINT_H

#include <iostream>
#include <vector>
#include <sstream>
#include <iomanip>

const int BASE = 10000;
const int MAX_POW = 4;

class TBigInt {

public:
    TBigInt() {};
    TBigInt(std::string& num);
    TBigInt(int n);
    std::vector<int> data;

    friend TBigInt operator + (const TBigInt& first, const TBigInt& second);
    friend TBigInt operator - (const TBigInt& first, const TBigInt& second);
    friend TBigInt operator * (const TBigInt& first, const TBigInt& second);
    friend TBigInt operator / (const TBigInt& first, const TBigInt& second);
    friend TBigInt Power(const TBigInt& num, int n);

    friend bool operator < (const TBigInt& first, const TBigInt& second);
    friend bool operator > (const TBigInt& first, const TBigInt& second);
    friend bool operator == (const TBigInt& first, const TBigInt& second);

    friend std::ostream& operator << (std::ostream& stream, const TBigInt& res);

private:

    void FilterZero();
    int GetDigit(int pos) const;
    
};

#endif
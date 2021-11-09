#include "BigInt.h"

TBigInt::TBigInt(std::string& num) {

    if (num[0] == '0') {
        int i = 1;
        for(;i < num.size(); ++i) {
            if (num[i] != '0') {
                break;
            }
        }
        num = (i == num.size()) ? "0" : num.substr(i);
    }
    this->data.clear();
    for (int i = num.size() - 1; i >= 0; i -= MAX_POW) {
        int first = i - MAX_POW + 1;
        if (first < 0) {
            first = 0;
        }
        this->data.push_back(std::stoi(num.substr(first, i - first + 1)));
    }
}

TBigInt::TBigInt(int n) {
    if (n < BASE) {
        this->data.push_back(n);
    }
    else {
        for(; n; n /= BASE) {
            this->data.push_back(n % BASE);
        }
    }
}

int TBigInt::GetDigit(int pos) const{
    if (pos >= this->data.size()) {
        return 0;
    }
    return this->data[pos];
}

void TBigInt::FilterZero() {
    for (int i = this->data.size() - 1; i > 0; --i) {
        if (this->data[i] != 0) {
            break;
        }
        this->data.pop_back();
    }
}

TBigInt operator +(const TBigInt& first, const TBigInt& second) {
    TBigInt res;
    res.data.resize(std::max(first.data.size(),second.data.size()));
    int r = 0;
    for (int i = 0; i < res.data.size(); ++i) {
        res.data[i] = first.GetDigit(i) + second.GetDigit(i) + r;
        r = res.data[i] / BASE;
        res.data[i] %= BASE;
    }
    if (r > 0) {
        res.data.push_back(r);
    }
    return res;
}

TBigInt operator -(const TBigInt& first, const TBigInt& second) {
    TBigInt res;
    
    res.data.resize(first.data.size());
    int r = 0;
    for (int i = 0; i < res.data.size(); ++i) {
        res.data[i] = first.GetDigit(i) - second.GetDigit(i) - r;
        r = 0;
        if (res.data[i] < 0) {
            res.data[i] += BASE;
            r = 1;
        }
    }
    res.FilterZero();
    return res;
}

TBigInt operator *(const TBigInt& first, const TBigInt& second) {
    TBigInt res;
    int sizeOne = first.data.size();
    int sizeTwo = second.data.size();
    res.data.resize(sizeOne + sizeTwo);

    for (int i = 0; i < sizeOne; ++i) {
        int r = 0;
        for (int j = 0; j < sizeTwo || r; ++j) {
            res.data[i + j] += first.GetDigit(i) * second.GetDigit(j) + r;
            r = res.data[i + j] / BASE;
            res.data[i + j] %= BASE;
        }
    }
    res.FilterZero();
    return res;
}

TBigInt operator /(const TBigInt& first, const TBigInt& second) {
    TBigInt res;
    TBigInt curVal = TBigInt(0);
    res.data.resize(first.data.size());

    for (int i = first.data.size() - 1; i >= 0; --i) {
        curVal.data.insert(curVal.data.begin(), first.data[i]);
        if (!curVal.data.back()) {
            curVal.data.pop_back();
        }
        int x = 0, l = 0, r = BASE;
        while (l <= r) {
            int m = (l + r) / 2;

            TBigInt cur = second * TBigInt(m);

            if ((cur < curVal) || (cur == curVal)) {
                x = m;
                l = m + 1;
            } 
            else {
                r = m - 1;
            }
        }
        res.data[i] = x;

        curVal = curVal - second * TBigInt(x);
    }
    res.FilterZero();
    return res;
}

TBigInt Power(const TBigInt& num, int n) {
    TBigInt res;
    if (n == 0) {
        res.data.push_back(1);
        return res;
    }
    if (n == 1) {
        return num;
    }
    if (n & 1) {
        res = Power(num, n/2);
        res = res * res;

        return res * num;
    }
    else {
        res = Power(num, n/2);
        return res * res;
    }
}

bool operator <(const TBigInt& first, const TBigInt& second) {
    if (second.data.size() != first.data.size()) {
        return first.data.size() < second.data.size();
    }
    for (int i = first.data.size() - 1; i >= 0; --i) {
        if (first.data[i] != second.data[i]) {
            return second.data[i] > first.data[i];
        }
    }
    return false;
}

bool operator >(const TBigInt& first, const TBigInt& second) {
    if (first.data.size() != second.data.size()) {
        return first.data.size() > second.data.size();
    }
    for (int i = first.data.size() - 1; i >= 0; --i) {
        if (first.data[i] != second.data[i]) {
            return first.data[i] > second.data[i];
        }
    }
    return false;
}

bool operator ==(const TBigInt& first, const TBigInt& second) {
    if (first.data.size() != second.data.size()) {
        return false;
    }
    for (int i = first.data.size() - 1; i >= 0; --i) {
        if (first.data[i] != second.data[i]) {
            return false;
        }
    }
    return true;
}

std::ostream& operator <<(std::ostream& stream, const TBigInt& num) {
    int n = num.data.size();
    if (!n)
        return stream;
    stream << num.data[n - 1];
    for (int i = n - 2; i >= 0; --i)
        stream << std::setfill('0') << std::setw(MAX_POW) << num.data[i];

    return stream;
}

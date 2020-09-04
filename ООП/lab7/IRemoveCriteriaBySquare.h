#ifndef IREMOVECRITERIABYSQUARE_H
#define IREMOVECRITERIABYSQUARE_H

#include "IRemoveCriteria.h"

template <class TT> class IRemoveCriteriaBySquare : public IRemoveCriteria<TT> {
public:
    IRemoveCriteriaBySquare(double _square, char _symbol) : square(_square), symbol(_symbol) {};
    bool isIt(std::shared_ptr<TT> obj) override {
        switch(symbol) {
            case '<' : {
                return obj->Square() < square;
                break;
            }
            case '=' : {
                return obj->Square() == square;
                break;
            }
            case '>' : {
                return obj->Square() > square;
                break;
            } 
        }
    }
private:
    double square;
    char symbol;
};

#endif
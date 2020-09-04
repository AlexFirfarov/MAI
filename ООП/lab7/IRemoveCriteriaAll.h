#ifndef IREMOVECRITERIABYTYPE_H
#define IREMOVECRITERIABYTYPE_H

#include "IRemoveCriteria.h"

template <class TT> class IRemoveCriteriaAll : public IRemoveCriteria<TT> {
public:
    IRemoveCriteriaAll(int _type) : type(_type) {};
    bool isIt(std::shared_ptr<TT> obj) override {
        return type == obj->GetType();
    }
private:
    int type;
};

#endif
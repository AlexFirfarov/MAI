#ifndef IREMOVECRITERIA_H
#define IREMOVECRITERIA_H
#include <memory>

template <class TT> class IRemoveCriteria {
public:
    virtual bool isIt(std::shared_ptr<TT> obj) {};
};

#endif
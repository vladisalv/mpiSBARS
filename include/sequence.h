#ifndef __SEQUENCE_HEADER__
#define __SEQUENCE_HEADER__

#include "arrayMPI.h"
#include "types.h"

class Sequence : public ArrayMPI<char, ulong> {
public:
    Sequence(MyMPI me);
    ~Sequence();
    friend class Profiling;
};

#endif /* __SEQUENCE_HEADER__ */

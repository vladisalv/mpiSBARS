#ifndef __SEQUENCE_HEADER__
#define __SEQUENCE_HEADER__

#include "types.h"
#include "arrayMPI.h"

class Sequence : public ArrayMPI<TypeSequence> {
public:
    Sequence(MyMPI me);
    ~Sequence();
    friend class Profiling;
};

#endif /* __SEQUENCE_HEADER__ */

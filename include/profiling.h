#ifndef __PROFILING_HEADER__
#define __PROFILING_HEADER__

#include "types.h"
#include "sequence.h"
#include "profile.h"

class Profiling {
    MyMPI me;
public:
    Profiling(MyMPI me);
    ~Profiling();

    Profile doProfile(Sequence &sequence, char ch1, char ch2, uint window);
};

#endif /* __PROFILING_HEADER__ */

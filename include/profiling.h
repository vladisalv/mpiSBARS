#ifndef __PROFILING_HEADER__
#define __PROFILING_HEADER__

#include "types.h"
#include "sequence.h"
#include "profile.h"

class Profiling {
    MyMPI me;
    uint window;
public:
    Profiling(MyMPI me, uint window);
    ~Profiling();

    Profile doProfile(Sequence &sequence, char ch1, char ch2);

    void setLengthWindowProfile(uint new_window);
    uint getLengthWindowProfile();
};

#endif /* __PROFILING_HEADER__ */

#ifndef __PROFILE_HEADER__
#define __PROFILE_HEADER__

#include "arrayMPI.h"
#include "types.h"

class Profile : public ArrayMPI<TypeProfile, ulong> {
public:
    Profile(MyMPI me);
    ~Profile();
    friend class Profiling;
    friend class Decompose;
};

#endif /* __PROFILE_HEADER__ */

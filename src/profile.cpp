#include "profile.h"

Profile::Profile(MyMPI me)
    : ArrayMPI<TypeProfile, ulong>(me)
{
}

Profile::~Profile()
{
}

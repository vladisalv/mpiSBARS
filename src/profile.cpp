#include "profile.h"

Profile::Profile(MyMPI me)
    : ArrayMPI<TypeProfile, ulong>(me, "Profile", MPI_TYPE_PROFILE)
{
}

Profile::~Profile()
{
}

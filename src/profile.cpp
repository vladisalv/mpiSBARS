#include "profile.h"

Profile::Profile(MyMPI me)
    : ArrayMPI<TypeProfile>(me, "Profile", MPI_TYPE_PROFILE)
{
}

Profile::~Profile()
{
}

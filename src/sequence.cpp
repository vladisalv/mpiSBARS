#include "sequence.h"

Sequence::Sequence(MyMPI me)
    : ArrayMPI<TypeSequence, ulong>(me, "Sequence", MPI_TYPE_SEQUENCE)
{
}

Sequence::~Sequence()
{
}

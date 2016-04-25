#include "sequence.h"

Sequence::Sequence(MyMPI me)
    : ArrayMPI<TypeSequence>(me)
{
}

Sequence::~Sequence()
{
}

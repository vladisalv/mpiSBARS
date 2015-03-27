#include "set_repeats.h"

Coordinates::Coordinates()
    : x(0), y(0)
{
}

Coordinates::Coordinates(long nx, long ny)
    : x(nx), y(ny)
{
}


Repeat::Repeat()
    : begin(0,0), end(0,0), length(0)
{
}

Repeat::Repeat(Coordinates nbegin, Coordinates nend, long len)
    : begin(nbegin), end(nend), length(len)
{
}

Repeat::Repeat(long x1, long y1, long x2, long y2, long len)
    : begin(x1, y1), end(x2, y2), length(len)
{
}

void Repeat::Print()
{
    printf("%ld %ld %ld %ld %ld\n", begin.x, begin.y, end.x, end.y, length);
}


SetRepeats::SetRepeats(MyMPI me)
    : DataMPI<TypeAnalysis, ulong>(me)
{
}

SetRepeats::~SetRepeats()
{
}

void SetRepeats::readMPI(char *file_name)
{
}

void SetRepeats::readUsually(char *file_name)
{
}

void SetRepeats::readMy(char *file_name)
{
}

void SetRepeats::writeMPI(char *file_name)
{
}

void SetRepeats::writeUsually(char *file_name)
{
}

void SetRepeats::writeMy(char *file_name)
{
}


void SetRepeats::analyzeOtherProcess()
{
}

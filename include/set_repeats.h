#ifndef __SET_REPEATS_HEADER__
#define __SET_REPEATS_HEADER__

#include "myMPI.h"
#include "dataMPI.h"
#include "types.h"

#include <vector>

using namespace std;

struct Coordinates {
    long x, y;
    Coordinates(); // ???
    Coordinates(long x, long y);
};

struct Repeat {
    Coordinates begin, end;
    long length;
    Repeat(); // ???
    Repeat(Coordinates begin, Coordinates end, long len);
    Repeat(long x1, long y1, long x2, long y2, long len); // ???
    void Print();
};


typedef vector<Repeat> TypeAnalysis;

class SetRepeats : public DataMPI<TypeAnalysis, ulong> {
    virtual void readMPI(char *file_name);
    virtual void readUsually(char *file_name);
    virtual void readMy(char *file_name);

    virtual void writeMPI(char *file_name);
    virtual void writeUsually(char *file_name);
    virtual void writeMy(char *file_name);
public:
    SetRepeats(MyMPI me);
    ~SetRepeats();

    void analyzeOtherProcess();
};

#endif /* __SET_REPEATS_HEADER__ */

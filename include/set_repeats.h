#ifndef __SET_REPEATS_HEADER__
#define __SET_REPEATS_HEADER__

#include "myMPI.h"
#include "dataMPI.h"
#include "types.h"

#include <vector>

using namespace std;


struct Repeat {
    ulong x_begin, y_begin;
    ulong x_end, y_end;
    ulong length;
    Repeat(ulong x_begin, ulong y_begin, ulong x_end, ulong y_end, ulong len);
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

    TypeAnalysis vec;
    ulong x_limit_left, x_limit_right, y_limit_above, y_limit_bottom;
public:
    SetRepeats(MyMPI me);
    ~SetRepeats();

    void analyzeOtherProcess();
    virtual void debugInfo(const char *file, int line, const char *info = 0);

    friend class Analyze;
};

#endif /* __SET_REPEATS_HEADER__ */

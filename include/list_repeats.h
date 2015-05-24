#ifndef __SET_REPEATS_HEADER__
#define __SET_REPEATS_HEADER__

#include "myMPI.h"
#include "dataMPI.h"
#include "types.h"

#include <list>
#include <sstream>
#include <string>

using namespace std;


struct Repeat {
    ulong x_begin, y_begin;
    ulong x_end, y_end;
    ulong length;
    Repeat(ulong x_begin, ulong y_begin, ulong x_end, ulong y_end, ulong len);
    void Print();
};

typedef list<Repeat> TypeAnalysis;

class ListRepeats : public DataMPI<TypeAnalysis, ulong> {
    virtual void readMPI(char *file_name);
    virtual void readUsually(char *file_name);
    virtual void readMy(char *file_name);

    virtual void writeMPI(char *file_name);
    virtual void writeUsually(char *file_name);
    virtual void writeMy(char *file_name);

    ulong x_limit_left, x_limit_right, y_limit_above, y_limit_bottom;
public:
    ListRepeats(MyMPI me);
    ~ListRepeats();

    void mergeRepeats();
    void convertToOriginalRepeats(uint window_profiling, uint window_decompose, uint step_decompose, uint number_coef);
    virtual void debugInfo(const char *file, int line, const char *info = 0);

    void makeOffsetRow(ulong offset);
    void makeOffsetColumn(ulong offset);
    void mergeRepeatsRow(ListRepeats list);
    void mergeRepeatsColumn(ListRepeats list);
    friend class Analyze;

    void writeRepeat(TypeAnalysis list);
};

#endif /* __SET_REPEATS_HEADER__ */

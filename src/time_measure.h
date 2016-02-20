#ifndef __TIME_MEASURE_HEADER__
#define __TIME_MEASURE_HEADER__

#include "types.h"
#include "myMPI.h"

struct CheckPoint  {
    char  *label;
    double time;
    bool   pause;
    CheckPoint *next;
    CheckPoint(char *label, double time, bool pause = false);
    ~CheckPoint();
};

class TimeMeasure {
    MyMPI *me;
    bool start, finish, pause, work;
    int number_pause;
public:
    CheckPoint *checkpoint;

    TimeMeasure(MyMPI *me);
    ~TimeMeasure();

    void startTime();
    void pauseIt();
    void continueIt();
    void doCheckPoint(char *label);
    void stopTime();
    void statisticTime(char *label);
    void resetClock();
};

#endif /* __TIME_MEASURE_HEADER__ */

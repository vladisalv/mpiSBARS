#include "time_measure.h"

CheckPoint::CheckPoint(char *new_label, double new_time, bool new_pause)
{
    label = new_label;
    time  = new_time;
    pause = new_pause;
    next = 0;
}


TimeMeasure::TimeMeasure(MyMPI *new_me)
{
    me = new_me;
    start = finish = pause = work = false;
    checkpoint = 0;
    number_pause = 0;
}

void TimeMeasure::startTime()
{
    start = work = true;
    finish = pause = false;
    checkpoint = new CheckPoint((char *)"start", me->getTime());
}

void TimeMeasure::pauseIt()
{
    work = false;
    pause = true;
    CheckPoint *tmp_link = checkpoint;
    char *pause_link = (char *)"pause000";
    while (tmp_link->next != 0) 
        tmp_link = tmp_link->next;
    tmp_link = new CheckPoint(pause_link, me->getTime(), true);
}

void TimeMeasure::continueIt()
{
    work = true;
    pause = false;
}

void TimeMeasure::doCheckPoint(char *label)
{
    CheckPoint *tmp_link = checkpoint;
    while (tmp_link->next != 0)
        tmp_link = tmp_link->next;
    tmp_link = new CheckPoint((char *)"label", me->getTime());
}

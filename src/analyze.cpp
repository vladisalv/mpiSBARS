#include "analyze.h"



Analyze::Analyze(MyMPI me_new, GpuComputing gpu_new, double eps_new, ulong min_length_new, double fidelity_repeat_new, size_t limit_memory_new)
    : me(me_new), gpu(gpu_new), matrix(me_new), eps(eps_new), min_length(min_length_new), fidelity_repeat(fidelity_repeat_new), limit_memory(limit_memory_new)
{
    MPI_Comm_group(MPI_COMM_WORLD, &group_comm_world);
    MPI_Type_get_extent(MPI::BOOL, &lb, &extent);
}

Analyze::~Analyze()
{
}


/*
SetRepeats Analyze::doAnalyze(MatrixGomology matrixGomology)
{
    SetRepeats result();
}
*/

SetRepeats Analyze::doAnalyze(MatrixGomology matrixGomology)
{
    matrix = matrixGomology;

    ulong *offset, *sum_offset;
    matrix.offsetLength(offset, sum_offset, &matrix.height);
    my_global_height = sum_offset[me.getRank()];
    delete [] offset;
    delete [] sum_offset;

    /*
    MPI_Win_create(flag_end, me.isLast() ? 0 : matrix.width * extent,
                    extent, MPI_INFO_NULL, MPI_COMM_WORLD, &win_flag_end);
    if (!me.isLast()) {
        flag_end = new bool [matrix.width];
        for (int x = 0; x < matrix.width; x++)
            flag_end[x] = matrix.data[x];
            //flag_end[x] = searchRepeat(Coordinates(x, matrix.height - 1)).x > 0 ? true : false;
        int next = me.getRank() + 1;
        MPI_Group_incl(group_comm_world, 1, &next, &group_next);
        MPI_Win_post(group_next, 0, win_flag_end);
    }
    */

    localSearch0();

    /*
    if (!me.isFirst()) {
        flag_end_prev = new bool [repeat_begin.size()];
        memset(flag_end_prev, 0, repeat_begin.size() * sizeof(bool));
        int prev = me.getRank() - 1;
        MPI_Group_incl(group_comm_world, 1, &prev, &group_prev);
        MPI_Win_start(group_prev, 0, win_flag_end);
        list<struct Repeat>::iterator iter = repeat_begin.begin();
        for (int i = 0; i < repeat_begin.size(); i++) {
            MPI_Get(&flag_end_prev[i], 1, MPI::BOOL, prev, iter->begin.x, 1, MPI::BOOL, win_flag_end);
            iter++;
        }
    }
    */

    localSearch1_n();

    //if (!me.isFirst())
        //MPI_Win_complete(win_flag_end);

    //analysisRepeatBegin();
    //formRepeatAlien();
    //analysisRepeatUnknow();

    sortRepeatAnswer();
    //MatrixAnalysis result_matrix = formResultMatrix();
    list<struct Repeat>::iterator beg = repeat_answer.begin();
    list<struct Repeat>::iterator end = repeat_answer.end();
    while (beg != end) {
        beg->Print();
        beg++;
    }
    repeat_answer.clear();

    //if (!me.isLast())
        //MPI_Win_wait(win_flag_end);
    //MPI_Win_wait(win_repeat_alien);
    //MPI_Win_free(&win_flag_end);
    //MPI_Win_free(&win_repeat_alien);

    //delete [] flag_end;
    //delete [] repeat_alien;

    //MPI_Group_free(&group_comm_world);
    //MPI_Group_free(&group_prev);
    //MPI_Group_free(&group_next);
    //MPI_Group_free(&group_prev_all);
    //MPI_Group_free(&group_next_all);

    SetRepeats result(me);
    return result;
}


SetRepeats Analyze::doAnalyze(Decomposition decomposition)
{
    SetRepeats result(me);
    return result;
}

SetRepeats Analyze::doAnalyze(Decomposition decomposition1, Decomposition decomposition2)
{
    SetRepeats result(me);
    return result;
}

SetRepeats Analyze::comparisonRepeats(SetRepeats setRepeats1, SetRepeats setRepeats2)
{
    SetRepeats result(me);
    return result;
}


void Analyze::localSearch0()
{
    for (ulong x = 0; x < matrix.width; x++)
        if (matrix.data[x]) {
            repeat_begin.push_back(findRepeat(Coordinates(x,0)));
        }
}

void Analyze::localSearch1_n()
{
    for (ulong y = 0; y < matrix.height; y++)
        for (ulong x = 0; x < matrix.width; x++)
            if (matrix.data[y * matrix.width + x]) {
                Repeat rep = findRepeat(Coordinates(x, y));
                if (rep.length > min_length)
                    repeat_answer.push_back(rep);
                else if (rep.length < 0)
                    repeat_unknow.push_back(rep);
                else
                    continue;
            }
}

Repeat Analyze::findRepeat(Coordinates cor)
{
    long x_end = cor.x;
    long y_end = cor.y;
    long length_rep = 0;
    while (true) {
        while (matrix.data[y_end * matrix.width + x_end]
                && x_end < matrix.width && y_end < matrix.height) {
            matrix.data[y_end * matrix.width + x_end] = false;
            x_end++;
            y_end++;
            length_rep++;
        }
        break;
        /*
        Coordinates tmp = searchRepeat(Coordinates(x_end, y_end));
        if (x_end > matrix.width)
            break;
        if (tmp.x == -1 || y_end > tmp.y)
            break;
        x_end = tmp.x;
        y_end = tmp.y;
        */
    }
    if (y_end == matrix.height)
        length_rep = -length_rep;
    return Repeat(Coordinates(cor.x, my_global_height + cor.y),
                  Coordinates(x_end, my_global_height + y_end), length_rep);
}

Coordinates Analyze::searchRepeat(Coordinates cor)
{
    if (matrix.data[cor.y * matrix.width + cor.x])
        return Coordinates(cor.x, cor.y);
    else
        return Coordinates(-1, -1);
}

void Analyze::analysisRepeatBegin()
{
    list<struct Repeat>::iterator i = repeat_begin.begin();
    list<struct Repeat>::iterator j = repeat_begin.end();
    int k = 0;
    while (i != j) {
        if ((!me.isFirst() && !flag_end_prev[k] || me.isFirst()) && i->length > min_length) {
            repeat_answer.push_back(*i);
            repeat_begin.erase(i++);
        } else if ((!me.isFirst() && !flag_end_prev[k] || me.isFirst()) && i->length < 0) {
            repeat_unknow.push_back(*i);
            repeat_begin.erase(i++);
        } else {
            i++;
        }
        k++;
    }
    delete [] flag_end_prev;
    if (me.isLast()) {
        list<struct Repeat>::iterator i = repeat_unknow.begin();
        list<struct Repeat>::iterator j = repeat_unknow.end();
        while (i != j)
            if (i->length < 0)
                i->length = -i->length;
    }
}

void Analyze::formRepeatAlien()
{
    if (repeat_begin.size() > 0) {
        repeat_alien_size = matrix.width;
        repeat_alien = new Repeat [repeat_alien_size];
        for (int i = 0; i < repeat_alien_size; i++) {
            repeat_alien[i].begin.x = repeat_alien[i].begin.y = -1;
            repeat_alien[i].end.x = repeat_alien[i].end.y = -1;
            repeat_alien[i].length = -1;
        }
        list<struct Repeat>::iterator i = repeat_begin.begin();
        list<struct Repeat>::iterator j = repeat_begin.end();
        while (i != j) {
            repeat_alien[i->begin.x].begin.x = i->begin.x;
            repeat_alien[i->begin.x].begin.y = i->begin.x;
            repeat_alien[i->begin.x].end.x   = i->end.x;
            repeat_alien[i->begin.x].end.y   = i->end.x;
            repeat_alien[i->begin.x].length  = i->length;
        }
        repeat_begin.clear();
    }
}

void Analyze::analysisRepeatUnknow()
{
    int *prev_all = new int [me.getRank()];
    for (int i = 0; i < me.getRank(); i++)
        prev_all[i] = i;
    MPI_Group_incl(group_comm_world, me.getRank(), prev_all, &group_prev_all);
    MPI_Win_create(repeat_alien, repeat_alien_size * sizeof(Repeat), sizeof(Repeat), MPI_INFO_NULL, MPI_COMM_WORLD, &win_repeat_alien);
    MPI_Win_post(group_prev_all, 0, win_repeat_alien);

    int *next_all = new int [me.getSize() - me.getRank() - 1];
    for (int i = 0; i < me.getSize() - me.getRank() - 1; i++)
        next_all[i] = me.getRank() + 1;
    MPI_Group_incl(group_comm_world, me.getSize() - me.getRank() - 1, next_all, &group_next_all);
    MPI_Win_start(group_next_all, 0, win_repeat_alien);
    if (!me.isLast()) {
        ;
    } else  {
        ;
    }
    MPI_Win_complete(win_repeat_alien);

}

Repeat Analyze::requestRepeat(int rank, ulong x)
{
    /*
    if (x > matrix.width)
        return Repeat(0,0,0,0,0);
    ulong flag;
    MPI_Get(&flag, 1, MPI::BOOL, rank + 1, x, 1, MPI::BOOL, win_flag_begin);
    if (!flag)
        return Repeat(0,0,0,0,0);
    Repeat rep;
    MPI_Get(&rep, 5, MPI_LONG, rank + 1, repeat_alien[flag].x_begin, 5, MPI_LONG, win_repeat_alien);
    if (rep.length > 0)
        return rep;
    Repeat rep_1 = requestRepeat(rank + 1, rep.x_end + 1);
    if (rep_1.length > 0)
        return Repeat(rep.x_begin, rep.y_begin, rep_1.x_end, rep_1.y_end, rep.length + rep_1.length);
    else
        rep.length = -rep.length;
    return rep;
    // repeat_unknow.clear();
    */
    return Repeat();
}

void Analyze::sortRepeatAnswer()
{
}

double Analyze::getEps()
{
    return eps;
}

ulong  Analyze::getMinLengthRepeat()
{
    return min_length;
}

double Analyze::getFidelityRepeat()
{
    return fidelity_repeat;
}

size_t Analyze::getLimitMemoryMatrix()
{
    return limit_memory;
}


void Analyze::setEps(double eps_new)
{
    eps = eps_new ;
}

void Analyze::setMinLengthRepeat(ulong min_length_new)
{
    min_length = min_length_new ;
}

void Analyze::setFidelityRepeat(double fidelity_repeat_new)
{
    fidelity_repeat = fidelity_repeat_new ;
}

void Analyze::setLimitMemoryMatrix(size_t limit_memory_new)
{
    limit_memory = limit_memory_new ;
}


#include "analyze.h"

Repeat::Repeat()
{
    x_begin = y_begin = x_end = y_end = length = 0;
}

Repeat::Repeat(long x1, long y1, long x2, long y2, long len)
{
    x_begin = x1;
    y_begin = y1;
    x_end   = x2;
    y_end   = y2;
    length  = len;
}

void Repeat::Print()
{
    printf("%lu %lu %lu %lu %lu\n", x_begin, y_begin, x_end, y_end, length);
}

Analyze::Analyze(MyMPI new_me)
    : me(new_me), matrix(me), result_matrix(me)
{
}

Analyze::~Analyze()
{
}


MatrixAnalysis Analyze::doAnalyze(MatrixGomology matrixGomology, ulong nlength, bool gpu_mode)
{
    matrix = matrixGomology;
    length = nlength;
    use_gpu = gpu_mode;

    ulong *offset, *sum_offset;
    matrix.offsetLength(offset, sum_offset, &matrix.height);
    my_global_height = sum_offset[me.getRank()];
    delete [] offset;
    delete [] sum_offset;

    MPI_Comm_group(MPI_COMM_WORLD, &group_comm_world);
    MPI_Type_get_extent(MPI::BOOL, &lb, &extent);
    me.allMessage("!!!\n");

    initFlagEnd();
    /*
    if (use_gpu)
        localSearch1_n_GPU();
    */
    localSearch0();
    initFlagEndPrev();
    if (!use_gpu)
        localSearch1_n();
    analysisRepeatBegin();
    analysisRepeatAlien();

    MPI_Win_free(&win_flag_end);
    MPI_Win_free(&win_flag_begin);
    MPI_Win_free(&win_repeat_alien);

    list<struct Repeat>::iterator iter = repeat_answer.begin();
    for (int i = 0; i < repeat_answer.size(); i++) {
        iter->Print();
        iter++;
    }
    return result_matrix;
}

void Analyze::initFlagEnd()
{
    if (!me.isLast()) {
        flag_end = new bool [matrix.width];
        for (int x = 0; x < matrix.width; x++)
            flag_end[x] = searchRepeat(matrix.height - 1, x);
    }

    MPI_Aint size_window = me.isLast() ? 0 : matrix.width * extent;
    MPI_Win_create(flag_end, size_window, extent, MPI_INFO_NULL, MPI_COMM_WORLD, &win_flag_end);

    if (!me.isLast()) {
        int next = me.getRank() + 1;
        MPI_Group_incl(group_comm_world, 1, &next, &group_next);
        MPI_Win_post(group_next, 0, win_flag_end);
    }
}

void Analyze::initFlagEndPrev()
{
    if (!me.isFirst()) {
        flag_end_prev = new bool [repeat_begin.size()];
        int prev = me.getRank() - 1;
        MPI_Group_incl(group_comm_world, 1, &prev, &group_prev);
        MPI_Win_start(group_prev, 0, win_flag_end);
        list<struct Repeat>::iterator iter = repeat_begin.begin();
        for (int i = 0; i < repeat_begin.size(); i++) {
            MPI_Get(&flag_end_prev[i], 1, MPI::BOOL, prev, iter->x_begin - 1, 1, MPI::BOOL, win_flag_end);
            iter++;
        }
    }
}

void Analyze::localSearch0()
{
    for (ulong x = 0; x < matrix.width; x++)
        if (matrix.data[x]) {
            Repeat rep = findRepeat(0, x);
            repeat_begin.push_back(rep);
        }
}

void Analyze::localSearch1_n()
{
    for (ulong y = 0; y < matrix.height; y++)
        for (ulong x = 0; x < matrix.width; x++)
            if (matrix.data[y * matrix.width + x]) {
                Repeat rep = findRepeat(y, x);
                if (rep.length > length)
                    repeat_answer.push_back(rep);
                else if (rep.length < 0)
                    repeat_unknow.push_back(rep);
                else
                    continue;
            }
}

Repeat Analyze::findRepeat(ulong y, ulong x)
{
    ulong x_end = x;
    ulong y_end = y;
    ulong length_rep = 0;
    while (matrix.data[y_end * matrix.width + x_end] && x_end < matrix.width
        && y_end < matrix.height) {
        matrix.data[y_end * matrix.width + x_end] = false;
        x_end++;
        y_end++;
        length_rep++;
    }
    if (y_end == matrix.height)
        length_rep = -length_rep;
    return Repeat(x, my_global_height + y, x_end, my_global_height + y_end, length_rep);
}

void Analyze::analysisRepeatBegin()
{
    if (!me.isFirst())
        MPI_Win_complete(win_flag_end);
    list<struct Repeat>::iterator i = repeat_begin.begin();
    list<struct Repeat>::iterator j = repeat_begin.end();
    int k = 0;
    while (i != j) {
        if ((!me.isFirst() && !flag_end_prev[k] || me.isFirst()) && i->length > length) {
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

    if (repeat_begin.size() > 0) {
        repeat_alien = new struct Repeat [repeat_begin.size()];
        repeat_alien_size = repeat_begin.size();
        list<struct Repeat>::iterator m = repeat_begin.begin();
        if (me.isLast()) {
            for (int l = 0; l < repeat_begin.size(); l++) {
                if (m->length < 0);
                    m->length = -m->length;
                m = repeat_begin.begin();
            }
        }
    }
    repeat_begin.clear();
    delete [] flag_end_prev;
    //MPI_Group_free(&group_prev);
}

void Analyze::analysisRepeatAlien()
{
    flag_begin = new long [matrix.width];
    for (int i = 0; i < matrix.width; i++)
        flag_begin[i] = -1;

    for (int i = 0; i < repeat_alien_size; i++)
        flag_begin[repeat_alien[i].x_begin] = i;

    int *prev_all = new int [me.getRank()];
    for (int i = 0; i < me.getRank(); i++)
        prev_all[i] = i;
    MPI_Group_incl(group_comm_world, me.getRank(), prev_all, &group_prev_all);
    MPI_Win_create(flag_begin, matrix.width, sizeof(long), MPI_INFO_NULL, MPI_COMM_WORLD, &win_flag_begin);
    MPI_Win_post(group_prev_all, 0, win_flag_begin);
    MPI_Win_create(repeat_alien, repeat_alien_size, sizeof(Repeat), MPI_INFO_NULL, MPI_COMM_WORLD, &win_repeat_alien);
    MPI_Win_post(group_prev_all, 0, win_repeat_alien);
}

void Analyze::analysisRepeatUnknow()
{
    int *next_all = new int [me.getSize() - me.getRank() - 1];
    for (int i = 0; i < me.getSize() - me.getRank() - 1; i++)
        next_all[i] = me.getRank() + 1;
    MPI_Group_incl(group_comm_world, me.getSize() - me.getRank() - 1, next_all, &group_next_all);
    MPI_Win_start(group_next_all, 0, win_flag_begin);
    MPI_Win_start(group_next_all, 0, win_repeat_alien);
    if (!me.isLast()) {
        ;
    } else  {
        ;
    }
    MPI_Win_complete(win_flag_begin);
    MPI_Win_complete(win_repeat_alien);

    MPI_Win_wait(win_flag_end);
    MPI_Win_wait(win_flag_begin);
    MPI_Win_wait(win_repeat_alien);

    list<struct Repeat>::iterator i = repeat_begin.begin();
    list<struct Repeat>::iterator j = repeat_begin.end();
    while (i != j) {
    }
}

Repeat Analyze::requestRepeat(int rank, ulong x)
{
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
}

void Analyze::formResultMatrix()
{
}

bool Analyze::searchRepeat(ulong y, ulong x)
{
    return matrix.data[y * matrix.width + x];
}

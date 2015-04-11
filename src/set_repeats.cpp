#include "set_repeats.h"

Repeat::Repeat(ulong x_bn, ulong y_bn, ulong x_en, ulong y_en, ulong len)
    : x_begin(x_bn), y_begin(y_bn), x_end(x_en), y_end(y_en), length(len)
{
}

void Repeat::Print()
{
    printf("%ld %ld %ld %ld %ld\n", x_begin, y_begin, x_end, y_end, length);
}


SetRepeats::SetRepeats(MyMPI me)
    : DataMPI<TypeAnalysis, ulong>(me, "SetRepeats"),
    x_limit_left(0), x_limit_right(0), y_limit_above(0), y_limit_bottom(0)
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
    for (int i = 0; i < vec.size(); i++)
        vec[i].Print();
}

void SetRepeats::writeUsually(char *file_name)
{
}

void SetRepeats::writeMy(char *file_name)
{
}


void SetRepeats::analyzeOtherProcess()
{
    int de = 1;
    TypeAnalysis setRepeatsBastard, setRepeatsNotFinish, setRepeatsResult;
    for (int i = 0; i < vec.size(); i++)
        if (vec[i].y_begin == y_limit_above && vec[i].x_begin != x_limit_left && !me.isFirst())
            setRepeatsBastard.push_back(vec[i]);
        else if (vec[i].y_end == y_limit_bottom && vec[i].x_end != x_limit_right && !me.isLast())
            setRepeatsNotFinish.push_back(vec[i]);
        else
            setRepeatsResult.push_back(vec[i]);
    vec.clear();

    MPI_Aint lb, extent_bool, extent_ulong;
    MPI_Type_get_extent(MPI::BOOL, &lb, &extent_bool);
    MPI_Type_get_extent(MPI_UNSIGNED_LONG, &lb, &extent_ulong);
    MPI_Group group_comm_world, group_prev, group_next;
    MPI_Comm_group(MPI_COMM_WORLD, &group_comm_world);

    bool *a = 0;
    if (!me.isLast()) {
        a = new bool [x_limit_right - x_limit_left + 1];
        memset(a, 0, (x_limit_right - x_limit_left + 1) * sizeof(bool));
        for (int i = 0; i < setRepeatsNotFinish.size(); i++)
            a[setRepeatsNotFinish[i].x_end] = true;
        for (int i = 0; i < setRepeatsBastard.size(); i++)
            if (setRepeatsBastard[i].y_end == y_limit_bottom)
                a[setRepeatsBastard[i].x_end] = true;
    }
    MPI_Win win_a;
    MPI_Win_create(a, (x_limit_right - x_limit_left + 1) * extent_bool,
                        extent_bool, MPI_INFO_NULL, MPI_COMM_WORLD, &win_a);

    if (!me.isLast()) {
        int next = me.getRank() + 1;
        MPI_Group_incl(group_comm_world, 1, &next, &group_next);
        MPI_Win_post(group_next, 0, win_a);
    }
    if (!me.isFirst()) {
        int prev = me.getRank() - 1;
        MPI_Group_incl(group_comm_world, 1, &prev, &group_prev);
        MPI_Win_start(group_prev, 0, win_a);
        bool *a_prev = 0;
        a_prev = new bool [setRepeatsBastard.size()];
        for (int i = 0; i < setRepeatsBastard.size(); i++)
            MPI_Get(&a_prev[i], 1, MPI::BOOL, me.getRank() - 1, setRepeatsBastard[i].x_begin - 1, 1, MPI::BOOL, win_a);
        // HERE MAY BE SOME WORK.
        MPI_Win_complete(win_a);
        vector<struct Repeat> ::iterator iter = setRepeatsBastard.begin();
        vector<struct Repeat> ::iterator iter_e = setRepeatsBastard.end();
        for (int t = 0; iter != iter_e; iter++, t++) {
            if (!a_prev[t]) {
                if (iter->y_end != y_limit_bottom || me.isLast())
                    setRepeatsResult.push_back(*iter);
                else
                    setRepeatsNotFinish.push_back(*iter);
                setRepeatsBastard.erase(iter);
            }
        }
        MPI_Group_free(&group_prev);
        delete [] a_prev;
    }

    ulong *b = 0;
    if (!me.isFirst()) {
        b = new ulong [6 * (x_limit_right - x_limit_left + 1)];
        memset(b, 0, 6 * (x_limit_right - x_limit_left + 1) * sizeof(ulong));
        for (int i = 0; i < setRepeatsBastard.size(); i++) {
            b[setRepeatsBastard[i].x_begin * 6 + 0] = setRepeatsBastard[i].x_begin;
            b[setRepeatsBastard[i].x_begin * 6 + 1] = setRepeatsBastard[i].y_begin;
            b[setRepeatsBastard[i].x_begin * 6 + 2] = setRepeatsBastard[i].x_end;
            b[setRepeatsBastard[i].x_begin * 6 + 3] = setRepeatsBastard[i].y_end;
            b[setRepeatsBastard[i].x_begin * 6 + 4] = setRepeatsBastard[i].length;
            if (setRepeatsBastard[i].y_end == y_limit_bottom && !me.isLast())
                b[setRepeatsBastard[i].x_begin * 6 + 5] = 1;
        }
    }
    MPI_Win win_b;
    MPI_Win_create(b, (x_limit_right - x_limit_left) * 6 * extent_ulong,
                        extent_ulong, MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);

    for (int next = me.getRank() + 1, prev = me.getRank() - 1; next < me.getSize() || prev >= 0; next++, prev--) {
        MPI_Group group_next_proc, group_prev_proc;
        if (prev >= 0) {
            MPI_Group_incl(group_comm_world, 1, &prev, &group_prev_proc);
            MPI_Win_post(group_prev_proc, 0, win_b);
        }

        if (next < me.getSize()) {
            MPI_Group_incl(group_comm_world, 1, &next, &group_next_proc);
            MPI_Win_start(group_next_proc, 0, win_b);
            ulong *b_next = 0;
            if (!setRepeatsNotFinish.empty()) {
                b_next = new ulong [6 * setRepeatsNotFinish.size()];
                for (int i = 0; i < setRepeatsNotFinish.size(); i++) {
                    MPI_Get(&b_next[6 * i], 6, MPI_UNSIGNED_LONG, next,
                        6 * (setRepeatsNotFinish[i].x_end + 1), 6, MPI_UNSIGNED_LONG, win_b);
                }
            }
            MPI_Win_complete(win_b);
            if (!setRepeatsNotFinish.empty()) {
                vector<struct Repeat> ::iterator iter = setRepeatsNotFinish.begin();
                vector<struct Repeat> ::iterator iter_e = setRepeatsNotFinish.end();
                for (int i = 0; iter != iter_e; iter++, i++) {
                    if (b_next[6 * i]) {
                        iter->x_end = b_next[6 * i + 2];
                        iter->y_end = b_next[6 * i + 3];
                        iter->length += b_next[6 * i + 4];
                    }
                    if (!b_next[6 * i] || !b_next[6 * i + 5]) {
                        setRepeatsResult.push_back(*iter);
                        setRepeatsNotFinish.erase(iter);
                    }
                }
            }
            MPI_Group_free(&group_next_proc);
            delete [] b_next;
        }

        if (prev >= 0) {
            MPI_Win_wait(win_b);
            MPI_Group_free(&group_prev_proc);
        }
    }

    if (!me.isLast()) {
        MPI_Win_wait(win_a);
        MPI_Group_free(&group_next);
    }

    MPI_Win_free(&win_a);
    MPI_Win_free(&win_b);
    MPI_Group_free(&group_comm_world);

    delete [] a;
    delete [] b;

    vec = setRepeatsResult;
    setRepeatsResult.clear();
    setRepeatsNotFinish.clear();
    setRepeatsBastard.clear();
}

void SetRepeats::debugInfo(const char *file, int line, const char *info)
{
    this->me.rootMessage("\n");
    this->me.rootMessage("This is debugInfo(%s) of %s in %s at line %d\n", info, this->class_name, file, line);
    this->me.allMessage("x_limit_left = %9ld x_limit_right = %9ld y_limit_above = %9ld y_limit_bottom = %9ld\n",
        x_limit_left, x_limit_right, y_limit_above, y_limit_bottom);
    this->me.rootMessage("\n");
}

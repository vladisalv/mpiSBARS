#include "list_repeats.h"

#include <sstream>
#include <string>

Repeat::Repeat(ulong x_bn, ulong y_bn, ulong x_en, ulong y_en, ulong len)
    : x_begin(x_bn), y_begin(y_bn), x_end(x_en), y_end(y_en), length(len)
{
}

void Repeat::Print()
{
    printf("%ld %ld %ld %ld %ld\n", x_begin, y_begin, x_end, y_end, length);
}


ListRepeats::ListRepeats(MyMPI me)
    : DataMPI<TypeAnalysis>(me),
    x_limit_left(0), x_limit_right(0), y_limit_above(0), y_limit_bottom(0)
{
    data = new TypeAnalysis;
}

ListRepeats::~ListRepeats()
{
}

void ListRepeats::readMPI(char *file_name)
{
}

void ListRepeats::readUsually(char *file_name)
{
}

void ListRepeats::readMy(char *file_name)
{
}

void ListRepeats::writeRepeat(TypeAnalysis list)
{
    string str;
    stringstream stm;
    stm << endl;
    stm << "proc" << this->me.getRank() << endl;
    for (TypeAnalysis::iterator list_iter = list.begin(); list_iter != list.end(); list_iter++)
        stm << list_iter->x_begin << " " << list_iter->y_begin << " " << list_iter->x_end << " " << list_iter->y_end << " " << list_iter->length << endl;
    stm << endl;
    me.allMessage("%s", stm.str().c_str());
}

void ListRepeats::sort()
{
    for (TypeAnalysis::iterator i = data->begin(); i != data->end(); i++) {
        for (TypeAnalysis::iterator j = i; j != data->end(); j++) {
            if (i->y_begin > j->y_begin) {
                ulong tmp;
                tmp = i->x_begin;
                i->x_begin = j->x_begin;
                j->x_begin = tmp;
                tmp = i->y_begin;
                i->y_begin = j->y_begin;
                j->y_begin = tmp;
                tmp = i->x_end;
                i->x_end = j->x_end;
                j->x_end = tmp;
                tmp = i->y_end;
                i->y_end = j->y_end;
                j->y_end = tmp;
                tmp = i->length;
                i->length = j->length;
                j->length = tmp;
            } else if (i->y_begin == j->y_begin && i->x_begin > j->x_begin) {
                ulong tmp;
                tmp = i->x_begin;
                i->x_begin = j->x_begin;
                j->x_begin = tmp;
                tmp = i->y_begin;
                i->y_begin = j->y_begin;
                j->y_begin = tmp;
                tmp = i->x_end;
                i->x_end = j->x_end;
                j->x_end = tmp;
                tmp = i->y_end;
                i->y_end = j->y_end;
                j->y_end = tmp;
                tmp = i->length;
                i->length = j->length;
                j->length = tmp;
            }
        }
    }
}

void ListRepeats::writeMPI(char *file_name)
{
    string str;
    stringstream stm;
    stm << "i proc " << me.getRank() << endl;
    sort();
    for (TypeAnalysis::iterator list_iter = data->begin(); list_iter != data->end(); list_iter++)
        stm << list_iter->x_begin << " " << list_iter->y_begin << " " << list_iter->length << endl;
    MPI_File hFile;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &hFile);
    MPI_File_set_view(hFile, 0, MPI_CHAR, MPI_CHAR, (char *)"native", MPI_INFO_NULL);
    for (int i = 0; i < me.getSize(); i++) {
        me.Synchronize();
        if (i == me.getRank()) {
            MPI_File_write_shared(hFile, (void *)stm.str().c_str(), stm.str().size(), MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(hFile);
        }
    }
    MPI_File_close(&hFile);
}

void ListRepeats::writeUsually(char *file_name)
{
}

void ListRepeats::writeMy(char *file_name)
{
}


void ListRepeats::makeOffsetRow(ulong offset)
{
    for (TypeAnalysis::iterator list_iter = data->begin(); list_iter != data->end(); list_iter++) {
        list_iter->y_begin += offset;
        list_iter->y_end   += offset;
    }
    y_limit_above  += offset;
    y_limit_bottom += offset;
}


void ListRepeats::makeOffsetColumn(ulong offset)
{
    for (TypeAnalysis::iterator list_iter = data->begin(); list_iter != data->end(); list_iter++) {
        list_iter->x_begin += offset;
        list_iter->x_end   += offset;
    }
    x_limit_left  += offset;
    x_limit_right += offset;
}


bool compare_ybegin(Repeat r1, Repeat r2)
{
    return (r1.y_begin < r2.y_begin);
}
bool compare_yend(Repeat r1, Repeat r2)
{
    return (r1.y_end < r2.y_end);
}
bool compare_xbegin(Repeat r1, Repeat r2)
{
    return (r1.x_begin < r2.x_begin);
}
bool compare_xend(Repeat r1, Repeat r2)
{
    return (r1.x_end < r2.x_end);
}

void ListRepeats::mergeRepeatsRow(ListRepeats listNext)
{
    TypeAnalysis::iterator list_iter;
    if (data->empty()) {
        for (list_iter = listNext.data->begin(); list_iter != listNext.data->end(); list_iter++)
            data->push_back(*list_iter);
        listNext.data->clear();
        x_limit_left   = listNext.x_limit_left;
        x_limit_right  = listNext.x_limit_right;
        y_limit_bottom = listNext.y_limit_bottom;
        return;
    }

    TypeAnalysis list1, list2;
    for (list_iter = data->begin(); list_iter != data->end();)
        if (list_iter->y_end == y_limit_bottom) {
            list1.push_back(*list_iter);
            list_iter = data->erase(list_iter);
        } else {
            list_iter++;
        }
    for (list_iter = listNext.data->begin(); list_iter != listNext.data->end();)
        if (list_iter->y_begin == listNext.y_limit_above) {
            list2.push_back(*list_iter);
            list_iter = listNext.data->erase(list_iter);
        } else {
            list_iter++;
        }

    for (list_iter = listNext.data->begin(); list_iter != listNext.data->end(); list_iter++)
        data->push_back(*list_iter);
    listNext.data->clear();

    TypeAnalysis::iterator l1, l2;
    for (l1 = list1.begin(); l1 != list1.end(); l1++)
        for (l2 = list2.begin(); l2 != list2.end(); l2++) {
            if (l1->x_end + 1 == l2->x_begin) {
                l1->x_end = l2->x_end;
                l1->y_end = l2->y_end;
                l1->length += l2->length;
                l2 = list2.erase(l2);
                break;
            }
        }

    for (l1 = list1.begin(); l1 != list1.end(); l1++)
        data->push_back(*l1);
    for (l2 = list2.begin(); l2 != list2.end(); l2++)
        data->push_back(*l2);

    list1.clear();
    list2.clear();
    y_limit_bottom = listNext.y_limit_bottom;
}

void ListRepeats::mergeRepeatsColumn(ListRepeats listNext)
{
    TypeAnalysis::iterator list_iter;
    if (data->empty()) {
        for (list_iter = listNext.data->begin(); list_iter != listNext.data->end(); list_iter++)
            data->push_back(*list_iter);
        listNext.data->clear();
        x_limit_right  = listNext.x_limit_right;
        y_limit_above  = listNext.y_limit_above;
        y_limit_bottom = listNext.y_limit_bottom;
        return;
    }

    TypeAnalysis list1, list2;
    for (list_iter = data->begin(); list_iter != data->end();)
        if (list_iter->x_end == x_limit_right) {
            list1.push_back(*list_iter);
            list_iter = data->erase(list_iter);
        } else {
            list_iter++;
        }
    for (list_iter = listNext.data->begin(); list_iter != listNext.data->end();)
        if (list_iter->x_begin == listNext.x_limit_left) {
            list2.push_back(*list_iter);
            list_iter = listNext.data->erase(list_iter);
        } else {
            list_iter++;
        }

    for (list_iter = listNext.data->begin(); list_iter != listNext.data->end(); list_iter++)
        data->push_back(*list_iter);
    listNext.data->clear();

    TypeAnalysis::iterator l1, l2;
    for (l1 = list1.begin(); l1 != list1.end(); l1++)
        for (l2 = list2.begin(); l2 != list2.end(); l2++) {
            if (l1->y_end + 1 == l2->y_begin) {
                l1->x_end = l2->x_end;
                l1->y_end = l2->y_end;
                l1->length += l2->length;
                l2 = list2.erase(l2);
                break;
            }
        }

    for (l1 = list1.begin(); l1 != list1.end(); l1++)
        data->push_back(*l1);
    for (l2 = list2.begin(); l2 != list2.end(); l2++)
        data->push_back(*l2);

    list1.clear();
    list2.clear();
    x_limit_right = listNext.x_limit_right;
}



void ListRepeats::mergeRepeats()
{
    TypeAnalysis listRepeatsBastard, listRepeatsNotFinish, listRepeatsResult;
    for (TypeAnalysis::iterator list_iter = data->begin(); list_iter != data->end(); list_iter++)
        if (list_iter->y_begin == y_limit_above && list_iter->x_begin != x_limit_left && !me.isFirst())
            listRepeatsBastard.push_back(*list_iter);
        else if (list_iter->y_end == y_limit_bottom && list_iter->x_end != x_limit_right && !me.isLast())
            listRepeatsNotFinish.push_back(*list_iter);
        else
            listRepeatsResult.push_back(*list_iter);
    data->clear();

    MPI_Aint lb, extent_bool, extent_ulong;
    MPI_Type_get_extent(MPI::BOOL, &lb, &extent_bool);
    MPI_Type_get_extent(MPI_UNSIGNED_LONG, &lb, &extent_ulong);
    MPI_Group group_comm_world, group_prev, group_next, group_prev_all;
    MPI_Comm_group(MPI_COMM_WORLD, &group_comm_world);

    bool *a = 0;
    if (!me.isLast()) {
        a = new bool [x_limit_right - x_limit_left + 1];
        memset(a, 0, (x_limit_right - x_limit_left + 1) * sizeof(bool));

        for (TypeAnalysis::iterator list_iter = listRepeatsNotFinish.begin();
                list_iter != listRepeatsNotFinish.end(); list_iter++)
            a[list_iter->x_end] = true;

        for (TypeAnalysis::iterator list_iter = listRepeatsBastard.begin();
                list_iter != listRepeatsBastard.end(); list_iter++)
            if (list_iter->y_end == y_limit_bottom)
                a[list_iter->x_end] = true;
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
        a_prev = new bool [listRepeatsBastard.size()];
        TypeAnalysis::iterator list_iter;
        list_iter   = listRepeatsBastard.begin();
        for (int i = 0; list_iter != listRepeatsBastard.end(); i++, list_iter++)
            MPI_Get(&a_prev[i], 1, MPI::BOOL, me.getRank() - 1, list_iter->x_begin - 1, 1, MPI::BOOL, win_a);
        // HERE MAY BE SOME WORK.
        MPI_Win_complete(win_a);
        list_iter   = listRepeatsBastard.begin();
        for (int i = 0; list_iter != listRepeatsBastard.end(); i++) {
            if ((!a_prev[i])) {
                if (list_iter->y_end != y_limit_bottom || me.isLast())
                    listRepeatsResult.push_back(*list_iter);
                else
                    listRepeatsNotFinish.push_back(*list_iter);
                list_iter = listRepeatsBastard.erase(list_iter);
            } else {
                list_iter++;
            }
        }
        MPI_Group_free(&group_prev);
        delete [] a_prev;
    }
    if (!me.isLast()) {
        MPI_Win_wait(win_a);
        MPI_Group_free(&group_next);
    }

    ulong *b = 0;
    if (!me.isFirst()) {
        b = new ulong [6 * (x_limit_right - x_limit_left + 1)];
        memset(b, 0, 6 * (x_limit_right - x_limit_left + 1) * sizeof(ulong));
        for (TypeAnalysis::iterator list_iter = listRepeatsBastard.begin();
                list_iter != listRepeatsBastard.end(); list_iter++) {
            b[list_iter->x_begin * 6 + 0] = list_iter->x_begin;
            b[list_iter->x_begin * 6 + 1] = list_iter->y_begin;
            b[list_iter->x_begin * 6 + 2] = list_iter->x_end;
            b[list_iter->x_begin * 6 + 3] = list_iter->y_end;
            b[list_iter->x_begin * 6 + 4] = list_iter->length;
            if (list_iter->y_end == y_limit_bottom && !me.isLast())
                b[list_iter->x_begin * 6 + 5] = 1;
        }
    }
    MPI_Win win_b;
    MPI_Win_create(b, (x_limit_right - x_limit_left) * 6 * extent_ulong,
                        extent_ulong, MPI_INFO_NULL, MPI_COMM_WORLD, &win_b);

    //printf("i %d board: x_left=%ld x_right=%ld y_above=%ld y_bottom=%ld\n", me.getRank(), x_limit_left, x_limit_right, y_limit_above, y_limit_bottom);
    for (int next = 1; next < me.getSize(); next++) {
        MPI_Win_fence(0, win_b);
        ulong *b_next = 0;
        if (next + me.getRank() < me.getSize()) {
            if (!listRepeatsNotFinish.empty()) {
                b_next = new ulong [6 * listRepeatsNotFinish.size()];
                int i = 0;
                //printf("i %d with %d size %d\n", me.getRank(), next + me.getRank(), listRepeatsNotFinish.size());
                for (TypeAnalysis::iterator list_iter = listRepeatsNotFinish.begin();
                        list_iter != listRepeatsNotFinish.end(); list_iter++, i++) {
                    //printf("i %d with %d ==  ==  == %d\n", me.getRank(), next + me.getRank(), list_iter->x_end);
                    MPI_Get(&b_next[6 * i], 6, MPI_UNSIGNED_LONG, next + me.getRank(),
                        6 * (list_iter->x_end + 1), 6, MPI_UNSIGNED_LONG, win_b);
                }
            }
        }
        MPI_Win_fence(0, win_b);
        if (next + me.getRank() < me.getSize()) {
            if (!listRepeatsNotFinish.empty()) {
                TypeAnalysis::iterator list_iter = listRepeatsNotFinish.begin();
                for (int i = 0; list_iter != listRepeatsNotFinish.end(); i++) {
                    if (b_next[6 * i]) {
                        list_iter->x_end   = b_next[6 * i + 2];
                        list_iter->y_end   = b_next[6 * i + 3];
                        list_iter->length += b_next[6 * i + 4];
                    }
                    if (!b_next[6 * i] || !b_next[6 * i + 5]) {
                        listRepeatsResult.push_back(*list_iter);
                        list_iter = listRepeatsNotFinish.erase(list_iter);
                    } else {
                        list_iter++;
                    }
                }
            }
        }
        delete [] b_next;
    }

    MPI_Win_free(&win_a);
    MPI_Win_free(&win_b);
    MPI_Group_free(&group_comm_world);

    delete [] a;
    delete [] b;

    *data = listRepeatsResult;
    listRepeatsResult.clear();
    listRepeatsNotFinish.clear();
    listRepeatsBastard.clear();
}

void ListRepeats::convertToOriginalRepeats(uint window_profiling,
        uint window_decompose, uint step_decompose, uint number_coef)
{
    for (TypeAnalysis::iterator list_iter = data->begin(); list_iter != data->end(); list_iter++) {
        list_iter->x_begin = list_iter->x_begin * step_decompose;
        list_iter->y_begin = list_iter->y_begin * step_decompose;
        list_iter->length  = list_iter->length * step_decompose + window_decompose + window_profiling - 1;
        list_iter->x_end   = list_iter->x_begin + list_iter->length;
        list_iter->y_end   = list_iter->y_begin + list_iter->length;
    }
}

void ListRepeats::debugInfo(const char *file, int line, const char *info)
{
    this->me.rootMessage("\n");
    this->me.rootMessage("This is debugInfo(%s) in %s at line %d\n", info, file, line);
    this->me.allMessage("x_limit_left = %9ld x_limit_right = %9ld y_limit_above = %9ld y_limit_bottom = %9ld\n",
        x_limit_left, x_limit_right, y_limit_above, y_limit_bottom);
    this->me.rootMessage("\n");
}

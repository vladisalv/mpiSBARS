#include "profiling.h"

Profiling::Profiling(MyMPI new_me, uint new_window, uint new_step)
    : me(new_me), window(new_window), step(new_step)
{
}

Profiling::~Profiling()
{
}

Profile Profiling::doProfileOld(Sequence &seq, char ch1, char ch2)
{
    uint modulo1 = seq.offset % step; // modulo before you
    ulong length_send_message = modulo1 ? window - modulo1 : window - step;
    MPI_Request req_send;
    if (!me.isFirst())
        me.iSend(seq.data, length_send_message, MPI_CHAR,
                me.getRank() - 1, 0, &req_send);

    uint begin = modulo1 ? step - modulo1 : 0;
    ulong work_length_seq = seq.length - begin;
    uint modulo2 = work_length_seq % step; // modulo after you
    ulong length_recv_message = modulo2 ? window - modulo2 : window - step;

    uint number_all_window, number_my_window, number_another_window;
    if (me.isLast()) {
        number_all_window = (work_length_seq - window + step - 1) / step;
        number_my_window = number_all_window;
        number_another_window = 0;
    } else {
        number_all_window = (work_length_seq + step - 1) / step;
        number_my_window  = (work_length_seq - window + step - 1) / step;
        number_another_window = number_all_window - number_my_window;
    }

    ulong length_your_elem = modulo2 ? (number_another_window - 1) * step + modulo2
                                     :  number_another_window * step;

    MPI_Request req_recv;
    TypeSequence *buf_recv;
    if (!me.isLast()) {
        buf_recv = new TypeSequence [length_your_elem + length_recv_message];
        me.iRecv(&buf_recv[length_your_elem], length_recv_message, MPI_CHAR,
                    me.getRank() + 1, 0, &req_recv);
    }

    Profile profile(me);
    profile.offset = (seq.offset + step - 1) / step;
    profile.length = number_all_window;
    profile.data   = new TypeProfile [profile.length];

    me.allMessage("%d \
    module1 = %d\n \
    module2 = %d\n \
    begin = %d\n \
    profile.offset = %d\n \
    profile.length = %d\n \
    work_length_seq = %d\n \
    length_send_message = %d\n \
    length_recv_message = %d\n \
    number_all_window = %d\n \
    number_my_window = %d\n \
    number_another_window = %d\n \
    length_your_elem = %d\n",
    me.getRank(), modulo1, modulo2, begin, profile.offset, profile.length, work_length_seq,
    length_send_message, length_recv_message, number_all_window, number_my_window, number_another_window, length_your_elem);

    TypeProfile count = 0;
    for (uint i = 0; i < window; i++)
        if (seq.data[begin + i] == ch1 || seq.data[begin + i] == ch2)
            count++;
    profile.data[0] = count;

    ulong offset = begin;
    for (int i = 1; i < number_my_window; i++) {
        for (int j = 0; j < step; j++, offset++) {
            if (seq.data[offset] == ch1 || seq.data[offset] == ch2 && count > 0)
                count--;
            if (seq.data[offset + window] == ch1 || seq.data[offset + window] == ch2)
                count++;
        }
        profile.data[i] = count;
    }

    if (!me.isLast()) {
        for (int i = 0; i < length_your_elem; i++)
            buf_recv[i] = seq.data[seq.length - length_your_elem + i];

        me.wait(&req_recv, MPI_STATUS_IGNORE);

        TypeProfile count = 0;
        for (uint i = 0; i < window; i++)
            if (buf_recv[i] == ch1 || buf_recv[i] == ch2)
                count++;
        profile.data[number_my_window] = count;

        for (int i = 1, offset = 0; i < number_another_window; i++) {
            for (int j = 0; j < step; j++, offset++) {
                if (buf_recv[offset] == ch1 || buf_recv[offset] == ch2 && count > 0)
                    count--;
                if (buf_recv[offset + window] == ch1 || buf_recv[offset + window] == ch2)
                    count++;
            }
            profile.data[number_my_window + i] = count;
        }
    }

    if (!me.isFirst())
        me.wait(&req_send, MPI_STATUS_IGNORE); // first wait message to last
    return profile;
}

Profile Profiling::doProfile(Sequence &seq, char ch1, char ch2)
{
    // @TODO: error, when you don't have data in seq_letter
    me.rootMessage("step_profiling = %ld\n", step);
    Profile profile(me);
    profile.offset = seq.offset;
    ulong length_without_exchange;
    char *buffer = 0;

    if (me.isLast()) {
        profile.length = seq.length - window + 1;
        length_without_exchange = profile.length;
    } else {
        profile.length = seq.length;
        length_without_exchange = profile.length - window + 1;
        buffer = new char [window - 1];
    }

    MPI_Request req_send, req_recv;
    if (!me.isFirst() && !me.isSingle())
        me.iSend(seq.data, window - 1, MPI_CHAR, me.getRank() - 1, 0, &req_send);
    if (!me.isLast() && !me.isSingle())
        me.iRecv(buffer, window - 1, MPI_CHAR, me.getRank() + 1, 0, &req_recv);

    me.allMessage("%d \
    profile.offset = %ld\n \
    profile.length = %ld\n \
    length_without_exchange = %ld\n \
    ", me.getRank(), profile.offset, profile.length, length_without_exchange);

    profile.data = new TypeProfile [profile.length];
    TypeProfile count = 0;
    for (uint i = 0; i < window; i++)
        if (seq.data[i] == ch1 || seq.data[i] == ch2)
            count++;
    profile.data[0] = count;

    for (uint i = 1; i < length_without_exchange; i++) {
        if ((seq.data[i - 1] == ch1 || seq.data[i - 1] == ch2) && count > 0)
            count--;
        if (seq.data[i+window-1] == ch1 || seq.data[i+window-1] == ch2)
            count++;
        profile.data[i] = count;
    }

    if (!me.isLast() && !me.isSingle()) {
        MPI_Status status;
        me.wait(&req_recv, &status);

        for (uint i = 0; i < window - 1; i++) {
            if ((seq.data[length_without_exchange + i - 1] == ch1 ||
                seq.data[length_without_exchange + i - 1] == ch2) && count > 0)
                count--;
            if (buffer[i] == ch1 || buffer[i] == ch2)
                count++;
            profile.data[length_without_exchange + i] = count;
        }
    }

    if (!me.isFirst() && !me.isSingle()) {
        MPI_Status status;
        me.wait(&req_send, &status);
    }
    return profile;
}

void Profiling::setLengthWindowProfile(uint new_window)
{
    window = new_window;
}

uint Profiling::getLengthWindowProfile()
{
    return window;
}

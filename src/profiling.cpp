#include "profiling.h"

Profiling::Profiling(MyMPI new_me, uint new_window)
    : me(new_me), window(new_window)
{
}

Profiling::~Profiling()
{
}

Profile Profiling::doProfile(Sequence &seq, char ch1, char ch2)
{
    // @TODO: error, when you don't have data in seq_letter
    Profile profile(me);
    ulong length_without_exchange;
    char *buffer = 0;

    if (me.isLast()) {
        profile.length= seq.length - window + 1;
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

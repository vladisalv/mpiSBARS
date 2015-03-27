#include "decompose.h"

Decompose::Decompose(MyMPI new_me, GpuComputing new_gpu, uint window_new, uint step_new, uint number_coef_new)
    : me(new_me), gpu(new_gpu), window(window_new), step(step_new), number_coef(number_coef_new)
{
}

Decompose::~Decompose()
{
}


Decomposition Decompose::doDecompose(Profile &profile)
{
    ulong length_other; // for last process. He must know length another process
    if (me.isRoot()) {
        MPI_Request req_len;
        me.iSend(&profile.length, 1, MPI_LONG, me.whoLast(), 1, &req_len);
    } else  if (me.isLast()) {
        MPI_Request req_len;
        MPI_Status status_len;
        me.iRecv(&length_other, 1, MPI_LONG, me.whoRoot(), 1, &req_len);
        me.wait(&req_len, &status_len);
    }

    uint modulo1; // modulo before you
    if (me.isLast())
        modulo1 = (length_other * me.getRank()) % step;
    else
        modulo1 = (profile.length * me.getRank()) % step;

    ulong length_send_message;
    if (modulo1)
        length_send_message = window - modulo1;
    else
        length_send_message = window - step;

    MPI_Request req_send;
    if (!me.isFirst())
        me.iSend(profile.data, length_send_message, MPI_INT, me.getRank() - 1,
                    0, &req_send);

    uint begin;
    if (modulo1)
        begin = step - modulo1;
    else
        begin = 0;

    ulong work_length_profile = profile.length - begin;
    uint modulo2 = work_length_profile % step; // modulo after you

    ulong length_recv_message;
    if (modulo2)
        length_recv_message = window - modulo2;
    else
        length_recv_message = window - step;

    uint number_all_window, number_my_window, number_another_window;
    if (me.isLast()) {
        number_all_window = ceil((double)(work_length_profile - window) / (double)step);
        number_my_window = number_all_window;
        number_another_window = 0;
    } else {
        number_all_window = ceil((double)work_length_profile / (double)step);
        number_my_window = ceil((double)(work_length_profile - window) / (double)step);
        number_another_window = number_all_window - number_my_window;
    }

    ulong length_your_elem;
    if (modulo2)
        length_your_elem = (number_another_window - 1) * step + modulo2;
    else
        length_your_elem = number_another_window * step;

    TypeProfile *buf_recv;
    if (!me.isLast())
        buf_recv = new TypeProfile [length_your_elem + length_recv_message];

    MPI_Request req_recv;
    if (!me.isLast())
        me.iRecv(&buf_recv[length_your_elem], length_recv_message, MPI_INT,
                    me.getRank() + 1, 0, &req_recv);

    Decomposition decomposition(me);
    decomposition.height = number_all_window;
    decomposition.width = number_coef;
    decomposition.length = number_all_window * number_coef;
    decomposition.data = new TypeDecomposition [decomposition.length];

    // do decompose with my profile
    if (gpu.isUse()) {
        gpu.doDecomposeGPU(decomposition.data, number_my_window, number_coef,
                        profile.data, window, step);
    } else {
        for (uint i = 0; i < number_my_window; i++)
            decomposeFourier(&decomposition.data[i * number_coef], number_coef,
                                &profile.data[begin + i * step], window);
    }

    if (!me.isLast()) {
        // copy to buffer your elem
        for (uint i = 0; i < length_your_elem; i++)
            buf_recv[i] = profile.data[profile.length - length_your_elem + i];

        MPI_Status status_recv;
        me.wait(&req_recv, &status_recv);

        // do decompose with buf
        // TODO: use here only HOST. gpu work async
        if (gpu.isUse()) {
            gpu.doDecomposeGPU(&decomposition.data[number_coef * number_my_window],
                        number_another_window, number_coef, buf_recv, window, step);
        } else {
            for (uint i = 0; i < number_another_window; i++)
                decomposeFourier(&decomposition.data[(i + number_my_window) * number_coef],
                                    number_coef, &buf_recv[i * step], window);
        }
    }

    if (!me.isFirst()) {
        MPI_Status status_send;
        me.wait(&req_send, &status_send);
    }
    return decomposition;
}

void Decompose::decomposeFourier(TypeDecomposition *u, uint m, TypeProfile *y, uint k)
{
    TypeDecomposition costi, yi;
    TypeDecomposition p1, p2, p3;
    TypeDecomposition q1, q2, q3;
    TypeDecomposition ti;
    TypeDecomposition c = M_PI / k;
    double sqrt2  =  sqrt(2.0);

    for (uint j = 0; j < m; j++)
        u[j] = 0.0;

    for (uint i = 0; i < k; i++) {
        ti = c * (2 * i + 1 - k);
        costi = 2 * cos(ti);
        yi = (TypeDecomposition)y[i];

        p1 = 0;
        p2 = yi * sin(ti);
        u[0] += yi / sqrt2;
        q2 = yi * costi / 2.0;
        q1 = yi;
        for (uint j = 1; j < m; j++) {
            if (j & 1) {
                p3 = p2;
                p2 = p1;
                p1 = costi * p2 - p3;
                u[j] += p1;
            } else {
                q3 = q2;
                q2 = q1;
                q1 = costi * q2 - q3;
                u[j] += q1;
            }
        }
    }

    for (uint j = 0; j < m; j++)
        u[j] /= k / 2;

}

uint Decompose::getLengthWindowDecompose()
{
    return window;
}

uint Decompose::getStepDecompose()
{
    return step;
}

uint Decompose::getNumberCoefDecompose()
{
    return number_coef;
}

void Decompose::setLengthWindowDecompose(uint window_new)
{
    window = window_new;
}

void Decompose::setStepDecompose(uint step_new)
{
    step = step_new;
}

void Decompose::setNumberCoefDecompose(uint number_coef_new)
{
    number_coef = number_coef_new;
}

#include "decompose.h"
#include <math.h> // ceil()

Decompose::Decompose(MyMPI new_me, GpuComputing new_gpu, uint window_new, uint step_new, uint number_coef_new)
    : me(new_me), gpu(new_gpu), window(window_new), step(step_new), number_coef(number_coef_new)
{
}

Decompose::~Decompose()
{
}


Decomposition Decompose::doDecompose(Profile &profile)
{
    uint modulo1 = profile.offset % step; // modulo before you
    ulong length_send_message = modulo1 ? window - modulo1 : window - step;
    MPI_Request req_send;
    if (!me.isFirst())
        // MPI_INT -> MPI_TYPE_PROFILE
        me.iSend(profile.data, length_send_message, MPI_INT,
                me.getRank() - 1, 0, &req_send);

    uint begin = modulo1 ? step - modulo1 : 0;
    ulong work_length_profile = profile.length - begin;
    uint modulo2 = work_length_profile % step; // modulo after you
    ulong length_recv_message = modulo2 ? window - modulo2 : window - step;

    uint number_all_window, number_my_window, number_another_window;
    if (me.isLast()) {
        //number_all_window = (work_length_profile - window + step - 1) / step;
        number_all_window = (work_length_profile - window + 1 + step - 1) / step;
        number_my_window = number_all_window;
        number_another_window = 0;
    } else {
        number_all_window = (work_length_profile + step - 1) / step;
        //number_my_window  = (work_length_profile - window + step - 1) / step;
        number_my_window  = (work_length_profile - window + 1 + step - 1) / step;
        number_another_window = number_all_window - number_my_window;
    }

    ulong length_your_elem = modulo2 ? (number_another_window - 1) * step + modulo2
                                     :  number_another_window * step;

    MPI_Request req_recv;
    TypeProfile *buf_recv;
    if (!me.isLast()) {
        buf_recv = new TypeProfile [length_your_elem + length_recv_message];
        me.iRecv(&buf_recv[length_your_elem], length_recv_message, MPI_INT,
                    me.getRank() + 1, 0, &req_recv);
    }

    Decomposition decomposition(me);
    decomposition.height = number_all_window;
    decomposition.offset_row = (profile.offset + step - 1) / step;
    decomposition.width = number_coef;
    decomposition.offset_column = 0;
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

        me.wait(&req_recv, MPI_STATUS_IGNORE);

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

    if (!me.isFirst())
        me.wait(&req_send, MPI_STATUS_IGNORE);
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

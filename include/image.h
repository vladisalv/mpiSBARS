#ifndef __IMAGE_HEADER__
#define __IMAGE_HEADER__

#include "myMPI.h"
#include "matrix_analysis.h"
#include "matrix_gomology.h"
#include "types.h"

#include <math.h>

typedef struct TMyPixel
{
    unsigned char Blue;
    unsigned char Green;
    unsigned char Red;
} Pix;

class Image {
    MyMPI me;
    void drawBmpMPI(MatrixGomology& matrixGomology, char *file_name);
public:
    Image(MyMPI me);
    ~Image();

    void drawImage(MatrixGomology& matrixGomology, char *file_name);
};

#endif /* __IMAGE_HEADER__ */

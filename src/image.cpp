#include "image.h"

Image::Image(MyMPI new_me)
    : me(new_me)
{
}

Image::~Image()
{
}


void Image::drawImage(MatrixGomology& matrixGomology, char *file_name)
{
    drawBmpMPI(matrixGomology, file_name);
}

void Image::drawBmpMPI(MatrixGomology& matrixGomology, char *file_name)
{
    ulong height_common, *offsetHeight, *sumOffsetHeight;
    height_common = matrixGomology.offsetLength(offsetHeight, sumOffsetHeight, &matrixGomology.height);

    size_t sizeRowPix = sizeof(Pix) * matrixGomology.width;
    size_t sizeRowNull = sizeRowPix % 4;
    size_t sizeRowBMP = sizeRowPix + sizeRowNull;

    MPI_File hFile;
    MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &hFile);
    if (me.isRoot()) {
        //------FILE HEADER------------------
        unsigned short int bfType = 0x4D42 ;
        unsigned short int bfReserved1 = 0;
        unsigned short int bfReserved2 = 0;
        unsigned int bfOffBits = 0x36;
        unsigned int bfSize = bfOffBits + height_common * sizeRowBMP;
        //-----INFO HEADER-----------------
        unsigned int       biSize          = 0x28;
        unsigned int       biWidth         = matrixGomology.width;  //long
        unsigned int       biHeight        = height_common; //long
        unsigned short int biPlanes        = 1;
        unsigned short int biBitCount      = 24;
        unsigned int       biCompression   = 0;
        unsigned int       biSizeImage     = 0;
        unsigned int       biXPelsPerMeter = 0; //long
        unsigned int       biYPelsPerMeter = 0; //long
        unsigned int       biClrUsed       = 65536;
        unsigned int       biClrImportant  = 0;

        MPI_File_write(hFile, &bfType,      1, MPI_UNSIGNED_SHORT, 0);
        MPI_File_write(hFile, &bfSize,      1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &bfReserved1, 1, MPI_UNSIGNED_SHORT, 0);
        MPI_File_write(hFile, &bfReserved2, 1, MPI_UNSIGNED_SHORT, 0);
        MPI_File_write(hFile, &bfOffBits,   1, MPI_UNSIGNED,       0);

        MPI_File_write(hFile, &biSize,          1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biWidth,         1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biHeight,        1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biPlanes,        1, MPI_UNSIGNED_SHORT, 0);
        MPI_File_write(hFile, &biBitCount,      1, MPI_UNSIGNED_SHORT, 0);
        MPI_File_write(hFile, &biCompression,   1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biSizeImage,     1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biXPelsPerMeter, 1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biYPelsPerMeter, 1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biClrUsed,       1, MPI_UNSIGNED,       0);
        MPI_File_write(hFile, &biClrImportant,  1, MPI_UNSIGNED,       0);
    }
    MPI_Offset offsetHead = sizeof(unsigned int) * 11 + sizeof(unsigned short int) * 5;
    MPI_Offset offsetFile = offsetHead + (me.isLast() ? 0 : (height_common - sumOffsetHeight[me.getRank() + 1]) * sizeRowBMP);
    MPI_File_set_view(hFile, offsetFile, MPI_BYTE, MPI_BYTE, (char *)"native", MPI_INFO_NULL);

    MPI_Datatype PixType;
    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &PixType);
    MPI_Type_commit(&PixType);
    MPI_Datatype PixTypeRow;
    MPI_Type_contiguous(matrixGomology.width, PixType, &PixTypeRow);
    MPI_Type_commit(&PixTypeRow);
    MPI_Datatype BmpTypeRow;
    MPI_Type_hvector(1, 1, sizeRowBMP, PixTypeRow, &BmpTypeRow);
    MPI_Type_commit(&BmpTypeRow);

    char *tmpBmpRow = new char [sizeRowBMP];
    for (int i = 0; i < sizeRowBMP; i++)
        tmpBmpRow[i] = 0;
    Pix *tmpPix = (Pix *)tmpBmpRow;
    Pix pixBlack, pixWhite;
    pixBlack.Blue = pixBlack.Green = pixBlack.Red = 0;
    pixWhite.Blue = pixWhite.Green = pixWhite.Red = 255;
    for (int i = matrixGomology.height - 1; i >= 0; i--) {
        for (int j = 0; j < matrixGomology.width; j++) {
            if (matrixGomology.data[i * matrixGomology.width + j])
                tmpPix[j] = pixBlack;
            else
                tmpPix[j] = pixWhite;
        }
        MPI_File_write(hFile, tmpBmpRow, 1, BmpTypeRow, MPI_STATUS_IGNORE);
    }

    delete [] tmpBmpRow;
    MPI_Type_free(&PixType);
    MPI_File_close(&hFile);
}

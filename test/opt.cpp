#include "options.h"

int main(int argc, char *argv[])
{
    Options opt(argc, argv);
    if (opt.errorMode()) {
        opt.errorPrint();
        opt.helpPrint();
        return 1;
    } else if (opt.versionMode()) {
        opt.versionPrint();
        return 0;
    } else if (opt.helpMode()) {
        opt.helpPrint();
        return 0;
    }
}

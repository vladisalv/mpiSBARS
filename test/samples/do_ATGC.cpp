#include <iostream>
#include <fstream>
#include <cstdlib>

#define NUM_REPEAT 2000

using namespace std;

char give(); // give you A,T,G or C

int main(int argc, char  * argv[])
{
    if (argc < 2) {
        cout << "You must have name of file" << endl;
        return 1;
    }
    ofstream file(argv[1]);

    srand(time(NULL));

    for (int i  =  0; i < 10000; i++)
        file << give();

    // do special repeat
    char repeat[NUM_REPEAT];
    for (int i = 0; i < NUM_REPEAT; i++) {
        repeat[i] = give();
        file << repeat[i];
    }

    // random
    for (int i  =  0; i < 40000; i++)
        file << give();

    // do our repeat again
    for (int i = 0; i < NUM_REPEAT; i++)
        file << repeat[i];

    // random
    for (int i  =  0; i < 20000; i++)
        file << give();

    file.close();
    cout << "He" << endl;
    return 0;
}


char give()
{
    switch (rand() % 4)
    {
        case 0: return 'A';
        case 1: return 'G';
        case 2: return 'T';
        case 3: return 'C';
        default:return 'A'; // this is unreal, but ...
    }
}

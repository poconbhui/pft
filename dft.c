#include "dft.h"

#define PI 3.14159265

void dft(
    double complex* in, double complex* out, int N,
    enum DFT_direction dir
) {
    int k, j;
    double complex exponent = dir*I*2.0*PI/N;

    for(k=0; k<N; ++k) {
        out[k] = 0;

        for(j=0; j<N; ++j) {
            out[k] = out[k] + in[j]*cexp(j*k*exponent);
        }
    }
}

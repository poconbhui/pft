#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex.h>

#include "funcs.h"


#define PI 3.14159265

enum DFT_direction {DFT_FORWARD, DFT_BACKWARD};
void dft(
    double complex* in, double complex* out, int N,
    enum DFT_direction dir
) {
    int k, j;
    double complex exponent = I*2.0*PI/N;

    if(dir == DFT_FORWARD) {
        printf("FORWARD\n");
        exponent = -exponent;
    }
    else if(dir == DFT_BACKWARD) {
        printf("BACKWARD\n");
        exponent = exponent;
    }

    for(k=0; k<N; ++k) {
        out[k] = 0;

        for(j=0; j<N; ++j) {
            out[k] = out[k] + in[j]*cexp(j*k*exponent);
        }
    }
}

int main(int argc, char* argv) {
    int N = 2*2*2*2;

    double complex *initial, *forward, *backward;

    initial  = (double complex*) malloc(N*sizeof(double complex));
    forward  = (double complex*) malloc(N*sizeof(double complex));
    backward = (double complex*) malloc(N*sizeof(double complex));

    set_sin(initial, N);

    dft(initial, forward, N, DFT_FORWARD);
    dft(forward, backward, N, DFT_BACKWARD);

    rescale(backward, N);

    print_arr(initial, N);
    printf("\n----\n\n");
    print_arr(forward, N);
    printf("\n----\n\n");
    print_arr(backward, N);

    free(initial);
    free(forward);
    free(backward);

    return EXIT_SUCCESS;
}

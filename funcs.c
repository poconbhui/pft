#include "funcs.h"

void set_sin(double complex* in, int N) {
    int i;
    double two_pi_over_N = 2*3.14159265/N;

    for(i=0; i<N; ++i) {
        in[i] = sin(2.0*i*two_pi_over_N);
    }
}

void rescale(double complex* in, int N) {
    int i;

    for(i=0; i<N; ++i) {
        in[i] = in[i]/N;
    }
}

void print_arr(double complex* in, int N) {
    int i;
    
    for(i=0; i<N; ++i) {
        printf("%d: (%f, %f)\n", i, creal(in[i]), cimag(in[i]));
    }
}

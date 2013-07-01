#include "funcs.h"

double complex set_sin_i(int i, int N) {
    double two_pi_over_N = 2*3.14159265/N;

    return sin(2.0*i*two_pi_over_N);
}

void set_sin(double complex* in, int N) {
    int i;

    for(i=0; i<N; ++i) {
        in[i] = set_sin_i(i, N);
    }
}

void rescale(double complex* in, int N, complex double scale) {
    int i;

    for(i=0; i<N; ++i) {
        in[i] = in[i]*scale;
    }
}

void print_arr(double complex* in, int N) {
    int i;
    
    for(i=0; i<N; ++i) {
        printf("%d: (%f, %f)\n", i, creal(in[i]), cimag(in[i]));
    }
}

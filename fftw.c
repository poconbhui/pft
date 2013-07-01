#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include <fftw3.h>

#include "funcs.h"


int main(int argc, char* argv[]) {
    int N = 2*2*2*2;

    double complex *initial, *forward, *backward;
    fftw_plan p_forward, p_backward;

    initial  = (double complex*) fftw_malloc(sizeof(double complex) * N);
    forward  = (double complex*) fftw_malloc(sizeof(double complex) * N);
    backward = (double complex*) fftw_malloc(sizeof(double complex) * N);

    set_sin(initial, N);

    p_forward = fftw_plan_dft_1d(
        N, (fftw_complex*) initial, (fftw_complex*) forward,
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    p_backward = fftw_plan_dft_1d(
        N, (fftw_complex*) forward, (fftw_complex*) backward,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    fftw_execute(p_forward);
    fftw_execute(p_backward);

    rescale(backward, N, 1.0/N);

    // Output data
    print_arr(initial, N);
    printf("\n----\n\n");
    print_arr(forward, N);
    printf("\n----\n\n");
    print_arr(backward, N);

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);

    fftw_free(initial);
    fftw_free(forward);
    fftw_free(backward);

    return EXIT_SUCCESS;
}

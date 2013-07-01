#include "dft.h"

int main(int argc, char* argv) {
    int N = 2*2*2*2;

    double complex *initial, *forward, *backward;


    initial  = (double complex*) malloc(N*sizeof(double complex));
    forward  = (double complex*) malloc(N*sizeof(double complex));
    backward = (double complex*) malloc(N*sizeof(double complex));


    set_sin(initial, N);


    dft(initial, forward, N, DFT_FORWARD);
    dft(forward, backward, N, DFT_BACKWARD);


    rescale(backward, N, 1.0/N);


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

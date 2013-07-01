#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include <mpi.h>

#include <fftw3.h>

#include "funcs.h"


static int MPI_main(int argc, char* argv[]) {
    int N;
    int num_reps;

    double complex *initial, *final;
    fftw_plan p_forward;

    int rep;


    // Set variables from command line
    // mpiexec -n 1 ./fftw.bench N num_reps
    if(argc != 3) {
        printf("%d\n", argc);
        printf("Usage: mpiexec -n 1 ./fftw.bench N num_reps\n");
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    num_reps = atoi(argv[2]);


    initial  = (double complex*) fftw_malloc(sizeof(double complex) * N);
    final  = (double complex*) fftw_malloc(sizeof(double complex) * N);

    set_sin(initial, N);

    p_forward = fftw_plan_dft_1d(
        N, (fftw_complex*) initial, (fftw_complex*) final,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    // Run initial execution
    fftw_execute(p_forward);


    // Start timing.
    double start_time = MPI_Wtime();
    for(rep=0; rep<num_reps; ++rep) {
        fftw_execute(p_forward);
    }
    double end_time = MPI_Wtime();


    printf("Time: %e, N: %d, Reps: %d\n", end_time-start_time, N, num_reps);


    fftw_destroy_plan(p_forward);


    fftw_free(initial);
    fftw_free(final);


    return EXIT_SUCCESS;
}


int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}

#include "dft.h"

#include <mpi.h>

static int MPI_main(int argc, char* argv[]) {
    int N = 2*2*2*2;
    int num_reps;

    double complex *initial, *final;

    int rep;


    if(argc != 3) {
        printf("Usage: mpiexec -n 1 ./dft.bench N num_reps\n");
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    num_reps = atoi(argv[2]);


    initial  = (double complex*) malloc(N*sizeof(double complex));
    final  = (double complex*) malloc(N*sizeof(double complex));


    set_sin(initial, N);


    // Do initial dft
    dft(initial, final, N, DFT_FORWARD);


    double start_time = MPI_Wtime();
    for(rep=0; rep<num_reps; ++rep) {
        dft(initial, final, N, DFT_FORWARD);
    }
    double end_time = MPI_Wtime();


    printf("Time: %e, N: %d, Reps: %d\n", end_time-start_time, N, num_reps);


    free(initial);
    free(final);


    return EXIT_SUCCESS;
}


int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}

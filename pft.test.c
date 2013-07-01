#include "pft.h"

#include <fftw3.h>


// A slightly crappy test.
// Compares output of pft and fftw. If results differ by more than
// 10^-6, it complains.
int MPI_main(int argc, char* argv[]) {

    // Array sizes
    int N = 2*2*2*2;
    int N_per_proc = 2*2*2;


    // Test arrays
    double complex *pft_initial_array;
    double complex *pft_forward_array;
    double complex *pft_backward_array;

    double complex* fftw_initial_array;
    double complex* fftw_forward_array;
    double complex* fftw_backward_array;


    // Plans
    pft_plan pft_forward_plan;
    pft_plan pft_backward_plan;

    fftw_plan fftw_forward_plan;
    fftw_plan fftw_backward_plan;


    // MPI stuff
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int nprocs;

    int i;


    if(argc != 3) {
      printf("Usage: mpiexec -n P ./pft.test N N_per_proc\n");
      return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    N_per_proc = atoi(argv[2]);
    MPI_Comm_size(comm, &nprocs);


    // Enforce assumptions
    if(N%nprocs != 0) {
        printf("Number of processes must divide N evenly.\n");
        return EXIT_FAILURE;
    }
    if(N%N_per_proc != 0) {
        printf("N_per_proc must divide N evenly.\n");
        return EXIT_FAILURE;
    }
    if(N/N_per_proc > nprocs) {
        printf("N/N_per_proc must be smaller than the number of processes.\n");
        return EXIT_FAILURE;
    }
    if(nprocs/(N/N_per_proc) > (N/N_per_proc)) {
        printf(
            "Number of replicas must not exceed number of systolic elements\n"
        );
        return EXIT_FAILURE;
    }
    if(nprocs % (N/N_per_proc)) {
        printf("Number of processes must be divisible by N/N_per_proc.\n");
        return EXIT_FAILURE;
    }



    // Allocate arrays
    pft_initial_array  = pft_alloc(N_per_proc, comm);
    pft_forward_array  = pft_alloc(N_per_proc, comm);
    pft_backward_array = pft_alloc(N_per_proc, comm);

    fftw_initial_array  =
        (double complex*) fftw_malloc(sizeof(double complex) * N);
    fftw_forward_array =
        (double complex*) fftw_malloc(sizeof(double complex) * N);
    fftw_backward_array =
        (double complex*) fftw_malloc(sizeof(double complex) * N);


    // Set initial arrays
    pft_set_sin(pft_initial_array, N, N_per_proc, comm);
    set_sin(fftw_initial_array, N);


    // Generate DFT plans
    pft_forward_plan = pft_plan_dft_1d(
        N, N_per_proc, pft_initial_array, pft_forward_array, PFT_FORWARD, comm
    );
    pft_backward_plan = pft_plan_dft_1d(
        N, N_per_proc, pft_forward_array, pft_backward_array, PFT_BACKWARD, comm
    );

    fftw_forward_plan = fftw_plan_dft_1d(
        N,
        (fftw_complex*) fftw_initial_array,
        (fftw_complex*) fftw_forward_array,
        FFTW_FORWARD, FFTW_ESTIMATE
    );
    fftw_backward_plan = fftw_plan_dft_1d(
        N,
        (fftw_complex*) fftw_forward_array,
        (fftw_complex*) fftw_backward_array,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );


    // Execute plans
    pft_execute(pft_forward_plan);
    pft_execute(pft_backward_plan);

    fftw_execute(fftw_forward_plan);
    fftw_execute(fftw_backward_plan);


    MPI_Comm_rank(comm, &rank);


    // Check equivalence of pft and fftw results.
    // Accept error of 1E-6.
    double complex* local = (double complex*) malloc(N*sizeof(double complex));

    pft_gather(pft_initial_array, local, N, N_per_proc, 0, comm);
    if(rank == 0) {
        /*
        printf("\n\n-------\n");
        print_arr(local, N);
        printf("-------\n");
        print_arr(fftw_initial_array, N);
        */

        // Compare arrays
        for(i=0; i<N; ++i) {
            double diff = cabs(local[i] - fftw_initial_array[i]);
            if(diff > 1E-6) printf("INITIAL DIFFERENCE: %e\n", creal(diff));
        }
    }
    pft_gather(pft_forward_array, local, N, N_per_proc, 0, comm);
    if(rank == 0) {
        /*
        printf("\n\n-------\n");
        print_arr(local, N);
        printf("-------\n");
        print_arr(fftw_forward_array, N);
        */

        // Compare arrays
        for(i=0; i<N; ++i) {
            double diff = cabs(local[i] - fftw_forward_array[i]);
            if(diff > 1E-6) printf("FORWARD DIFFERENCE: %e\n", creal(diff));
        }
    }
    pft_gather(pft_backward_array, local, N, N_per_proc, 0, comm);
    if(rank == 0) {
        /*
        printf("\n\n-------\n");
        print_arr(local, N);
        printf("-------\n");
        print_arr(fftw_backward_array, N);
        */

        // Compare arrays
        for(i=0; i<N; ++i) {
            double diff = cabs(local[i] - fftw_backward_array[i]);
            if(diff > 1E-6) printf("BACKWARD DIFFERENCE: %e\n", creal(diff));
        }
    }


    // Destroy plans
    pft_destroy_plan(pft_forward_plan);
    pft_destroy_plan(pft_backward_plan);

    fftw_destroy_plan(fftw_forward_plan);
    fftw_destroy_plan(fftw_backward_plan);


    // Free memory
    pft_free(pft_initial_array, comm);
    pft_free(pft_forward_array, comm);
    pft_free(pft_backward_array, comm);

    fftw_free(fftw_initial_array);
    fftw_free(fftw_forward_array);
    fftw_free(fftw_backward_array);


    printf("Tests finished.\n");


    return EXIT_SUCCESS;
}


int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}

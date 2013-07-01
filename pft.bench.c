#include "pft.h"

#include <mpi.h>


static int MPI_main(int argc, char* argv[]) {
    int N;
    int N_per_proc;
    int num_reps;

    double complex *initial, *final;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int nprocs;

    pft_plan p_forward;

    int rep;


    if(argc != 4) {
        printf("Usage: mpiexec -n P ./pft.bench N N_per_proc num_reps\n");
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    N_per_proc = atoi(argv[2]);
    num_reps = atoi(argv[3]);


    // Enforce assumptions
    MPI_Comm_size(comm, &nprocs);

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

    MPI_Comm_rank(comm, &rank);


    initial = pft_alloc(N_per_proc, comm);
    final   = pft_alloc(N_per_proc, comm);


    pft_set_sin(initial, N, N_per_proc, comm);


    p_forward = pft_plan_dft_1d(
        N, N_per_proc, initial, final , PFT_FORWARD, comm
    );


    // Perform initial PFT
    pft_execute(p_forward);


    double start_time = MPI_Wtime();
    for(rep=0; rep<num_reps; ++rep) {
        pft_execute(p_forward);
    }
    double end_time = MPI_Wtime();


    // Use highest overall time
    double diff = end_time - start_time;
    double max_diff;
    MPI_Reduce(&diff, &max_diff, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if(rank == 0) {
        printf(
            "Time: %e, N: %d, N_per_proc: %d, Reps: %d\n",
            max_diff, N, N_per_proc, num_reps
        );
    }


    pft_destroy_plan(p_forward);


    pft_free(initial, comm);
    pft_free(final, comm);


    return EXIT_SUCCESS;
}


int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}

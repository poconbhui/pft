#include "pft.h"


int MPI_main(int argc, char* argv[]) {
    int N = 2*2*2*2;
    int N_per_proc = 2*2*2;

    double complex *initial, *forward, *backward;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int nprocs;

    pft_plan p_forward, p_backward;


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


    initial  = pft_alloc(N_per_proc, comm);
    forward  = pft_alloc(N_per_proc, comm);
    backward = pft_alloc(N_per_proc, comm);


    pft_set_sin(initial, N, N_per_proc, comm);
    pft_print_arr(initial, N, N_per_proc, comm);


    p_forward = pft_plan_dft_1d(N, N_per_proc, initial, forward , PFT_FORWARD, comm);
    p_backward = pft_plan_dft_1d(N, N_per_proc, forward, backward, PFT_BACKWARD, comm);


    pft_execute(p_forward);
    pft_execute(p_backward);

    rescale(backward, N_per_proc, 1.0/N);


    if(rank == 0) printf("\n\n\n");
    pft_print_arr(initial, N, N_per_proc, comm);
    if(rank == 0) printf("\n\n-----\n");
    pft_print_arr(forward, N, N_per_proc, comm);
    if(rank == 0) printf("\n\n-----\n");
    pft_print_arr(backward, N, N_per_proc, comm);


    pft_destroy_plan(p_forward);
    pft_destroy_plan(p_backward);


    pft_free(initial, comm);
    pft_free(forward, comm);
    pft_free(backward, comm);


    return EXIT_SUCCESS;
}


int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}

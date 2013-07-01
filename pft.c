#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex.h>

#include <mpi.h>

#include "funcs.h"

int MPI_main(int argc, char* argv[]);
int main(int argc, char* argv[]) {
    int MPI_main_return;

    MPI_Init(&argc, &argv);
    MPI_main_return = MPI_main(argc, argv);
    MPI_Finalize();
    return MPI_main_return;
}


int pft_chunk_size(int N, MPI_Comm comm) {
    int nprocs;

    MPI_Comm_size(comm, &nprocs);

    return (N + nprocs - 1)/nprocs;
}

void pft_get_chunk_data(
    int N, int N_per_proc, int rank, MPI_Comm comm,
    int *chunk_size, int *chunk_start, int *chunk_end
) {
    *chunk_size = N_per_proc;

    // Get equivalent rank in 0th systolic loop
    rank = rank%(N/N_per_proc);

    *chunk_start = rank*(*chunk_size);
    *chunk_end = (*chunk_start) + (*chunk_size);
    if((*chunk_end) > N) *chunk_end = N;

    *chunk_size = (*chunk_end) - (*chunk_start);
}


void pft_set_sin(double complex* in, int N, int N_per_proc, MPI_Comm comm) {
    int rank;

    int chunk_size;
    int chunk_start;
    int chunk_end;

    int i;


    MPI_Comm_rank(comm, &rank);
    pft_get_chunk_data(
        N, N_per_proc, rank, comm,
        &chunk_size, &chunk_start, &chunk_end
    );

    for(i=0; (chunk_start+i) < chunk_end; ++i) {
        in[i] = set_sin_i(chunk_start+i, N);
    }
}

void pft_print_arr(double complex* in, int N, int N_per_proc, MPI_Comm comm) {
    int rank;
    int this_rank;

    int chunk_size;
    int chunk_start;
    int chunk_end;

    int i;


    MPI_Comm_rank(comm, &this_rank);
    pft_get_chunk_data(
        N, N_per_proc, this_rank, comm,
        &chunk_size, &chunk_start, &chunk_end
    );

    for(rank=0; rank<(N/N_per_proc); ++rank) {
        MPI_Barrier(comm);
        if(this_rank == rank) {
            for(i=0; chunk_start+i < chunk_end; ++i) {
                printf(
                    "%d: (%f, %f)\n",
                    chunk_start+i, creal(in[i]), cimag(in[i])
                );
            }
        }
    }
    MPI_Barrier(comm);
}


double complex* pft_alloc(int N_per_proc, MPI_Comm comm) {
    //int alloc_size = pft_chunk_size(N, comm);
    int alloc_size = N_per_proc;

    MPI_Barrier(comm);

    return (double complex*) malloc(alloc_size*sizeof(double complex));
}

void pft_free(double complex* in, MPI_Comm comm) {
    MPI_Barrier(comm);
    free(in);
}


enum PFT_direction {PFT_FORWARD, PFT_BACKWARD};
void pft(
    double complex* in, double complex* out, int N, int N_per_proc,
    enum PFT_direction dir,
    MPI_Comm comm
) {
    int nprocs;
    int local_rank;

    int local_chunk_size;
    int local_chunk_start;
    int local_chunk_end;

    int foreign_rank;

    int systolic_count;

    int systolic_offset;
    int systolic_rank;
    int systolic_replica_rank;
    int systolic_nprocs;
    int systolic_nreplicas;

    double complex* foreign_array;
    double complex* swap_array;

    int chunk_size = N_per_proc;

    const double PI = 3.14159265;
    int k, j;
    double complex exponent = I*2.0*PI/N;


    if(dir == PFT_FORWARD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0) printf("FORWARD\n");
        exponent = -exponent;
    }
    else if(dir == PFT_BACKWARD) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(rank == 0) printf("BACKWARD\n");
        exponent = exponent;
    }


    foreign_array = pft_alloc(N, comm);
    swap_array    = pft_alloc(N, comm);


    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &local_rank);


    pft_get_chunk_data(
        N, N_per_proc, local_rank, comm,
        &local_chunk_size, &local_chunk_start, &local_chunk_end
    );


    // Initialise output array
    for(k=0; k<local_chunk_size; ++k) {
        out[k] = 0;
    }


    systolic_rank = local_rank % (N/N_per_proc);
    systolic_nprocs = (N/N_per_proc);
    systolic_replica_rank = local_rank/(N/N_per_proc);
    systolic_offset = systolic_replica_rank*systolic_nprocs;
    systolic_nreplicas = nprocs/systolic_nprocs;


    // Initialise foreign array
    for(k=0; k<chunk_size; ++k) {
        foreign_array[k] = in[k];
    }

    // Find initial foreign rank
    foreign_rank = (
        systolic_rank
        + systolic_replica_rank*systolic_nprocs/systolic_nreplicas
    ) % systolic_nprocs;

    // Do initial systolic swap
    {
        int swap_rank = systolic_offset + foreign_rank;


        for(k=0; k<chunk_size; k++) {
            swap_array[k] = foreign_array[k];
        }


        MPI_Sendrecv(
            swap_array, chunk_size, MPI_DOUBLE_COMPLEX, swap_rank, 0,
            foreign_array, chunk_size, MPI_DOUBLE_COMPLEX, swap_rank, 0,
            comm, MPI_STATUS_IGNORE
        );

    }

    foreign_rank--;
    for(systolic_count=0; systolic_count<systolic_nprocs/systolic_nreplicas; ++systolic_count) {
        foreign_rank = (foreign_rank + 1) % systolic_nprocs;

        int foreign_chunk_size;
        int foreign_chunk_start;
        int foreign_chunk_end;


        pft_get_chunk_data(
            N, N_per_proc, foreign_rank, comm,
            &foreign_chunk_size, &foreign_chunk_start, &foreign_chunk_end
        );


        // do systolic pulse
        // Receive from left, send to right
        if(systolic_count != 0) {
            MPI_Request left_req, right_req;
            int left_rank = systolic_offset
                + ((systolic_rank+1)%systolic_nprocs);
            int right_rank = systolic_offset
                + ((systolic_rank-1+systolic_nprocs)%systolic_nprocs);


            for(k=0; k<chunk_size; k++) {
                swap_array[k] = foreign_array[k];
            }


            MPI_Isend(
                swap_array, chunk_size, MPI_DOUBLE_COMPLEX,
                right_rank, 0, comm,
                &right_req
            );

            MPI_Irecv(
                foreign_array, chunk_size, MPI_DOUBLE_COMPLEX,
                left_rank, 0, comm,
                &left_req
            );

            {
                MPI_Request reqs[2] = {left_req, right_req};
                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            }

        }



        for(k=0; k<local_chunk_size; ++k) {
            for(j=0; j<foreign_chunk_size; ++j) {
                int absolute_k = local_chunk_start + k;
                int absolute_j = foreign_chunk_start + j;

                out[k] = out[k] + foreign_array[j]*cexp(
                    absolute_j*absolute_k*exponent
                );
            }
        }

    }

    // Reduce values between equivalent systolic elements
    // Define new communicator on elements with same systolic_rank
    MPI_Comm elem_comm;
    MPI_Comm_split(comm, systolic_rank, 0, &elem_comm);

    for(k=0; k<chunk_size; ++k) {
        foreign_array[k] = out[k];
    }

    MPI_Allreduce(
        foreign_array, out, chunk_size, MPI_DOUBLE_COMPLEX,
        MPI_SUM, elem_comm
    );

    MPI_Comm_free(&elem_comm);


    pft_free(foreign_array, comm);
    pft_free(swap_array, comm);
}


int MPI_main(int argc, char* argv[]) {
    int N = 2*2*2*2;
    int N_per_proc = 2*2*2;

    double complex *initial, *forward, *backward;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int nprocs;


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


    pft(initial, forward , N, N_per_proc, PFT_FORWARD, comm);
    pft(forward, backward, N, N_per_proc, PFT_BACKWARD, comm);

    rescale(backward, N_per_proc, 1.0/N);


    if(rank == 0) printf("\n\n\n");
    pft_print_arr(initial, N, N_per_proc, comm);
    if(rank == 0) printf("\n\n-----\n");
    pft_print_arr(forward, N, N_per_proc, comm);
    if(rank == 0) printf("\n\n-----\n");
    pft_print_arr(backward, N, N_per_proc, comm);


    pft_free(initial, comm);
    pft_free(forward, comm);
    pft_free(backward, comm);


    return EXIT_SUCCESS;
}

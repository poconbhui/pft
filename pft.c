#include "pft.h"


/*
 * Not currently in use. Superceded by explicit N_per_proc.
 *
 * static int pft_chunk_size(int N, MPI_Comm comm) {
 *     int nprocs;
 * 
 *     MPI_Comm_size(comm, &nprocs);
 * 
 *     return (N + nprocs - 1)/nprocs;
 * }
 *
 */

static void pft_get_chunk_data(
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



struct systolic_loop_data {
    MPI_Comm comm;
    int rank;
    int nprocs;
    int left_rank;
    int right_rank;
};
static void gen_systolic_loop_data(
    int N, int N_per_proc, MPI_Comm comm,
    struct systolic_loop_data* systolic,
    int* replica_num, int* num_replicas
) {
    int rank;
    int nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // Set replicated systolic loop data
    systolic->rank = rank % (N/N_per_proc);
    systolic->nprocs = (N/N_per_proc);
    systolic->left_rank  = (systolic->rank+1)%systolic->nprocs;
    systolic->right_rank = (systolic->rank-1+systolic->nprocs)%systolic->nprocs;

    *replica_num = rank/(N/N_per_proc);
    *num_replicas = nprocs/systolic->nprocs;

    // Define new communicator on local systolic loop
    MPI_Comm_split(comm, *replica_num, rank, &systolic->comm);
}
static void free_systolic_loop_data(struct systolic_loop_data* systolic) {
    MPI_Comm_free(&systolic->comm);
}


// Gather whole array on to process root.
void pft_gather(
    double complex* in, double complex* out, int N, int N_per_proc,
    int root, MPI_Comm comm
) {
    int rank;

    struct systolic_loop_data systolic;
    int replica_num;
    int num_replicas;

    int systolic_root;


    MPI_Comm_rank(comm, &rank);
    replica_num = rank/(N/N_per_proc);


    //Generate systolic loop data
    gen_systolic_loop_data(
        N, N_per_proc, comm,
        &systolic, &replica_num, &num_replicas
    );


    // Find if this process is in the requested ring, and also
    // find the requested process in the ring.
    //
    // We do this by having all processes set systolic root to zero,
    // except the actual root. By summing these values, if a process
    // receives a nonzero result, they are in the ring, and the value
    // they've received is also the reduce rank.
    //
    // To allow for rank 0 root, we increment by 1 before the test
    // reduction and decrement afterwards.
    //
    {
        int recv_systolic_root;

        if(rank == root) {
            systolic_root = rank + 1;
        }
        else {
            systolic_root = 0;
        }

        MPI_Allreduce(
            &systolic_root, &recv_systolic_root, 1, MPI_INT,
            MPI_SUM, systolic.comm
        );

        if(recv_systolic_root > 0) {
            systolic_root = recv_systolic_root - 1;
        }
        else {
            systolic_root = -1;
        }
    }


    // Only if we're in the required loop
    if(systolic_root != -1) {
        MPI_Gather(
            in,  N_per_proc, MPI_DOUBLE_COMPLEX,
            out, N_per_proc, MPI_DOUBLE_COMPLEX,
            systolic_root, systolic.comm
        );
    }


    free_systolic_loop_data(&systolic);

    
}


// pft_plan
typedef struct {
    int N;
    int N_per_proc;

    enum PFT_direction dir;

    struct systolic_loop_data systolic;
    int num_replicas;
    MPI_Comm equiv_comm;

    double complex* swap_array;
    double complex* foreign_array;
    double complex* in;
    double complex* out;

    int local_chunk_size;
    int local_chunk_start;

    int initial_foreign_rank;
    int initial_right;
    int initial_left;
} pft_plan_type;


pft_plan pft_plan_dft_1d(
    int N, int N_per_proc, double complex* in, double complex* out,
    enum PFT_direction dir,
    MPI_Comm comm
) {
    int nprocs;
    int local_rank;

    int local_chunk_size;
    int local_chunk_start;
    int local_chunk_end;

    double complex* foreign_array;
    double complex* swap_array;


    foreign_array = (double complex*) malloc(N*sizeof(double complex));
    swap_array    = (double complex*) malloc(N*sizeof(double complex));


    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &local_rank);


    pft_get_chunk_data(
        N, N_per_proc, local_rank, comm,
        &local_chunk_size, &local_chunk_start, &local_chunk_end
    );


    // Set replicated systolic loop data
    struct systolic_loop_data systolic;
    int replica_num;
    int num_replicas;

    gen_systolic_loop_data(
        N, N_per_proc, comm,
        &systolic, &replica_num, &num_replicas
    );


    // Find initial swap rank from initial foreign rank
    int initial_left = (
        systolic.rank
        + (replica_num*systolic.nprocs)/num_replicas
    ) % systolic.nprocs;
    int initial_right = (
        systolic.rank 
        - (replica_num*systolic.nprocs)/num_replicas + systolic.nprocs
    ) % systolic.nprocs;

    // Find initial foreign rank
    int initial_foreign_rank = initial_left;



    // Define new communicator on elements with same systolic_rank
    MPI_Comm equiv_comm;
    MPI_Comm_split(comm, systolic.rank, 0, &equiv_comm);


    // Set pft_plan
    pft_plan_type* p = (pft_plan_type*) malloc(sizeof(pft_plan_type));
    p->N = N;
    p->N_per_proc = N_per_proc;
    p->dir = dir;
    p->systolic = systolic;
    p->num_replicas = num_replicas;
    p->swap_array = swap_array;
    p->foreign_array = foreign_array;
    p->in = in;
    p->out = out;
    p->local_chunk_size = local_chunk_size;
    p->local_chunk_start = local_chunk_start;
    p->equiv_comm = equiv_comm;
    p->initial_foreign_rank = initial_foreign_rank;
    p->initial_right = initial_right;
    p->initial_left = initial_left;


    return p;
}

void pft_destroy_plan(pft_plan p) {
    pft_plan_type* plan = p;

    free_systolic_loop_data(&plan->systolic);
    MPI_Comm_free(&(plan->equiv_comm));

    free(plan->foreign_array);
    free(plan->swap_array);

    free(plan);
}


void pft_execute(pft_plan p) {

    /*
     * Set data from pft_plan
     */
    pft_plan_type* plan = p;

    // Array sizes
    int N = plan->N;
    int N_per_proc = plan->N_per_proc;

    // DFT direction
    enum PFT_direction dir = plan->dir;

    // Systolic loop stuff
    // equiv_comm = communicator on equivalent elements in different replicas
    struct systolic_loop_data systolic = plan->systolic;
    int num_replicas = plan->num_replicas;
    MPI_Comm equiv_comm = plan->equiv_comm;


    // Arrays to be FTed and work arrays.
    double complex* in = plan->in;
    double complex* out = plan->out;
    double complex* swap_array = plan->swap_array;
    double complex* foreign_array = plan->foreign_array;

    int local_chunk_size = plan->local_chunk_size;
    int local_chunk_start = plan->local_chunk_start;

    int initial_foreign_rank = plan->initial_foreign_rank;
    int initial_right = plan->initial_right;
    int initial_left = plan->initial_left;



    // Iterators
    int systolic_count;
    int k, j;


    // DFT stuff
    const double PI = 3.14159265;
    double complex exponent = I*2.0*PI/N;

    if(dir == PFT_FORWARD) {
        exponent = -exponent;
    }
    else if(dir == PFT_BACKWARD) {
        exponent = exponent;
    }



    // Do initial systolic swap to initialize the foreign_array
    for(k=0; k<N_per_proc; k++) {
        swap_array[k] = in[k];
    }

    MPI_Sendrecv(
        swap_array, N_per_proc, MPI_DOUBLE_COMPLEX, initial_right, 0,
        foreign_array, N_per_proc, MPI_DOUBLE_COMPLEX, initial_left, 0,
        systolic.comm, MPI_STATUS_IGNORE
    );


    // Initialise output array
    for(k=0; k<local_chunk_size; ++k) {
        out[k] = 0;
    }


    for(
        systolic_count=0;
        systolic_count<systolic.nprocs/num_replicas;
        ++systolic_count
    ) {
        int foreign_rank;

        int foreign_chunk_size;
        int foreign_chunk_start;
        int foreign_chunk_end;

        foreign_rank = 
            ( initial_foreign_rank + systolic_count ) % systolic.nprocs;


        pft_get_chunk_data(
            N, N_per_proc, foreign_rank, systolic.comm,
            &foreign_chunk_size, &foreign_chunk_start, &foreign_chunk_end
        );


        // do systolic pulse
        // Receive from left, send to right
        if(systolic_count != 0) {
            for(k=0; k<N_per_proc; k++) {
                swap_array[k] = foreign_array[k];
            }

            MPI_Sendrecv(
                swap_array, N_per_proc, MPI_DOUBLE_COMPLEX,
                systolic.right_rank, 0,
                foreign_array, N_per_proc, MPI_DOUBLE_COMPLEX,
                systolic.left_rank, 0,
                systolic.comm, MPI_STATUS_IGNORE
            );
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


    // Sum values between equivalent systolic elements
    for(k=0; k<N_per_proc; ++k) {
        foreign_array[k] = out[k];
    }

    MPI_Allreduce(
        foreign_array, out, N_per_proc, MPI_DOUBLE_COMPLEX,
        MPI_SUM, equiv_comm
    );

}

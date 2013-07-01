#ifndef PFT_H_
#define PFT_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex.h>

#include <mpi.h>

#include "funcs.h"


enum PFT_direction {PFT_FORWARD, PFT_BACKWARD};

typedef void* pft_plan;

double complex* pft_alloc(int N_per_proc, MPI_Comm comm);
void pft_free(double complex* in, MPI_Comm comm);


void pft_set_sin(double complex* in, int N, int N_per_proc, MPI_Comm comm);
void pft_print_arr(double complex* in, int N, int N_per_proc, MPI_Comm comm);


pft_plan pft_plan_dft_1d(
    int N, int N_per_proc, double complex* in, double complex* out,
    enum PFT_direction dir,
    MPI_Comm comm
);
void pft_execute(pft_plan p);


void pft_destroy_plan(pft_plan p);


void pft_gather(
    double complex* in, double complex* out, int N, int N_per_proc,
    int root, MPI_Comm comm
);


#endif  // PFT_H_

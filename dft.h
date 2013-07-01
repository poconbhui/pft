#ifndef DFT_H_
#define DFT_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex.h>

#include "funcs.h"



enum DFT_direction {DFT_FORWARD=-1, DFT_BACKWARD=1};


void dft(
    double complex* in, double complex* out, int N,
    enum DFT_direction dir
);


#endif  // DFT_H_

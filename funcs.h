#include <stdio.h>
#include <math.h>
#include <complex.h>

double complex set_sin_i(int i, int N);
void set_sin(double complex* in, int N);
void rescale(double complex* in, int N, complex double scale);
void print_arr(double complex* in, int N);

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <iostream>
#include <array>
#include <stdio.h>
#include <cufft.h>
#include "CudaCommon.h"

class CudaPseudo
{
private:
    int MM, MM_COMPLEX, NN, NNf;
    int N_BLOCKS;
    int N_THREADS;
    double volume;

    cufftHandle plan_for, plan_bak;

    double *temp_d, *temp_arr;
    double *q1_d, *q2_d;
    double *qstep1_d, *qstep2_d;
    ftsComplex *kqin_d;

    double *expdwa_d, *expdwa_half_d;
    double *expdwb_d, *expdwb_half_d;
    double *phia_d, *phib_d;

    double *expf_d, *expf_half_d, *dv_d;
    double *expf,   *expf_half;
    
    void init_gaussian_factor(std::array<int,3> nx, std::array<double,3> dx, double ds);
    void onestep(double *qin1_d, double *qout1_d,
                 double *qin2_d, double *qout2_d,
                 double *expdw1_d, double *expdw1_half_d,
                 double *expdw2_d, double *expdw2_half_d);

public:

    CudaPseudo(std::array<int,3> nx,  std::array<double,3> dx,
               double *dv, double volume, double ds, int MM, int MMf,
               int n_blocks=256, int n_threads=256,  int process_idx=0);
    ~CudaPseudo();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb, 
                  double ds, double &QQ);

    void get_partition(double *q1_out,  double *q2_out, int n);
};

#endif

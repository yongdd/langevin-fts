#ifndef MKL_PSEUDO_H_
#define MKL_PSEUDO_H_

#include <iostream>
#include <array>
#include "MklFFT.h"

class MklPseudo
{
private:

    int MM, MM_COMPLEX;
    int NN, NNf;
    double *dv, *expf, *expf_half;
    double *q1, *q2;
    double volume;
    
    MklFFT *fft;

    void init_gaussian_factor(std::array<int,3> nx, std::array<double,3> dx, double ds);
    void onestep(double *qin, double *qout,
                 double *expdw, double *expdw_half);

public:

    MklPseudo(std::array<int,3> nx, std::array<double,3> dx,
               double *dv, double volume, double ds, int NN, int NNf);
    MklPseudo(int *nx, int *dx,
               double *dv, double volume, double ds, int NN, int NNf)
               :  MklPseudo({nx[0],nx[1],nx[2]}, {dx[0],dx[1],dx[2]},
               dv, volume, ds, NN, NNf) {};
    ~MklPseudo();

    void find_phi(double *phia,  double *phib,
                  double *q1_init, double *q2_init,
                  double *wa, double *wb,
                  double ds, double &QQ);

    void get_partition(double *q1_out,  double *q2_out, int n);
};
#endif

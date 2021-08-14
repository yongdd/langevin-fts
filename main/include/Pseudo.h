/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include "SimulationBox.h"
#include "FFT.h"

class Pseudo
{
protected:

    SimulationBox *sb;
    FFT *fft;

    int MM, MM_COMPLEX;
    int NN, NNf;
    double *expf, *expf_half;
    double *q1, *q2;

    void init_gaussian_factor(int *nx, double *dx, double ds);
    void onestep(double *qin, double *qout,
                 double *expdw, double *expdw_half);
public:
    Pseudo(SimulationBox *sb, double ds, int NN, int NNf);
    virtual ~Pseudo();

    virtual void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb,
        double ds, double &QQ);

    void get_partition(double *q1_out,  double *q2_out, int n);
};
#endif

/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include "SimulationBox.h"
#include "PolymerChain.h"
#include "FFT.h"

class Pseudo
{
protected:

    SimulationBox *sb;
    FFT *fft;

    int MM, MM_COMPLEX;
    int NN, NNf;
    double ds;
    double *expf, *expf_half;
    double *q1, *q2;

    void init_gaussian_factor(
        std::array<int,3> nx, std::array<double,3> dx, double ds);
    void onestep(double *qin, double *qout,
                 double *expdw, double *expdw_half);
public:
    Pseudo(SimulationBox *sb, PolymerChain *pc);
    virtual ~Pseudo();

    virtual void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb, double &QQ);
    void get_partition(double *q1_out,  double *q2_out, int n);
    
    // Methods for SWIG
    void find_phi(
        double *phia, int len_pa,
        double *phib, int len_pb,
        double *q1_init, int len_q1,
        double *q2_init, int len_q2,
        double *wa, int len_wa,
        double *wb, int len_wb,
        double &QQ)
    {
        find_phi(phia, phib, q1_init, q2_init, wa, wb, QQ);
    }
};
#endif

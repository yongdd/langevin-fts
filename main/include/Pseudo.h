/*-------------------------------------------------------------
* This is an abstract Pseudo class
*------------------------------------------------------------*/

#ifndef PSEUDO_H_
#define PSEUDO_H_

#include <cassert>
#include "SimulationBox.h"
#include "PolymerChain.h"

class Pseudo
{
protected:
    SimulationBox *sb;
    PolymerChain *pc;

    int n_complex_grid;
    double *expf, *expf_half;

    void set_exp_factor(
        std::array<int,3> nx, std::array<double,3> dx, double ds);
public:
    Pseudo(SimulationBox *sb, PolymerChain *pc);
    virtual ~Pseudo();

    virtual void find_phi(
        double *phia,  double *phib,
        double *q1_init, double *q2_init,
        double *wa, double *wb, double &QQ) = 0;

    virtual void get_partition(
        double *q1_out,
        double *q2_out, int n) = 0;

    virtual void update(){
        set_exp_factor(sb->get_nx(), sb->get_dx(), pc->get_ds());
    }

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
        assert(len_pa == sb->get_n_grid());
        assert(len_pb == sb->get_n_grid());
        assert(len_q1 == sb->get_n_grid());
        assert(len_q2 == sb->get_n_grid());
        assert(len_wa == sb->get_n_grid());
        assert(len_wb == sb->get_n_grid());
        find_phi(phia, phib, q1_init, q2_init, wa, wb, QQ);
    }
};
#endif

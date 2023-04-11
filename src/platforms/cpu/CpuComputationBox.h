/*-------------------------------------------------------------
* This is an CpuComputationBox class.
*--------------------------------------------------------------*/
#ifndef CPU_COMPUTATION_BOX_H_
#define CPU_COMPUTATION_BOX_H_

#include <array>
#include <vector>
#include <cassert>

#include "Exception.h"
#include "ComputationBox.h"

class CpuComputationBox : public ComputationBox
{
public:
    CpuComputationBox(std::vector<int> nx, std::vector<double> lx) : ComputationBox(nx, lx) {};
    virtual ~CpuComputationBox() {};

    // methods with device array
    double integral_device(double *g) override
    {
        return ComputationBox::integral(g);
    };
    double inner_product_device(double *g, double *h) override
    {
        return ComputationBox::inner_product(g ,h);
    };
    double inner_product_inverse_weight_device(double *g, double *h, double *w) override
    {
        return ComputationBox::inner_product_inverse_weight(g, h, w);
    };
    double multi_inner_product_device(int n_comp, double *g, double *h) override
    {
        return ComputationBox::multi_inner_product(n_comp, g, h);
    };
    void zero_mean_device(double *g) override
    {
        ComputationBox::zero_mean(g);
    };
};
#endif

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
    CpuComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr)
        : ComputationBox(nx, lx, bc, mask) {};
    virtual ~CpuComputationBox() {};

    // Methods with device array
    double integral_device(const double *g) override
    {
        return ComputationBox::integral(g);
    };
    double inner_product_device(const double *g, const double *h) override
    {
        return ComputationBox::inner_product(g ,h);
    };
    double inner_product_inverse_weight_device(const double *g, const double *h, const double *w) override
    {
        return ComputationBox::inner_product_inverse_weight(g, h, w);
    };
    double multi_inner_product_device(int n_comp, const double *g, const double *h) override
    {
        return ComputationBox::multi_inner_product(n_comp, g, h);
    };
    void zero_mean_device(double *g) override
    {
        ComputationBox::zero_mean(g);
    };
};
#endif

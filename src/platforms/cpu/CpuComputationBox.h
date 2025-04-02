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

    // // Methods with device array
    // T integral_device(const T *g) override
    // {
    //     return ComputationBox::integral(g);
    // };
    // T inner_product_device(const T *g, const T *h) override
    // {
    //     return ComputationBox::inner_product(g ,h);
    // };
    // T inner_product_inverse_weight_device(const T *g, const T *h, const T *w) override
    // {
    //     return ComputationBox::inner_product_inverse_weight(g, h, w);
    // };
    // T multi_inner_product_device(int n_comp, const T *g, const T *h) override
    // {
    //     return ComputationBox::multi_inner_product(n_comp, g, h);
    // };
    // void zero_mean_device(T *g) override
    // {
    //     ComputationBox::zero_mean(g);
    // };
};
#endif

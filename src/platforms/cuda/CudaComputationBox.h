/*-------------------------------------------------------------
* This is a derived CudaComputationBox class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <vector>
#include "ComputationBox.h"

class CudaComputationBox : public ComputationBox
{
private:

    // temporal storage for reduction in integral_gpu
    double *sum, *d_sum; 
    // temporal storage for mutiple_inner_product_gpu
    double *d_multiple;
    // dV for GPU
    double *d_dv; 

    // temporal arrays
    double *d_g, *d_h, *d_w; 

    // variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    double *d_temp_storage = nullptr;
    double *d_sum_out;

    void initialize();
public:
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx);
    ~CudaComputationBox() override;

    // double integral(double *g) override;
    // double inner_product(double *g, double *h) override;
    // double inner_product_inverse_weight(double *g, double *h, double *w) override;
    // double multi_inner_product(int n_comp, double *g, double *h) override;
    // void zero_mean(double *g) override;

    double integral_gpu(double *d_g);
    double inner_product_gpu(double *d_g, double *d_h);
    double inner_product_inverse_weight_gpu(double *d_g, double *d_h, double *d_w);
    double mutiple_inner_product_gpu(int n_comp, double *d_g, double *d_h);
    void set_lx(std::vector<double> new_lx) override;
};
#endif

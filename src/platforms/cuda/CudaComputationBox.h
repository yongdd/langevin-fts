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

    void set_lx(std::vector<double> new_lx) override;

    // methods with device array
    double integral_device(double *d_g) override;
    double inner_product_device(double *d_g, double *d_h) override;
    double inner_product_inverse_weight_device(double *d_g, double *d_h, double *d_w) override;
    double multi_inner_product_device(int n_comp, double *d_g, double *d_h) override;
    void zero_mean_device(double *d_g) override;
};
#endif

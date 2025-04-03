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
    // Temporal storage for reduction in integral_gpu
    double *sum, *d_sum; 
    
    // Temporal storage for mutiple_inner_product_gpu
    double *d_multiple;
    // dV for GPU
    double *d_dv;

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    double *d_temp_storage = nullptr;
    double *d_sum_out;

    void initialize();
public:
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);
    ~CudaComputationBox() override;

    void set_lx(std::vector<double> new_lx) override;

    // Methods with device array
    double integral_device(const double *d_g);
    double inner_product_device(const double *d_g, const double *d_h);
    double inner_product_inverse_weight_device(const double *d_g, const double *d_h, const double *d_w);
    double multi_inner_product_device(int n_comp, const double *d_g, const double *d_h);
    void zero_mean_device(double *d_g);
};
#endif

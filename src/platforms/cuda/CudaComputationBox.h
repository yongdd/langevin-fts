/*-------------------------------------------------------------
* This is a derived CudaComputationBox class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <vector>
#include "ComputationBox.h"

template <typename T>
class CudaComputationBox : public ComputationBox<T>
{
private:
    // Temporal storage for reduction in integral_gpu
    T *sum, *d_sum; 
    // Temporal storage for mutiple_inner_product_gpu
    T *d_multiple;
    // dV for GPU
    T *d_dv;

    // Temporal arrays
    T *d_g, *d_h, *d_w; 

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    T *d_temp_storage = nullptr;
    T *d_sum_out;

    void initialize();
public:
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);
    ~CudaComputationBox() override;

    void set_lx(std::vector<double> new_lx) override;

    // Methods with device array
    T integral_device(const T *d_g) override;
    T inner_product_device(const T *d_g, const T *d_h) override;
    T inner_product_inverse_weight_device(const T *d_g, const T *d_h, const T *d_w) override;
    T multi_inner_product_device(int n_comp, const T *d_g, const T *d_h) override;
    void zero_mean_device(T *d_g) override;
};
#endif

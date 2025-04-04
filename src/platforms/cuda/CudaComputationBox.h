/*-------------------------------------------------------------
* This is a derived CudaComputationBox class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <vector>

#include "CudaCommon.h"
#include "ComputationBox.h"

template <typename T>
class CudaComputationBox : public ComputationBox
{
private:
    // Temporal storage for reduction in integral_gpu
    T *sum;
    CuDeviceData<T> *d_sum; 
    
    // Temporal storage for mutiple_inner_product_gpu
    CuDeviceData<T> *d_multiple;
    // dV for GPU
    double *d_dv;

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    CuDeviceData<T> *d_temp_storage = nullptr;
    CuDeviceData<T> *d_sum_out;

    void initialize();
public:
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);
    ~CudaComputationBox() override;

    void set_lx(std::vector<double> new_lx) override;

    // Methods with device array
    T integral_device(const CuDeviceData<T> *d_g);
    T inner_product_device(const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h);
    T inner_product_inverse_weight_device(const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h, const CuDeviceData<T> *d_w);
    T multi_inner_product_device(int n_comp, const CuDeviceData<T> *d_g, const CuDeviceData<T> *d_h);
    void zero_mean_device(CuDeviceData<T> *d_g);
};
#endif

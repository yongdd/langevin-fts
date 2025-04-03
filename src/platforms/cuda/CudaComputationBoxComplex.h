/*-------------------------------------------------------------
* This is a derived CudaComputationBoxComplex class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_COMPLEX_H_
#define CUDA_SIMULATION_BOX_COMPLEX_H_

#include <vector>
#include <complex>

#include "CudaCommon.h"
#include "ComputationBox.h"

class CudaComputationBoxComplex : public ComputationBox
{
private:
    // Temporal storage for reduction in integral_gpu
    ftsComplex *sum, *d_sum; 
    // Temporal storage for mutiple_inner_product_gpu
    ftsComplex *d_multiple;
    // dV for GPU
    double *d_dv;

    // Variables for cub reduction sum
    size_t temp_storage_bytes = 0;
    ftsComplex *d_temp_storage = nullptr;
    ftsComplex *d_sum_out;

    void initialize();
public:
    CudaComputationBoxComplex(std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask=nullptr);
    ~CudaComputationBoxComplex() override;

    void set_lx(std::vector<double> new_lx) override;

    // Methods with device array
    std::complex<double> integral_device(const ftsComplex *d_g);
    std::complex<double> inner_product_device(const ftsComplex *d_g, const ftsComplex *d_h);
    std::complex<double> inner_product_inverse_weight_device(const ftsComplex *d_g, const ftsComplex *d_h, const ftsComplex *d_w);
    std::complex<double> multi_inner_product_device(int n_comp, const ftsComplex *d_g, const ftsComplex *d_h);
    void zero_mean_device(ftsComplex *d_g);
};
#endif

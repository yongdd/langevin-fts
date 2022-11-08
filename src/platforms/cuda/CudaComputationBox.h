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
    double *sum, *d_sum;   // temporal storage for reduction in integral_gpu
    double *d_multiple;    // temporal storage for mutiple_inner_product_gpu
    double *d_dv; // dV for GPU
    
    void initialize();
public:
    CudaComputationBox(std::vector<int> nx, std::vector<double> lx);
    ~CudaComputationBox() override;

    double integral_gpu(double *d_g);
    double inner_product_gpu(double *d_g, double *d_h);
    double mutiple_inner_product_gpu(int n_comp, double *d_g, double *d_h);
    void set_lx(std::vector<double> new_lx) override;
};
#endif

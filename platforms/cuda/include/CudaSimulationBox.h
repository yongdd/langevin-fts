/*-------------------------------------------------------------
* This is a derived CudaSimulationBox class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <vector>
#include "SimulationBox.h"

class CudaSimulationBox : public SimulationBox
{
private:
    double *sum, *d_sum;   // temporal storage for reduction in integral_gpu
    double *d_multiple;    // temporal storage for mutiple_inner_product_gpu
    double *d_dv; // dV for GPU
    
    void initialize();
public:
    CudaSimulationBox(std::vector<int> nx, std::vector<double> lx);
    ~CudaSimulationBox() override;

    double integral_gpu(double *d_g);
    double inner_product_gpu(double *d_g, double *d_h);
    double mutiple_inner_product_gpu(int n_comp, double *d_g, double *d_h);
    void set_lx(std::vector<double> new_lx) override;
};
#endif

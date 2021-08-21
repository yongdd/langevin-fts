/*-------------------------------------------------------------
* This is a derived CudaSimulationBox class
*------------------------------------------------------------*/

#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <array>
#include "SimulationBox.h"

class CudaSimulationBox : public SimulationBox
{
private:
    double *sum, *sum_d; // temporal storage for reduction in integral_gpu
    double *multiple_d;       // temporal storage for mutiple_inner_product_gpu
public:
    double *dv_d; // dV for GPU

    CudaSimulationBox(std::array<int,3> nx, std::array<double,3> lx);
    CudaSimulationBox(int *nx, double *lx) 
            : CudaSimulationBox({nx[0],nx[1],nx[2]},{lx[0],lx[1],lx[2]}) {};
    ~CudaSimulationBox() override;

    double integral_gpu(double *g_d);
    double inner_product_gpu(double *g_d, double *h_d);
    double mutiple_inner_product_gpu(int n_comp, double *g_d, double *h_d);

};
#endif

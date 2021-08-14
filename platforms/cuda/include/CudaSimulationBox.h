/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#ifndef CUDA_SIMULATION_BOX_H_
#define CUDA_SIMULATION_BOX_H_

#include <array>

class CudaSimulationBox
{
private:

    int N_BLOCKS, N_THREADS;
    double *sum, *sum_d; // temporal storage for multi_dot_gpu
public:

    int nx[3];  // the number of grid in each direction
    int MM;  // the number of total grid
    double lx[3];  // length of the block copolymer in each direction (in units of aN^1/2)
    double dx[3];  // grid interval in each direction
    double *dv; // dV, simple integral weight,
    double *dv_d; // dV for GPU
    double volume; // volume of the system.

    CudaSimulationBox(std::array<int,3> nx, std::array<double,3> lx);
    CudaSimulationBox(int *nx, double *lx) 
            : CudaSimulationBox({nx[0],nx[1],nx[2]},{lx[0],lx[1],lx[2]}) {};
    ~CudaSimulationBox();

    double dot(double *g, double *h);
    double multi_dot(int n_comp, double *g, double *h);
    double multi_dot_gpu(int n_comp, double *g_d, double *h_d);
    void zero_mean(double *w);
};
#endif

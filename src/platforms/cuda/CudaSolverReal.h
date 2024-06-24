/*----------------------------------------------------------
* This class defines a class for real-space method
*-----------------------------------------------------------*/

#ifndef CUDA_SOLVER_REAL_H_
#define CUDA_SOLVER_REAL_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "FiniteDifference.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

__device__ int d_max_of_two(int x, int y);
__device__ int d_min_of_two(int x, int y);

__global__ void compute_crank_1d(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_q_out, const double *d_q_in, const int M);

__global__ void compute_crank_2d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_in, const int M);

__global__ void compute_crank_2d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

__global__ void compute_crank_3d_step_1(
    BoundaryCondition bc_xl, BoundaryCondition bc_xh,
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_xl, const double *d_xd, const double *d_xh, const int I,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J,
    const double *d_zl, const double *d_zd, const double *d_zh, const int K,
    double *d_q_out, const double *d_q_in, const int M);

__global__ void compute_crank_3d_step_2(
    BoundaryCondition bc_yl, BoundaryCondition bc_yh,
    const double *d_yl, const double *d_yd, const double *d_yh, const int J, const int K,
    double *d_q_out, const double *d_q_star, const double *d_q_in, const int M);

__global__ void compute_crank_3d_step_3(
    BoundaryCondition bc_zl, BoundaryCondition bc_zh,
    const double *d_zl, const double *d_zd, const double *d_zh, const int J, const int K,
    double *d_q_out, const double *d_q_dstar, const double *d_q_in, const int M);

__global__ void tridiagonal(
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_c_star,  const double *d_d, double *d_x,
    const int *d_offset, const int REPEAT,
    const int INTERVAL, const int M);

__global__  void tridiagonal_periodic(
    const double *d_xl, const double *d_xd, const double *d_xh,
    double *d_c_star, double *d_q_sparse, 
     const double *d_d, double *d_x,
    const int *d_offset, const int REPEAT,
    const int INTERVAL, const int M);

class CudaSolverReal : public CudaSolver
{
private:
    ComputationBox *cb;
    Molecules *molecules;

    std::string chain_model;
    bool reduce_gpu_memory_usage;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy
    
    // Trigonal matrix for x direction
    std::map<std::string, double*> d_xl[MAX_GPUS];
    std::map<std::string, double*> d_xd[MAX_GPUS];
    std::map<std::string, double*> d_xh[MAX_GPUS];

    // Trigonal matrix for y direction
    std::map<std::string, double*> d_yl[MAX_GPUS];
    std::map<std::string, double*> d_yd[MAX_GPUS];
    std::map<std::string, double*> d_yh[MAX_GPUS];

    // Trigonal matrix for z direction
    std::map<std::string, double*> d_zl[MAX_GPUS];
    std::map<std::string, double*> d_zd[MAX_GPUS];
    std::map<std::string, double*> d_zh[MAX_GPUS];

    // Arrays for tridiagonal computation
    double *d_q_star  [MAX_STREAMS];
    double *d_q_dstar [MAX_STREAMS];
    double *d_c_star  [MAX_STREAMS];
    double *d_q_sparse[MAX_STREAMS];
    double *d_temp    [MAX_STREAMS];

    // Offset 
    // For 3D
    int* d_offset_xy[MAX_GPUS];
    int* d_offset_yz[MAX_GPUS];
    int* d_offset_xz[MAX_GPUS];
    // For 2D
    int* d_offset_x[MAX_GPUS];
    int* d_offset_y[MAX_GPUS];
    // For 1D
    int* d_offset[MAX_GPUS];

    void advance_propagator_3d(
        std::vector<BoundaryCondition> bc,      
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type);
    void advance_propagator_2d(
        std::vector<BoundaryCondition> bc,
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type);
    void advance_propagator_1d(
        std::vector<BoundaryCondition> bc,
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out, std::string monomer_type);
public:

    CudaSolverReal(ComputationBox *cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_gpu_memory_usage);
    ~CudaSolverReal();

    void update_laplacian_operator() override;
    void update_dw(std::string device, std::map<std::string, const double*> d_w_input) override;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_propagator_continuous(
            const int GPU, const int STREAM,
            double *d_q_in, double *d_q_out,
            std::string monomer_type, double *d_q_mask) override;

    // void compute_single_segment_stress_fourier(const int GPU, double *d_q) override;
    void compute_single_segment_stress_continuous(
        const int GPU, const int STREAM,
        double *d_q_pair, double *d_segment_stress, std::string monomer_type) override;
};
#endif

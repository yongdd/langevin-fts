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

class CudaSolverReal : public CudaSolver
{
private:
    ComputationBox *cb;
    Molecules *molecules;

    std::string chain_model;
    bool reduce_gpu_memory_usage;

    // Two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy
    
    // Trigonal matrix for x direction
    std::map<std::string, double*> xl;
    std::map<std::string, double*> xd;
    std::map<std::string, double*> xh;

    // Trigonal matrix for y direction
    std::map<std::string, double*> yl;
    std::map<std::string, double*> yd;
    std::map<std::string, double*> yh;

    // Trigonal matrix for z direction
    std::map<std::string, double*> zl;
    std::map<std::string, double*> zd;
    std::map<std::string, double*> zh;

    int max_of_two(int x, int y);
    int min_of_two(int x, int y);

    void advance_propagator_3d(
        double *d_q_in, double *d_q_out, std::string monomer_type);
    void advance_propagator_2d(
        double *d_q_in, double *d_q_out, std::string monomer_type);
    void advance_propagator_1d(
        double *d_q_in, double *d_q_out, std::string monomer_type);
public:

    CudaSolverReal(ComputationBox *cb, Molecules *molecules, cudaStream_t streams[MAX_GPUS][2], bool reduce_gpu_memory_usage);
    ~CudaSolverReal();

    void update_laplacian_operator() override;
    void update_dw(std::string device, std::map<std::string, const double*> d_w_input) override;

    static void tridiagonal(
        const double *d_xl, const double *d_xd, const double *d_xh,
        double *d_x, const int OFFSET, const double *d_d, const int M);

    static void tridiagonal_periodic(
        const double *d_xl, const double *d_xd, const double *d_xh,
        double *d_x, const int OFFSET, const double *d_d, const int M);

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_one_propagator_continuous(const int GPU,
            double *d_q_in, double *d_q_out,
            std::string monomer_type, double *d_q_mask) override;

    // Advance two propagators by one contour step
    void advance_two_propagators_continuous(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask) override {};

    // Advance two propagators by one segment step in two GPUs
    void advance_two_propagators_continuous_two_gpus(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double **d_q_mask) override {};

    void compute_single_segment_stress_fourier(const int GPU, double *d_q) override;
    std::vector<double> compute_single_segment_stress_continuous(const int GPU, std::string monomer_type) override;
};
#endif

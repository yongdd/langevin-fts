/*----------------------------------------------------------
* This class defines a class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CUDA_SOLVER_PSEUDO_H_
#define CUDA_SOLVER_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

class CudaSolverPseudo : public CudaSolver
{
private:
    ComputationBox *cb;
    Molecules *molecules;

    std::string chain_model;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy

    // For pseudo-spectral: advance_propagator()
    double *d_q_unity; // All elements are 1 for initializing propagators
    cufftHandle plan_for_one[MAX_STREAMS], plan_bak_one[MAX_STREAMS];
    cufftHandle plan_for_two[MAX_STREAMS], plan_bak_two[MAX_STREAMS];

    double *d_q_step_1_one[MAX_STREAMS], *d_q_step_2_one[MAX_STREAMS];
    double *d_q_step_1_two[MAX_STREAMS], *d_q_step_2_two[MAX_STREAMS];

    ftsComplex *d_qk_in_1_one[MAX_STREAMS];
    ftsComplex *d_qk_in_2_one[MAX_STREAMS];
    ftsComplex *d_qk_in_1_two[MAX_STREAMS];
    ftsComplex *d_qk_in_2_two[MAX_STREAMS];

    // For stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];

    // Variables for cub reduction sum
    size_t temp_storage_bytes[MAX_STREAMS];
    double *d_temp_storage[MAX_STREAMS];
    double *d_stress_sum[MAX_STREAMS];
    double *d_stress_sum_out[MAX_STREAMS];
    double *d_q_multi[MAX_STREAMS];

public:
    // GPU arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // Boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // Boltzmann factor for the half bond

    CudaSolverPseudo(ComputationBox *cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_gpu_memory_usage);
    ~CudaSolverPseudo();

    void update_laplacian_operator() override;
    void update_dw(std::string device, std::map<std::string, const double*> w_input) override;

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_propagator_continuous(
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask) override;

    void compute_single_segment_stress_continuous(
        const int GPU, const int STREAM,
        double *d_q_pair, double *d_segment_stress,
        std::string monomer_type) override;

    //---------- Discrete chain model -------------
    // Advance one propagator by one segment step
    void advance_propagator_discrete(
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type, double *d_q_mask);

    // Advance propagator by half bond step
    void advance_propagator_discrete_half_bond_step(
        const int GPU, const int STREAM,
        double *d_q_in, double *d_q_out,
        std::string monomer_type);

    // Compute stress of single segment
    void compute_single_segment_stress_discrete(
        const int GPU, const int STREAM,
        double *d_q_pair, double *d_segment_stress, std::string monomer_type, bool is_half_bond_length);
};
#endif

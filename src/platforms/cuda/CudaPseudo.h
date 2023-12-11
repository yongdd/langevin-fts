/*----------------------------------------------------------
* This class defines a derived class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CUDA_PSEUDO_H_
#define CUDA_PSEUDO_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaCommon.h"

class CudaPseudo : public Pseudo
{
private:
    std::string chain_model;
    bool reduce_gpu_memory_usage;

    // Two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // For pseudo-spectral: advance_propagator()
    double *d_q_unity; // All elements are 1 for initializing propagators
    cufftHandle plan_for_one[MAX_GPUS], plan_bak_one[MAX_GPUS];
    cufftHandle plan_for_two[MAX_GPUS], plan_bak_two[MAX_GPUS];
    cufftHandle plan_for_four,          plan_bak_four;

    double *d_q_step_1_one[MAX_GPUS], *d_q_step_2_one[MAX_GPUS];
    double *d_q_step_1_two[MAX_GPUS], *d_q_step_2_two[MAX_GPUS];
    double *d_q_step_1_four;

    ftsComplex *d_qk_in_1_one[MAX_GPUS];
    ftsComplex *d_qk_in_2_one[MAX_GPUS];
    ftsComplex *d_qk_in_1_two[MAX_GPUS];
    ftsComplex *d_qk_in_2_two[MAX_GPUS];
    ftsComplex *d_qk_in_1_four;

    // q_mask to make impenetrable region for nano particles
    double *d_q_mask[MAX_GPUS];

    // For stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];
    double *d_q_multi[MAX_GPUS];

    // Variables for cub reduction sum
    size_t temp_storage_bytes[MAX_GPUS];
    double *d_temp_storage[MAX_GPUS];
    double *d_stress_sum[MAX_GPUS];
    double *d_stress_sum_out[MAX_GPUS];

public:
    // GPU arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // Boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // Boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];            // Boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half[MAX_GPUS];       // Boltzmann factor for the half segment

    CudaPseudo(ComputationBox *cb, Molecules *molecules, cudaStream_t streams[MAX_GPUS][2], bool reduce_gpu_memory_usage);
    ~CudaPseudo();
    void update_bond_function() override;

    void update_dw(std::string device, std::map<std::string, const double*> w_input);

    //---------- Continuous chain model -------------
    // Advance propagator by one contour step
    void advance_one_propagator_continuous(const int GPU,
            double *d_q_in, double *d_q_out,
            std::string monomer_type, double *d_q_mask);

    // Advance two propagators by one contour step
    void advance_two_propagators_continuous(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask);

    // Advance two propagators by one segment step in two GPUs
    void advance_two_propagators_continuous_two_gpus(
            double *d_q_in_1,  double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double **d_q_mask);

    //---------- Discrete chain model -------------
    // Advance one propagator by one segment step
    void advance_one_propagator_discrete(const int GPU,
            double *d_q_in, double *d_q_out, std::string monomer_type, double *d_q_mask);

    // Advance two propagators by one segment step
    void advance_two_propagators_discrete(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask);

    // Advance two propagators by one segment step
    void advance_two_propagators_discrete_without_copy(
            double *d_q_two_in, double *d_q_two_out,
            std::string monomer_type_1, std::string monomer_type_2,
            double *d_q_mask);

    // Advance two propagators by one segment step in two GPUs
    void advance_two_propagators_discrete_two_gpus(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            std::string monomer_type_1, std::string monomer_type_2,
            double **d_q_mask);

    // Advance propagator by half bond step
    void advance_propagator_discrete_half_bond_step(const int GPU, double *d_q_in, double *d_q_out, std::string monomer_type);

    // Compute stress of single segment
    void compute_single_segment_stress_fourier(const int GPU, double *d_q);
    std::vector<double> compute_single_segment_stress_continuous(const int GPU, std::string monomer_type);
    std::vector<double> compute_single_segment_stress_discrete(const int GPU, std::string monomer_type, bool is_half_bond_length);
};
#endif

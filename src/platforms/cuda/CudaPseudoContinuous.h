/*-------------------------------------------------------------
* This is a derived CudaPseudoContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_CONTINUOUS_H_
#define CUDA_PSEUDO_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "Pseudo.h"
#include "CudaCommon.h"
#include "Scheduler.h"

class CudaPseudoContinuous : public Pseudo
{
private:
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

    ftsComplex *d_qk_in_2_one[MAX_GPUS];
    ftsComplex *d_qk_in_1_two[MAX_GPUS];
    ftsComplex *d_qk_in_2_two[MAX_GPUS];
    ftsComplex *d_qk_in_1_four;

    // For stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];
    double *d_stress_q[MAX_GPUS][2];  // one for prev, the other for next
    double *d_q_multi[MAX_GPUS];

    // Variables for cub reduction sum
    size_t temp_storage_bytes[MAX_GPUS];
    double *d_temp_storage[MAX_GPUS];
    double *d_stress_sum[MAX_GPUS];
    double *d_stress_sum_out[MAX_GPUS];

    // Scheduler for propagator computation 
    Scheduler *sc;
    // The number of parallel streams
    const int N_SCHEDULER_STREAMS = 2; 
    // gpu memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> d_propagator; 
    // Map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // Temporary arrays for device_1, one for prev, the other for next
    double *d_propagator_device_1[2];
    // Check if computation of propagator is finished
    #ifndef NDEBUG  
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Total partition functions
    double *single_partitions; 
    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_superposed)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // gpu memory space to store concentration, key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_block_phi;
    // Temp array for concentration computation
    double *d_phi;

    // GPU arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // Boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // Boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];            // Boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half[MAX_GPUS];       // Boltzmann factor for the half segment

    // Advance one propagator by one contour step
    void advance_one_propagator(const int GPU,
            double *d_q_in, double *d_q_out,
            double *d_boltz_bond, double *d_boltz_bond_half,
            double *d_exp_dw, double *d_exp_dw_half);

    // Advance two propagators by one contour step
    void advance_two_propagators(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            double *d_boltz_bond_1, double *d_boltz_bond_2, 
            double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
            double *d_exp_dw_1, double *d_exp_dw_2,
            double *d_exp_dw_half_1, double *d_exp_dw_half_2);

    // Advance two propagators by one segment step in two GPUs
    void advance_two_propagators_two_gpus(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            double *d_boltz_bond_1, double *d_boltz_bond_2, 
            double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
            double *d_exp_dw_1, double *d_exp_dw_2,
            double *d_exp_dw_half_1, double *d_exp_dw_half_2);

    // Calculate concentration of one block
    void calculate_phi_one_block(double *d_phi, double **d_q_1, double **d_q_2, const int N, const int N_OFFSET, const int N_ORIGINAL);

    // Compute statistics with inputs from selected device arrays
    void compute_statistics(
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init, std::string device);
public:

    CudaPseudoContinuous(ComputationBox *cb, Molecules *pc);
    ~CudaPseudoContinuous();

    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init) override
    {
        compute_statistics(w_input, q_init, "cpu");
    };
    void compute_statistics_device(
        std::map<std::string, const double*> d_w_input,
        std::map<std::string, const double*> d_q_init) override
    {
        compute_statistics(d_w_input, d_q_init, "gpu");
    };
    double get_total_partition(int polymer) override;
    void get_total_concentration(std::string monomer_type, double *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, double *phi) override;
    void get_block_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;
};

#endif

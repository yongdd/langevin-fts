/*-------------------------------------------------------------
This is a derived CudaPseudoReduceMemoryContinuous class

GPU memory usage is reduced by storing propagators in main memory.
In the GPU memory, array space that can store only two steps of propagator is allocated.
There are two streams. One is responsible for data transfers between CPU and GPU, another is responsible
for kernel executions. Overlapping of kernel execution and data transfers is utilized so that 
they can be simultaneously executed. As a result, data transfer time can be hided. For more explanation,
please see the supporting information of [Macromolecules 2021, 54, 24, 11304].
*------------------------------------------------------------*/

#ifndef CUDA_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_
#define CUDA_PSEUDO_REDUCE_MEMORY_CONTINUOUS_H_

#include <array>
#include <cufft.h>

#include "ComputationBox.h"
#include "PolymerChain.h"
#include "Mixture.h"
#include "Pseudo.h"
#include "CudaCommon.h"
#include "Scheduler.h"

class CudaPseudoReduceMemoryContinuous : public Pseudo
{
private:

    // two streams for each gpu
    cudaStream_t streams[MAX_GPUS][2]; // one for kernel execution, the other for memcpy

    // for pseudo-spectral: advance_one propagator()
    double *d_q_unity; // all elements are 1 for initializing propagators
    cufftHandle plan_for_one[MAX_GPUS], plan_bak_one[MAX_GPUS];
    cufftHandle plan_for_two[MAX_GPUS], plan_bak_two[MAX_GPUS];

    double *d_q_step_1_one[MAX_GPUS], *d_q_step_2_one[MAX_GPUS];
    double *d_q_step_1_two[MAX_GPUS];

    ftsComplex *d_qk_in_2_one[MAX_GPUS];
    ftsComplex *d_qk_in_1_two[MAX_GPUS];

    double *d_q_one[MAX_GPUS][2];     // one for prev, the other for next
    double *d_propagator_sub_dep[2];  // one for prev, the other for next

    // for concentration computation
    double *d_q_block_v[2];    // one for prev, the other for next
    double *d_q_block_u[2];    // one for prev, the other for next
    double *d_phi;

    // for stress calculation: compute_stress()
    double *d_fourier_basis_x[MAX_GPUS];
    double *d_fourier_basis_y[MAX_GPUS];
    double *d_fourier_basis_z[MAX_GPUS];
    double *d_stress_q[MAX_GPUS][2];  // one for prev, the other for next
    double *d_q_multi[MAX_GPUS];

    // variables for cub reduction sum
    size_t temp_storage_bytes[MAX_GPUS];
    double *d_temp_storage[MAX_GPUS];
    double *d_stress_sum[MAX_GPUS];
    double *d_stress_sum_out[MAX_GPUS];

    // scheduler for propagator computation 
    Scheduler *sc;
    // the number of parallel streams
    const int N_SCHEDULER_STREAMS = 2;
    // host pinned memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, double **> propagator;
    // map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // total partition function
    double *single_partitions; 
    // remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_superposed)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // host pinned space to store concentration, key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> block_phi;

    // gpu arrays for pseudo-spectral
    std::map<std::string, double*> d_boltz_bond[MAX_GPUS];        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half[MAX_GPUS];   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw[MAX_GPUS];            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half[MAX_GPUS];       // boltzmann factor for the half segment

    // advance one propagator by one contour step
    void advance_one_propagator(const int GPU,
            double *d_q_in, double *d_q_out,
            double *d_boltz_bond, double *d_boltz_bond_half,
            double *d_exp_dw, double *d_exp_dw_half);

    // advance two propagators by one segment step in two GPUs
    void advance_two_propagators_two_gpus(double *d_q_in_1, double *d_q_in_2,
            double *d_q_out_1, double *d_q_out_2,
            double *d_boltz_bond_1, double *d_boltz_bond_2, 
            double *d_boltz_bond_half_1, double *d_boltz_bond_half_2,         
            double *d_exp_dw_1, double *d_exp_dw_2,
            double *d_exp_dw_half_1, double *d_exp_dw_half_2);

    // calculate concentration of one block
    void calculate_phi_one_block(double *phi, double **q_1, double **q_2, const int N, const int N_OFFSET, const int N_ORIGINAL, const double NORM);

    // compute statistics with inputs from selected device arrays
    void compute_statistics(
        std::map<std::string, const double*> w_input,
        std::map<std::string, const double*> q_init, std::string device);
public:

    CudaPseudoReduceMemoryContinuous(ComputationBox *cb, Mixture *pc);
    ~CudaPseudoReduceMemoryContinuous();

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

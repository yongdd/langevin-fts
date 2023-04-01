/*-------------------------------------------------------------
This is a derived CudaPseudoReduceMemoryContinuous class

GPU memory usage is reduced by storing propagators in main memory.
In the GPU memory, array space that can store only two steps of propagator is allocated.
There are three streams. One is responsible for data transfer between CPU and GPU, another is responsible
for the compute_statistics() using single batched cufft, and the other is responsible for compute_stress()
using double batched cufft. Overlapping of kernel execution and data transfers is utilized so that 
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
    // for pseudo-spectral: one_step()
    double *d_q_unity; // all elements are 1 for initializing propagators
    cufftHandle plan_for, plan_bak;
    double *d_q_step_1, *d_q_step_2;
    ftsComplex *d_qk_in;
    double *d_q[2];                   // one for prev, the other for next
    double *d_propagator_sub_dep[2];  // one for prev, the other for next

    // for concentration computation
    double *d_q_block_v[2];    // one for prev, the other for next
    double *d_q_block_u[2];    // one for prev, the other for next
    double *d_phi;

    // for stress calculation: compute_stress()
    cufftHandle plan_for_two;
    ftsComplex *d_two_qk_in;
    double *d_q_two_partition[2];    // one for prev, the other for next

    double *d_fourier_basis_x;
    double *d_fourier_basis_y;
    double *d_fourier_basis_z;
    double *d_q_multi, *d_stress_sum;

    // remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_superposed)
    std::vector<std::tuple<int, double *, double *, int>> single_partition_segment;

    // total partition functions for each polymer
    double* single_partitions;

    // three streams for overlapping kernel execution and data transfers
    cudaStream_t *streams;

    // key: (dep) + monomer_type, value: partition function
    std::map<std::string, double *> propagator;

    // check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // key: (polymer id, dep_v, dep_u) (assert(dep_v <= dep_u)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, double *> d_block_phi;

    std::map<std::string, double*> d_boltz_bond;        // boltzmann factor for the single bond
    std::map<std::string, double*> d_boltz_bond_half;   // boltzmann factor for the half bond
    std::map<std::string, double*> d_exp_dw;            // boltzmann factor for the single segment
    std::map<std::string, double*> d_exp_dw_half;       // boltzmann factor for the half segment

    void one_step(double *d_q_in, double *d_q_out,
                  double *d_boltz_bond, double *d_boltz_bond_half,
                  double *d_exp_dw, double *d_exp_dw_half);

    void calculate_phi_one_block(double *d_phi, double *q_1, double *q_2, const int N, const int N_OFFSET, const int N_ORIGINAL);
public:

    CudaPseudoReduceMemoryContinuous(ComputationBox *cb, Mixture *pc);
    ~CudaPseudoReduceMemoryContinuous();

    void update_bond_function() override;
    void compute_statistics(
        std::map<std::string, double*> w_input,
        std::map<std::string, double*> q_init) override;
    double get_total_partition(int polymer) override;
    void get_monomer_concentration(std::string monomer_type, double *phi) override;
    void get_polymer_concentration(int polymer, double *phi) override;
    std::vector<double> compute_stress() override;
    void get_chain_propagator(double *q_out, int polymer, int v, int u, int n) override;
};

#endif

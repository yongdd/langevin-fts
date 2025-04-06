/*-------------------------------------------------------------
This is a derived CudaComputationReduceMemoryContinuous class

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
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputation.h"
#include "CudaCommon.h"
#include "CudaSolver.h"
#include "Scheduler.h"

template <typename T>
class CudaComputationReduceMemoryContinuous : public PropagatorComputation<T>
{
private:
    // Pseudo-spectral PDE solver
    CudaSolver<T> *propagator_solver;
    std::string method;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy

    // For pseudo-spectral: advance_one propagator()
    CuDeviceData<T> *d_q_one[MAX_STREAMS][2];               // one for prev, the other for next
    CuDeviceData<T> *d_propagator_sub_dep[MAX_STREAMS][2];  // one for prev, the other for next

    // All elements are 1 for initializing propagators
    CuDeviceData<T> *d_q_unity;

    // q_mask to make impenetrable region for nano particles
    double *d_q_mask;

    // For concentration computation
    CuDeviceData<T> *d_q_block_v[2];    // one for prev, the other for next
    CuDeviceData<T> *d_q_block_u[2];    // one for prev, the other for next
    CuDeviceData<T> *d_phi;

    // For stress calculation: compute_stress()
    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];  // one for prev, the other for next

    // Scheduler for propagator computation 
    Scheduler *sc;

    // Host pinned memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, T **> propagator;
    // Map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_repeated)
    std::vector<std::tuple<int, T *, T *, int>> single_partition_segment;

    // Host pinned space to store concentration, key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, T *> phi_block;

    // Solvent concentrations
    std::vector<T *> phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(T *phi, T **q_1, T **q_2, const int N_RIGHT, const int N_LEFT, const T NORM);

public:

    CudaComputationReduceMemoryContinuous(ComputationBox<T>* cb, Molecules *pc, PropagatorComputationOptimizer *propagator_computation_optimizer, std::string method);
    ~CudaComputationReduceMemoryContinuous();

    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    void compute_concentrations() override;

    // Compute statistics with inputs
    void compute_statistics(
        std::map<std::string, const T*> w_input,
        std::map<std::string, const T*> q_init = {}) override;

    void compute_stress() override;
    T get_total_partition(int polymer) override;
    void get_chain_propagator(T *q_out, int polymer, int v, int u, int n) override;

    // Canonical ensemble
    void get_total_concentration(std::string monomer_type, T *phi) override;
    void get_total_concentration(int polymer, std::string monomer_type, T *phi) override;
    void get_block_concentration(int polymer, T *phi) override;

    T get_solvent_partition(int s) override;
    void get_solvent_concentration(int s, T *phi) override;

    // Grand canonical ensemble
    void get_total_concentration_gce(double fugacity, int polymer, std::string monomer_type, T *phi) override;

    // For tests
    bool check_total_partition() override;
};

#endif

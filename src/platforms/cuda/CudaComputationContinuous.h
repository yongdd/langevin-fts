/*-------------------------------------------------------------
* This is a derived CudaComputationContinuous class
*------------------------------------------------------------*/

#ifndef CUDA_COMPUTATION_CONTINUOUS_H_
#define CUDA_COMPUTATION_CONTINUOUS_H_

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
class CudaComputationContinuous : public PropagatorComputation<T>
{
private:
    // Pseudo-spectral PDE solver
    CudaSolver<T> *propagator_solver;
    std::string method;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy

    // All elements are 1 for initializing propagators
    CuDeviceData<T> *d_q_unity; 

    // q_mask to make impenetrable region for nano particles
    double *d_q_mask;

    // One for prev, the other for next
    CuDeviceData<T> *d_q_pair[MAX_STREAMS][2];

    // Scheduler for propagator computation 
    Scheduler *sc;
    // gpu memory space to store propagator, key: (dep) + monomer_type, value: propagator
    std::map<std::string, CuDeviceData<T> **> d_propagator; 
    // Map for deallocation of d_propagator
    std::map<std::string, int> propagator_size;
    // Check if computation of propagator is finished
    #ifndef NDEBUG  
    std::map<std::string, bool *> propagator_finished;
    #endif

    // Remember one segment for each polymer chain to compute total partition function
    // (polymer id, propagator forward, propagator backward, n_repeated)
    std::vector<std::tuple<int, CuDeviceData<T> *, CuDeviceData<T> *, int>> single_partition_segment;

    // gpu memory space to store concentration, key: (polymer id, key_left, key_right) (assert(key_left <= key_right)), value: concentration
    std::map<std::tuple<int, std::string, std::string>, CuDeviceData<T> *> d_phi_block;
    // Temp array for concentration computation
    CuDeviceData<T> *d_phi;

    // Solvent concentrations
    std::vector<CuDeviceData<T> *> d_phi_solvent;

    // Calculate concentration of one block
    void calculate_phi_one_block(CuDeviceData<T> *d_phi, CuDeviceData<T> **d_q_1, CuDeviceData<T> **d_q_2, const int N_LEFT, const int N_RIGHT);
    
public:

    CudaComputationContinuous(ComputationBox<T>* cb, Molecules *pc, PropagatorComputationOptimizer *propagator_computation_optimizer, std::string method);
    ~CudaComputationContinuous();

    void update_laplacian_operator() override;

    void compute_propagators(
        std::map<std::string, const T*> w_block,
        std::map<std::string, const T*> q_init = {}) override;

    void advance_propagator_single_segment(T* q_init, T *q_out, std::string monomer_type) override;

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

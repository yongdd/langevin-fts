/*----------------------------------------------------------
* This class defines a class for pseudo-spectral method
*-----------------------------------------------------------*/

#ifndef CUDA_SOLVER_PSEUDO_DISCRETE_H_
#define CUDA_SOLVER_PSEUDO_DISCRETE_H_

#include <string>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "Pseudo.h"
#include "CudaSolver.h"
#include "CudaCommon.h"

template <typename T>
class CudaSolverPseudoDiscrete : public CudaSolver<T>
{
private:
    ComputationBox<T>* cb;
    Molecules *molecules;
    Pseudo<T> *pseudo;
    std::string chain_model;

    // The number of parallel streams for propagator computation
    int n_streams;

    // Two streams for each gpu
    cudaStream_t streams[MAX_STREAMS][2]; // one for kernel execution, the other for memcpy

    // For pseudo-spectral: advance_propagator()
    T *d_q_unity; // All elements are 1 for initializing propagators
    cufftHandle plan_for_one[MAX_STREAMS], plan_bak_one[MAX_STREAMS];
    cufftHandle plan_for_two[MAX_STREAMS], plan_bak_two[MAX_STREAMS];

    T *d_q_step_1_one[MAX_STREAMS], *d_q_step_2_one[MAX_STREAMS];
    T *d_q_step_1_two[MAX_STREAMS], *d_q_step_2_two[MAX_STREAMS];

    cuDoubleComplex *d_qk_in_1_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_one[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_1_two[MAX_STREAMS];
    cuDoubleComplex *d_qk_in_2_two[MAX_STREAMS];

    // Variables for cub reduction sum
    size_t temp_storage_bytes[MAX_STREAMS];
    CuDeviceData<T> *d_temp_storage[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum[MAX_STREAMS];
    CuDeviceData<T> *d_stress_sum_out[MAX_STREAMS];
    CuDeviceData<T> *d_q_multi[MAX_STREAMS];

public:
    CudaSolverPseudoDiscrete(ComputationBox<T>* cb, Molecules *molecules, int n_streams, cudaStream_t streams[MAX_STREAMS][2], bool reduce_gpu_memory_usage);
    ~CudaSolverPseudoDiscrete();

    void update_laplacian_operator() override;
    void update_dw(std::string device, std::map<std::string, const T*> w_input) override;

    //---------- Discrete chain model -------------
    // Advance propagator by one contour step
    void advance_propagator(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type, double *d_q_mask) override;

    // Advance propagator by half bond step
    void advance_propagator_half_bond_step(
        const int STREAM,
        CuDeviceData<T> *d_q_in, CuDeviceData<T> *d_q_out,
        std::string monomer_type) override;

    // Compute stress of single segment
    void compute_single_segment_stress(
        const int STREAM,
        CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
        std::string monomer_type, bool is_half_bond_length) override;
};
#endif

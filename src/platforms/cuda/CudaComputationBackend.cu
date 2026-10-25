/**
 * @file CudaComputationBackend.cu
 * @brief CUDA implementation of ComputationBackend.
 */

#include <algorithm>
#include <complex>
#include <utility>
#include <cuComplex.h>

#include "CudaComputationBackend.h"
#include "NumericalMethods.h"
#include "CudaSolverCNADI.h"
#include "CudaSolverPseudoDiscrete.h"
#include "CudaSolverPseudoRK2.h"
#include "CudaSolverPseudoRQM4.h"

namespace {

/// Host scalar to the device scalar type.
template <typename T>
CuDeviceData<T> to_device_scalar(T value)
{
    if constexpr (std::is_same_v<T, double>)
        return value;
    else
        return stdToCuDoubleComplex(value);
}

/// Segment pairs added to a concentration in one launch (passed as kernel parameters)
constexpr int MAX_PAIRS_PER_LAUNCH = 32;

template <typename TD>
struct SegmentProducts
{
    const TD* q_1[MAX_PAIRS_PER_LAUNCH];
    const TD* q_2[MAX_PAIRS_PER_LAUNCH];
    double weight[MAX_PAIRS_PER_LAUNCH];
    int n;
};

// phi += Σ_k weight_k q_1k q_2k in the order of the pairs, with the same arithmetic as one
// ker_add_multi per pair, but phi is read and written once per launch
__global__ void ker_add_segment_products(double* phi, SegmentProducts<double> p, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        double acc = phi[i];
        for (int k = 0; k < p.n; ++k)
            acc += p.weight[k] * p.q_1[k][i] * p.q_2[k][i];
        phi[i] = acc;
        i += blockDim.x * gridDim.x;
    }
}

__global__ void ker_add_segment_products(cuDoubleComplex* phi, SegmentProducts<cuDoubleComplex> p, const int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        cuDoubleComplex acc = phi[i];
        for (int k = 0; k < p.n; ++k)
        {
            const cuDoubleComplex a = p.q_1[k][i];
            const cuDoubleComplex b = p.q_2[k][i];
            const double w = p.weight[k];
            const double tx = w * (a.x * b.x - a.y * b.y);
            const double ty = w * (a.x * b.y + a.y * b.x);
            acc.x += tx;
            acc.y += ty;
        }
        phi[i] = acc;
        i += blockDim.x * gridDim.x;
    }
}

}  // namespace

template <typename T>
CudaComputationBackend<T>::CudaComputationBackend(
    ComputationBox<T>* cb_in, Molecules* molecules,
    const NumericalMethodInfo& method_info,
    SpaceGroup* space_group, int max_n_streams)
    : cb(nullptr), space_group(space_group),
      solver_discrete(nullptr), n_streams(0), n_basis(0), force_off_diagonal_stress(false)
{
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        streams[i][0] = nullptr;
        streams[i][1] = nullptr;
        kernel_done[i] = nullptr;
        copy_done[i] = nullptr;
    }

    // Everything allocated so far is released if a step below throws
    try
    {
        cb = dynamic_cast<CudaComputationBox<T>*>(cb_in);
        if (cb == nullptr)
            throw_with_line_number("The CUDA backend requires a CudaComputationBox.");

        const int M = cb->get_total_grid();
        n_basis = cb->get_n_basis();

        // Index maps between the full grid and the reduced basis
        if (space_group != nullptr)
        {
            const int n_reduced = space_group->get_n_reduced_basis();
            d_reduced_basis_indices.allocate(n_reduced);
            CUDA_CHECK(cudaMemcpy(d_reduced_basis_indices, space_group->get_reduced_basis_indices().data(),
                                       sizeof(int)*n_reduced, cudaMemcpyHostToDevice));
            d_full_to_reduced_map.allocate(M);
            CUDA_CHECK(cudaMemcpy(d_full_to_reduced_map, space_group->get_full_to_reduced_map().data(),
                                       sizeof(int)*M, cudaMemcpyHostToDevice));
        }

        n_streams = std::min({this->n_streams_from_env(/*default_n_streams=*/MAX_STREAMS), MAX_STREAMS, max_n_streams});
        for (int i = 0; i < n_streams; i++)
        {
            CUDA_CHECK(count_cuda_stream(cudaStreamCreate(&streams[i][0])));  // kernels
            CUDA_CHECK(count_cuda_stream(cudaStreamCreate(&streams[i][1])));  // copies
            CUDA_CHECK(count_cuda_event(cudaEventCreateWithFlags(&kernel_done[i], cudaEventDisableTiming)));
            CUDA_CHECK(count_cuda_event(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming)));
        }

        switch (method_info.id)
        {
        case NumericalMethodId::Discrete:
        {
            auto discrete = std::make_unique<CudaSolverPseudoDiscrete<T>>(cb, molecules, n_streams, streams, false);
            solver_discrete = discrete.get();
            solver = std::move(discrete);
            break;
        }
        case NumericalMethodId::RQM4:
            solver = std::make_unique<CudaSolverPseudoRQM4<T>>(cb, molecules, n_streams, streams, false);
            break;
        case NumericalMethodId::RK2:
            solver = std::make_unique<CudaSolverPseudoRK2<T>>(cb, molecules, n_streams, streams, false);
            break;
        case NumericalMethodId::CNADI2:
            if constexpr (std::is_same_v<T, double>)
                solver = std::make_unique<CudaSolverCNADI>(cb, molecules, n_streams, streams, false);
            else
                throw_with_line_number("The cn-adi2 method is only available for real fields.");
            break;
        }

        // Ones for free chain ends
        std::vector<T> unity(M, T(1.0));
        d_q_unity.allocate(M);
        CUDA_CHECK(cudaMemcpy(d_q_unity, unity.data(), sizeof(T)*M, cudaMemcpyHostToDevice));

        // Mask (the box stores it in the reduced basis when a space group is set)
        if (cb->get_mask() != nullptr)
        {
            d_q_mask.allocate(n_basis);
            CUDA_CHECK(cudaMemcpy(d_q_mask, cb->get_mask(), sizeof(double)*n_basis, cudaMemcpyHostToDevice));
        }

        d_sum.allocate(M);
        for (int i = 0; i < n_streams; i++)
        {
            d_q_pair[i][0].allocate(2*M);
            d_q_pair[i][1].allocate(2*M);
        }
        for (int i = 0; i < MAX_STREAMS; i++)
        {
            d_segment_stress[i].allocate(6);
            d_block_stress[i].allocate(6);
            h_block_stress[i].allocate(6);
        }

        solver->update_laplacian_operator();

        // The solver expands/reduces around its FFTs when a space group is set
        if (space_group != nullptr)
            solver->set_space_group(space_group, d_reduced_basis_indices, d_full_to_reduced_map,
                                    space_group->get_n_reduced_basis());
    }
    catch (...)
    {
        release();
        throw;
    }
}

template <typename T>
CudaComputationBackend<T>::~CudaComputationBackend()
{
    release();
}

template <typename T>
void CudaComputationBackend<T>::release()
{
    // The solver first: its cuFFT plans run on the streams destroyed below.
    // The device buffers are owning members (released by the destructor, or
    // here for a partly built object).
    solver.reset();
    solver_discrete = nullptr;

    d_q_unity.release();
    d_q_mask.release();
    d_reduced_basis_indices.release();
    d_full_to_reduced_map.release();
    d_sum.release();
    d_exp_dw_reduced.clear();

    for (int i = 0; i < MAX_STREAMS; i++)
    {
        d_q_pair[i][0].release();
        d_q_pair[i][1].release();
        d_segment_stress[i].release();
        d_block_stress[i].release();
        h_block_stress[i].release();
        if (kernel_done[i] != nullptr)
            release_cuda_event(kernel_done[i]);
        if (copy_done[i] != nullptr)
            release_cuda_event(copy_done[i]);
        if (streams[i][0] != nullptr)
            release_cuda_stream(streams[i][0]);
        if (streams[i][1] != nullptr)
            release_cuda_stream(streams[i][1]);
        kernel_done[i] = nullptr;
        copy_done[i] = nullptr;
        streams[i][0] = nullptr;
        streams[i][1] = nullptr;
    }
}

template <typename T>
void CudaComputationBackend<T>::synchronize()
{
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==================== Memory ====================

// The propagator and checkpoint arrays are raw handles owned by the
// computations; they are counted like CudaBuffer allocations, so the tests'
// live counters and failure hook cover them as well.
template <typename T>
BackendArray<T>* CudaComputationBackend<T>::allocate()
{
    const size_t bytes = sizeof(T)*n_basis;
    void* d_a = counted_cuda_allocate<CudaDeviceMemory>(bytes);
    const cudaError_t err = cudaMemset(d_a, 0, bytes);
    if (err != cudaSuccess)
    {
        counted_cuda_release<CudaDeviceMemory>(d_a, bytes);
        CUDA_CHECK(err);
    }
    return wrap(static_cast<CuDeviceData<T>*>(d_a));
}

template <typename T>
void CudaComputationBackend<T>::deallocate(BackendArray<T>* a)
{
    counted_cuda_release<CudaDeviceMemory>(raw(a), sizeof(T)*n_basis);
}

template <typename T>
void CudaComputationBackend<T>::copy_from_host(int stream, BackendArray<T>* dst, const T* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), src, sizeof(T)*n_basis, cudaMemcpyHostToDevice, streams[stream][0]));
}

template <typename T>
void CudaComputationBackend<T>::copy_to_host(T* dst, BackendArray<T>* src)
{
    CUDA_CHECK(cudaMemcpy(dst, raw(src), sizeof(T)*n_basis, cudaMemcpyDeviceToHost));
}

template <typename T>
CheckpointArray<T>* CudaComputationBackend<T>::allocate_checkpoint()
{
    void* h_a = counted_cuda_allocate<CudaPinnedMemory>(sizeof(T)*n_basis);
    return reinterpret_cast<CheckpointArray<T>*>(h_a);
}

template <typename T>
void CudaComputationBackend<T>::deallocate_checkpoint(CheckpointArray<T>* a)
{
    counted_cuda_release<CudaPinnedMemory>(raw(a), sizeof(T)*n_basis);
}

template <typename T>
void CudaComputationBackend<T>::store_checkpoint(int stream, CheckpointArray<T>* dst, BackendArray<T>* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), raw(src), sizeof(T)*n_basis, cudaMemcpyDeviceToHost, streams[stream][0]));
}

template <typename T>
void CudaComputationBackend<T>::load_checkpoint(int stream, BackendArray<T>* dst, CheckpointArray<T>* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), raw(src), sizeof(T)*n_basis, cudaMemcpyHostToDevice, streams[stream][0]));
}

// ==================== Solver ====================

template <typename T>
void CudaComputationBackend<T>::update_laplacian_operator()
{
    solver->update_laplacian_operator();
}

template <typename T>
void CudaComputationBackend<T>::update_dw(std::map<std::string, const T*> w_input)
{
    solver->update_dw("cpu", w_input);

    // The discrete solver keeps exp(-w ds) on the full grid unless its CrysFFT
    // path stores it in the reduced basis already
    if (solver_discrete != nullptr && space_group != nullptr)
    {
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        for (const auto& item : w_input)
        {
            const std::string& monomer_type = item.first;
            auto found = d_exp_dw_reduced.find(monomer_type);
            if (found == d_exp_dw_reduced.end())
            {
                // Enter the buffer only once it exists, so a failed allocation can be retried
                found = d_exp_dw_reduced.emplace(monomer_type, CudaDeviceBuffer<CuDeviceData<T>>(n_basis)).first;
            }
            CuDeviceData<T>* d_exp_dw_basis  = found->second.get();
            CuDeviceData<T>* d_exp_dw_solver = solver->d_exp_dw.at(0).at(monomer_type);
            if (solver_discrete->exp_dw_stored_reduced())
            {
                CUDA_CHECK(cudaMemcpy(d_exp_dw_basis, d_exp_dw_solver,
                                           sizeof(T)*n_basis, cudaMemcpyDeviceToDevice));
            }
            else
            {
                ker_reduce_to_basis<<<N_BLOCKS, N_THREADS>>>(
                    d_exp_dw_basis, d_exp_dw_solver, d_reduced_basis_indices, n_basis);
                CUDA_CHECK_LAUNCH();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

template <typename T>
void CudaComputationBackend<T>::set_force_off_diagonal_stress(bool force)
{
    force_off_diagonal_stress = force;
    solver->set_force_off_diagonal_stress(force);
}

template <typename T>
void CudaComputationBackend<T>::advance_propagator(
    int stream, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type, int ds_index)
{
    solver->advance_propagator(stream, raw(q_in), raw(q_out), monomer_type, d_q_mask, ds_index);
}

template <typename T>
void CudaComputationBackend<T>::advance_propagator_half_bond_step(
    int stream, BackendArray<T>* q_in, BackendArray<T>* q_out, const std::string& monomer_type)
{
    solver->advance_propagator_half_bond_step(stream, raw(q_in), raw(q_out), monomer_type);
}

template <typename T>
BackendArray<T>* CudaComputationBackend<T>::boltzmann_factor(const std::string& monomer_type)
{
    if (space_group != nullptr)
        return wrap(d_exp_dw_reduced.at(monomer_type));
    return wrap(solver->d_exp_dw.at(0).at(monomer_type));
}

// ==================== Propagator initial conditions ====================

template <typename T>
void CudaComputationBackend<T>::set_unity(int stream, BackendArray<T>* q)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(q), d_q_unity, sizeof(T)*n_basis,
                                    cudaMemcpyDeviceToDevice, streams[stream][0]));
}

template <typename T>
void CudaComputationBackend<T>::set_zero(int stream, BackendArray<T>* a)
{
    CUDA_CHECK(cudaMemsetAsync(raw(a), 0, sizeof(T)*n_basis, streams[stream][0]));
}

template <typename T>
void CudaComputationBackend<T>::copy(int stream, BackendArray<T>* dst, BackendArray<T>* src)
{
    CUDA_CHECK(cudaMemcpyAsync(raw(dst), raw(src), sizeof(T)*n_basis,
                                    cudaMemcpyDeviceToDevice, streams[stream][0]));
}

template <typename T>
void CudaComputationBackend<T>::junction_sum(
    int stream, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, int>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);

    CUDA_CHECK(cudaMemsetAsync(d_q, 0, sizeof(T)*n_basis, streams[stream][0]));
    for (const auto& [src, n_repeated] : sources)
    {
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(
            d_q, 1.0, d_q, n_repeated, raw(src), n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_product(
    int stream, BackendArray<T>* q, const std::vector<std::pair<BackendArray<T>*, int>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);

    CUDA_CHECK(cudaMemcpyAsync(d_q, d_q_unity, sizeof(T)*n_basis,
                                    cudaMemcpyDeviceToDevice, streams[stream][0]));
    for (const auto& [src, n_repeated] : sources)
    {
        // q_i^n as n multiplications
        for (int r = 0; r < n_repeated; r++)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(d_q, d_q, raw(src), 1.0, n_basis);
            CUDA_CHECK_LAUNCH();
        }
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_sum(
    int stream, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, int>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);
    // The stress pair buffer is free while propagators are computed; stage each checkpoint in it
    CuDeviceData<T>* d_stage = d_q_pair[stream][0];

    CUDA_CHECK(cudaMemsetAsync(d_q, 0, sizeof(T)*n_basis, streams[stream][0]));
    for (const auto& [src, n_repeated] : sources)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_stage, raw(src), sizeof(T)*n_basis,
                                        cudaMemcpyHostToDevice, streams[stream][0]));
        ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(
            d_q, 1.0, d_q, n_repeated, d_stage, n_basis);
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaComputationBackend<T>::junction_product(
    int stream, BackendArray<T>* q, const std::vector<std::pair<CheckpointArray<T>*, int>>& sources)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_q = raw(q);
    // The stress pair buffer is free while propagators are computed; stage each checkpoint in it
    CuDeviceData<T>* d_stage = d_q_pair[stream][0];

    CUDA_CHECK(cudaMemcpyAsync(d_q, d_q_unity, sizeof(T)*n_basis,
                                    cudaMemcpyDeviceToDevice, streams[stream][0]));
    for (const auto& [src, n_repeated] : sources)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_stage, raw(src), sizeof(T)*n_basis,
                                        cudaMemcpyHostToDevice, streams[stream][0]));
        for (int r = 0; r < n_repeated; r++)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(d_q, d_q, d_stage, 1.0, n_basis);
            CUDA_CHECK_LAUNCH();
        }
    }
}

template <typename T>
void CudaComputationBackend<T>::apply_mask(int stream, BackendArray<T>* q)
{
    if (d_q_mask.empty())
        return;
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(raw(q), raw(q), d_q_mask, 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaComputationBackend<T>::multiply_boltzmann_factor(int stream, BackendArray<T>* q, const std::string& monomer_type)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(
        raw(q), raw(q), raw(boltzmann_factor(monomer_type)), 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
}

// ==================== Statistics ====================

template <typename T>
T CudaComputationBackend<T>::inner_product(BackendArray<T>* a, BackendArray<T>* b)
{
    return cb->inner_product_device(raw(a), raw(b));
}

template <typename T>
T CudaComputationBackend<T>::inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w)
{
    return cb->inner_product_inverse_weight_device(raw(a), raw(b), raw(w));
}

template <typename T>
void CudaComputationBackend<T>::add_segment_products(
    int stream, BackendArray<T>* phi, const std::vector<SegmentPair<T>>& pairs)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    // Up to MAX_PAIRS_PER_LAUNCH pairs per launch, in order
    SegmentProducts<CuDeviceData<T>> chunk{};
    for (size_t first = 0; first < pairs.size(); first += MAX_PAIRS_PER_LAUNCH)
    {
        chunk.n = static_cast<int>(std::min<size_t>(MAX_PAIRS_PER_LAUNCH, pairs.size() - first));
        for (int k = 0; k < chunk.n; ++k)
        {
            chunk.q_1[k] = raw(pairs[first + k].q_1);
            chunk.q_2[k] = raw(pairs[first + k].q_2);
            chunk.weight[k] = pairs[first + k].weight;
        }
        ker_add_segment_products<<<N_BLOCKS, N_THREADS, 0, streams[stream][0]>>>(raw(phi), chunk, n_basis);
        CUDA_CHECK_LAUNCH();
    }
    CUDA_CHECK_LAUNCH();
}

template <typename T>
void CudaComputationBackend<T>::normalize_concentration(int stream, BackendArray<T>* phi, T norm, BackendArray<T>* divisor)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_phi = raw(phi);
    cudaStream_t s = streams[stream][0];
    if (divisor != nullptr)
        ker_divide<<<N_BLOCKS, N_THREADS, 0, s>>>(d_phi, d_phi, raw(divisor), 1.0, n_basis);
    CUDA_CHECK_LAUNCH();
    ker_lin_comb<<<N_BLOCKS, N_THREADS, 0, s>>>(d_phi, to_device_scalar(norm), d_phi, 0.0, d_phi, n_basis);
    CUDA_CHECK_LAUNCH();
}

template <typename T>
std::array<T, 6> CudaComputationBackend<T>::block_stress(
    int stream, const std::vector<SegmentPair<T>>& pairs, const std::string& monomer_type)
{
    std::array<T, 6> stress = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (pairs.empty())
        return stress;

    // Components the solver computes: 3D orthogonal 3, 3D non-orthogonal 6, 2D 3, 1D 1.
    // Forced off-diagonal stress (angle optimization from 90°) counts as non-orthogonal.
    const int DIM = cb->get_dim();
    const bool is_ortho = cb->is_orthogonal() && !force_off_diagonal_stress;
    const int N_STRESS_DIM = (DIM == 3) ? (is_ortho ? 3 : 6) : ((DIM == 2) ? 3 : 1);

    cudaStream_t s_kernel = streams[stream][0];
    cudaStream_t s_copy   = streams[stream][1];
    CuDeviceData<T>* d_segment = d_segment_stress[stream];
    CuDeviceData<T>* d_block   = d_block_stress[stream];
    CUDA_CHECK(cudaMemsetAsync(d_segment, 0, sizeof(T)*6, s_kernel));
    CUDA_CHECK(cudaMemsetAsync(d_block,   0, sizeof(T)*6, s_kernel));

    cudaEvent_t kernel_done_s = kernel_done[stream];
    cudaEvent_t copy_done_s   = copy_done[stream];

    // The pairs may still be computed on the kernel stream (reduce-memory
    // recomputation): the copies wait for it
    CUDA_CHECK(cudaEventRecord(kernel_done_s, s_kernel));
    CUDA_CHECK(cudaStreamWaitEvent(s_copy, kernel_done_s, 0));

    // The solver reads the pair as one contiguous [q_1 | q_2] buffer; copy the
    // next pair on the copy stream while the current one is computed.
    auto copy_pair = [&](size_t i, CuDeviceData<T>* d_pair)
    {
        CUDA_CHECK(cudaMemcpyAsync(&d_pair[0], raw(pairs[i].q_1), sizeof(T)*n_basis,
                                        cudaMemcpyDeviceToDevice, s_copy));
        CUDA_CHECK(cudaMemcpyAsync(&d_pair[n_basis], raw(pairs[i].q_2), sizeof(T)*n_basis,
                                        cudaMemcpyDeviceToDevice, s_copy));
        CUDA_CHECK(cudaEventRecord(copy_done_s, s_copy));
    };

    int prev = 0, next = 1;
    copy_pair(0, d_q_pair[stream][prev]);
    CUDA_CHECK(cudaStreamWaitEvent(s_kernel, copy_done_s, 0));

    for (size_t i = 0; i < pairs.size(); i++)
    {
        if (i + 1 < pairs.size())
            copy_pair(i + 1, d_q_pair[stream][next]);

        // block += weight * segment, summed on the device
        solver->compute_single_segment_stress(stream, d_q_pair[stream][prev], d_segment,
                                              monomer_type, pairs[i].is_half_bond_length);
        ker_lin_comb<<<1, 32, 0, s_kernel>>>(d_block, 1.0, d_block, pairs[i].weight, d_segment, N_STRESS_DIM);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaEventRecord(kernel_done_s, s_kernel));

        CUDA_CHECK(cudaStreamWaitEvent(s_copy, kernel_done_s, 0));
        CUDA_CHECK(cudaStreamWaitEvent(s_kernel, copy_done_s, 0));
        std::swap(prev, next);
    }

    // One copy per block; waiting on this stream only lets the other streams run on
    CUDA_CHECK(cudaMemcpyAsync(h_block_stress[stream], d_block, sizeof(T)*N_STRESS_DIM,
                                    cudaMemcpyDeviceToHost, s_kernel));
    CUDA_CHECK(cudaStreamSynchronize(s_kernel));
    CUDA_CHECK(cudaStreamSynchronize(s_copy));

    for (int d = 0; d < N_STRESS_DIM; d++)
        stress[d] = h_block_stress[stream][d];
    return stress;
}

template <typename T>
T CudaComputationBackend<T>::solvent_concentration(
    BackendArray<T>* phi, BackendArray<T>* w, double ds, double volume_fraction)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T>* d_phi = raw(phi);

    ker_exp<<<N_BLOCKS, N_THREADS>>>(d_phi, raw(w), 1.0, -ds, n_basis);
    CUDA_CHECK_LAUNCH();

    const T partition = cb->mean_device(d_phi);

    CuDeviceData<T> norm;
    if constexpr (std::is_same_v<T, double>)
        norm = volume_fraction / partition;
    else
        norm = cuCdiv(make_cuDoubleComplex(volume_fraction, 0.0), stdToCuDoubleComplex(partition));
    ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_phi, norm, d_phi, 0.0, d_phi, n_basis);
    CUDA_CHECK_LAUNCH();

    // No solvent inside impenetrable regions (the mask is in the field basis)
    if (!d_q_mask.empty())
    {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_phi, d_phi, d_q_mask, 1.0, n_basis);
        CUDA_CHECK_LAUNCH();
    }
    return partition;
}

template <typename T>
void CudaComputationBackend<T>::sum_to_host(
    T* out, const std::vector<BackendArray<T>*>& arrays, const std::vector<T>& weights)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(T)*n_basis));
    for (size_t a = 0; a < arrays.size(); a++)
    {
        ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(d_sum, to_device_scalar(weights[a]), raw(arrays[a]), 1.0, d_sum, n_basis);
        CUDA_CHECK_LAUNCH();
    }
    CUDA_CHECK(cudaMemcpy(out, d_sum, sizeof(T)*n_basis, cudaMemcpyDeviceToHost));
}

// Explicit template instantiation
template class CudaComputationBackend<double>;
template class CudaComputationBackend<std::complex<double>>;

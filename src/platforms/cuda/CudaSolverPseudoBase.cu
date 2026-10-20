/**
 * @file CudaSolverPseudoBase.cu
 * @brief Shared implementation of the CUDA pseudo-spectral solvers.
 *
 * See CudaSolverPseudoBase.h.
 */

#include <algorithm>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>

#include "CudaSolverPseudoBase.h"
#include "CudaPseudo.h"
#include "CudaFFT.h"
#include "CudaTypeDispatch.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTStress.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"
#include "CrysFFTSelector.h"
#include "CrysFFTBasisMapping.h"
#include "SpaceGroup.h"

namespace {

// Stress dyad sums V_c = Σ_k factor·integrand(k)·basis_c(k) for up to six components
// in one pass over k: each block writes its partial sums, and a one-block kernel adds
// the partials in block order, so the result does not depend on scheduling.
constexpr int STRESS_THREADS = 256;      // threads per block of both stress kernels
constexpr int STRESS_MAX_BLOCKS = 1024;  // capacity of the partial-sum buffer

struct StressBases { const double* basis[6]; };

__device__ inline double stress_zero(double) { return 0.0; }
__device__ inline cuDoubleComplex stress_zero(cuDoubleComplex) { return make_cuDoubleComplex(0.0, 0.0); }
__device__ inline void stress_add(double& acc, double v) { acc += v; }
__device__ inline void stress_add(cuDoubleComplex& acc, cuDoubleComplex v) { acc.x += v.x; acc.y += v.y; }
// Same product as ker_multi(dst, integrand, basis, factor)
__device__ inline double stress_term(double integrand, double basis, double factor) { return factor * integrand * basis; }
__device__ inline cuDoubleComplex stress_term(cuDoubleComplex integrand, double basis, double factor)
{
    return make_cuDoubleComplex(factor * integrand.x * basis, factor * integrand.y * basis);
}

// Warp sum (all 32 lanes take part; lane 0 gets the total)
__device__ inline double stress_warp_sum(double v)
{
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}
__device__ inline cuDoubleComplex stress_warp_sum(cuDoubleComplex v)
{
    return make_cuDoubleComplex(stress_warp_sum(v.x), stress_warp_sum(v.y));
}

// Block sum of NC per-thread values (blockDim.x == STRESS_THREADS); thread 0 gets the totals
template <typename TD, int NC>
__device__ inline void stress_block_sum(TD (&acc)[NC])
{
    __shared__ TD sh[NC][STRESS_THREADS / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int c = 0; c < NC; ++c)
    {
        acc[c] = stress_warp_sum(acc[c]);
        if (lane == 0)
            sh[c][warp] = acc[c];
    }
    __syncthreads();
    if (warp == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            acc[c] = stress_warp_sum(lane < STRESS_THREADS / 32 ? sh[c][lane] : stress_zero(TD{}));
    }
}

template <typename TD, int NC>
__global__ void ker_stress_dyad_partials(TD* partial, const TD* integrand, StressBases bases, double factor, int M)
{
    TD acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c)
        acc[c] = stress_zero(TD{});
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < M)
    {
        const TD v = integrand[i];
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            stress_add(acc[c], stress_term(v, bases.basis[c][i], factor));
        i += blockDim.x * gridDim.x;
    }
    stress_block_sum<TD, NC>(acc);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            partial[blockIdx.x * NC + c] = acc[c];
    }
}

template <typename TD, int NC>
__global__ void ker_stress_dyad_finalize(TD* out, const TD* partial, int n_partial)
{
    TD acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c)
        acc[c] = stress_zero(TD{});
    for (int b = threadIdx.x; b < n_partial; b += blockDim.x)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            stress_add(acc[c], partial[b * NC + c]);
    }
    stress_block_sum<TD, NC>(acc);
    if (threadIdx.x == 0)
    {
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            out[c] = acc[c];
    }
}

// out[0..NC-1] = Σ_i factor·integrand[i]·basis_c[i]: block partials, then their sum in block order
template <typename TD, int NC>
void launch_stress_dyads(TD* out, TD* partial, const TD* integrand, const StressBases& bases,
                         double factor, int M, cudaStream_t stream)
{
    // About two elements per thread, at most STRESS_MAX_BLOCKS blocks
    const int n_partial = std::max(1, std::min(STRESS_MAX_BLOCKS, (M + 2 * STRESS_THREADS - 1) / (2 * STRESS_THREADS)));
    ker_stress_dyad_partials<TD, NC><<<n_partial, STRESS_THREADS, 0, stream>>>(partial, integrand, bases, factor, M);
    CUDA_CHECK_LAUNCH();
    ker_stress_dyad_finalize<TD, NC><<<1, STRESS_THREADS, 0, stream>>>(out, partial, n_partial);
    CUDA_CHECK_LAUNCH();
}

template <typename TD>
void launch_stress_dyads(TD* out, TD* partial, const TD* integrand, const StressBases& bases, int n_comp,
                         double factor, int M, cudaStream_t stream)
{
    switch (n_comp)
    {
        case 1: launch_stress_dyads<TD, 1>(out, partial, integrand, bases, factor, M, stream); break;
        case 2: launch_stress_dyads<TD, 2>(out, partial, integrand, bases, factor, M, stream); break;
        case 3: launch_stress_dyads<TD, 3>(out, partial, integrand, bases, factor, M, stream); break;
        case 6: launch_stress_dyads<TD, 6>(out, partial, integrand, bases, factor, M, stream); break;
        default: throw_with_line_number("Unsupported number of stress components: " + std::to_string(n_comp));
    }
}

}  // namespace

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename T>
CudaSolverPseudoBase<T>::CudaSolverPseudoBase(
    ComputationBox<T>* cb,
    Molecules *molecules,
    int n_streams,
    cudaStream_t streams[MAX_STREAMS][2],
    int max_fft_batch,
    int q_full_in_factor,
    bool with_exp_dw_half,
    bool complex_requires_periodic)
{
    try{
        this->cb = cb;
        this->molecules = molecules;
        this->n_streams = n_streams;
        this->dim_ = cb->get_dim();
        this->q_full_in_factor_ = q_full_in_factor;
        this->max_fft_batch_ = max_fft_batch;

        for(int i=0; i<n_streams; i++)
        {
            this->streams[i][0] = streams[i][0];
            this->streams[i][1] = streams[i][1];
        }
        // The device buffers and transform objects are owning members, released
        // automatically (also when this constructor throws); the cuFFT plans are
        // tracked by their created flags for release_base()
        for(int i=0; i<MAX_STREAMS; i++)
        {
            temp_storage_bytes[i] = 0;
            for (int b = 0; b < 2; b++)
            {
                plan_for_[i][b] = plan_bak_[i][b] = 0;
                plan_for_created_[i][b] = plan_bak_created_[i][b] = false;
            }
        }
        space_group_ = nullptr;
        d_reduced_basis_indices_ = nullptr;
        d_full_to_reduced_map_ = nullptr;
        n_basis_ = 0;
        use_crysfft_ = false;
        crysfft_mode_ = CudaCrysFFTMode::None;
        crysfft_identity_map_ = false;
        crysfft_physical_size_ = 0;

        // Boundary conditions: the pseudo-spectral method needs the same BC on
        // both faces of each direction.
        auto bc_vec = cb->get_boundary_conditions();
        for (int d = 0; d < dim_; ++d)
        {
            if (bc_vec[2*d] != bc_vec[2*d + 1])
            {
                throw_with_line_number("Pseudo-spectral method requires matching boundary conditions on both sides of each direction. "
                    "Direction " + std::to_string(d) + " has mismatched BCs. Use real-space method for mixed BCs.");
            }
        }
        is_periodic_ = true;
        for (const auto& b : bc_vec)
        {
            if (b != BoundaryCondition::PERIODIC)
            {
                is_periodic_ = false;
                break;
            }
        }
        // [type-specific algorithm] The continuous-chain solvers have no DCT/DST
        // path for complex fields; the discrete solver has.
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (complex_requires_periodic && !is_periodic_)
                throw_with_line_number(
                    "Complex fields with non-periodic boundary conditions are "
                    "not yet ported to this CUDA solver; use the discrete "
                    "chain model or a CPU platform.");
        }

        // Pseudo-spectral helper (Fourier basis, Boltzmann bond factors).
        // Squeeze face BCs to one per dimension.
        {
            std::vector<BoundaryCondition> bc_dim_ctor;
            for (int d = 0; d < dim_; ++d)
                bc_dim_ctor.push_back(bc_vec[2 * d]);
            pseudo = std::make_unique<CudaPseudo<T>>(
                molecules->get_bond_lengths(),
                bc_dim_ctor,
                cb->get_nx(), cb->get_dx(),
                cb->get_recip_metric(),
                cb->get_recip_vec());
        }
        const int M = cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        // Register every distinct contour step and allocate the Boltzmann
        // factor buffers exp(-w ds) per (ds, monomer type).
        molecules->finalize_contour_length_mapping();
        const ContourLengthMapping& mapping = molecules->get_contour_length_mapping();
        const int n_unique_ds = mapping.get_n_unique_ds();
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const double local_ds = mapping.get_ds_from_index(ds_idx);
            pseudo->add_ds_value(ds_idx, local_ds);
            for(const auto& item: molecules->get_bond_lengths())
            {
                const std::string& monomer_type = item.first;
                this->d_exp_dw[ds_idx][monomer_type].allocate(M);
                if (with_exp_dw_half)
                    this->d_exp_dw_half[ds_idx][monomer_type].allocate(M);
            }
        }
        pseudo->finalize_ds_values();

        // Stress reduction workspace
        for(int i=0; i<n_streams; i++)
        {
            d_stress_sum[i].allocate(M_COMPLEX);
            d_stress_sum_out[i].allocate(1);
            d_q_multi[i].allocate(M_COMPLEX);
            d_stress_partial_[i].allocate(6*STRESS_MAX_BLOCKS);
            if constexpr (std::is_same<T, std::complex<double>>::value)
                d_qk_conj_[i].allocate(M_COMPLEX);
        }
        for(int i=0; i<n_streams; i++)
        {
            // the first call only queries the size of the CUB scratch
            temp_storage_bytes[i] = 0;
            CUDA_CHECK(cuda_reduce_sum(nullptr, temp_storage_bytes[i], d_stress_sum[i].get(), d_stress_sum_out[i].get(), M_COMPLEX, streams[i][0]));
            d_temp_storage[i].allocate(temp_storage_bytes[i]);
        }

        // Spectral transforms: cuFFT plans (periodic) or DCT/DST objects (non-periodic)
        init_fft();
    }
    catch(std::exception& exc)
    {
        release_base();
        throw_without_line_number(exc.what());
    }
    catch(...)
    {
        release_base();
        throw;
    }
}

template <typename T>
CudaSolverPseudoBase<T>::~CudaSolverPseudoBase()
{
    release_base();
}

template <typename T>
void CudaSolverPseudoBase<T>::release_base()
{
    // The cuFFT plans are tracked by their created flags; the transforms,
    // the pseudo object and the device buffers are owning members and are
    // released here explicitly as well, so a derived constructor that throws
    // leaves nothing behind. Calling this twice is harmless.
    cleanup_crysfft();
    pseudo.reset();

    for(int i=0; i<MAX_STREAMS; i++)
    {
        for (int b = 0; b < 2; b++)
        {
            if (plan_for_created_[i][b]) cufftDestroy(plan_for_[i][b]);
            if (plan_bak_created_[i][b]) cufftDestroy(plan_bak_[i][b]);
            plan_for_created_[i][b] = plan_bak_created_[i][b] = false;
            plan_for_[i][b] = plan_bak_[i][b] = 0;
        }
        fft_[i].reset();
    }

    this->d_exp_dw.clear();
    this->d_exp_dw_half.clear();
    for(int i=0; i<MAX_STREAMS; i++)
    {
        d_stress_sum[i].release();
        d_stress_sum_out[i].release();
        d_q_multi[i].release();
        d_qk_conj_[i].release();
        d_stress_partial_[i].release();
        d_temp_storage[i].release();
        temp_storage_bytes[i] = 0;
        d_q_full_in_[i].release();
        d_q_full_out_[i].release();
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::init_fft()
{
    // fft_ and the plan handles/flags were reset by the constructor
    if (is_periodic_)
    {
        // One plan per stream and batch size; cufftPlanMany reads dim_ entries of total_grid
        const int NRANK = dim_;
        int total_grid[3] = {0, 0, 0};
        for (int d = 0; d < dim_; ++d)
            total_grid[d] = cb->get_nx(d);

        for (int i = 0; i < n_streams; i++)
        {
            for (int b = 0; b < max_fft_batch_; b++)
            {
                CUFFT_CHECK(cufftPlanMany(&plan_for_[i][b], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, CufftPlanType<T>::forward,  b + 1));
                plan_for_created_[i][b] = true;
                CUFFT_CHECK(cufftPlanMany(&plan_bak_[i][b], NRANK, total_grid, nullptr, 1, 0, nullptr, 1, 0, CufftPlanType<T>::backward, b + 1));
                plan_bak_created_[i][b] = true;
                CUFFT_CHECK(cufftSetStream(plan_for_[i][b], streams[i][0]));
                CUFFT_CHECK(cufftSetStream(plan_bak_[i][b], streams[i][0]));
            }
        }
    }
    else
    {
        // Per-stream DCT/DST objects: each has its own work buffers, so streams
        // never race on them. One BC per dimension (both faces match).
        auto bc_vec = cb->get_boundary_conditions();
        for (int i = 0; i < n_streams; i++)
        {
            if (dim_ == 3)
            {
                std::array<int, 3> nx_arr = {cb->get_nx(0), cb->get_nx(1), cb->get_nx(2)};
                std::array<BoundaryCondition, 3> bc_arr = {bc_vec[0], bc_vec[2], bc_vec[4]};
                fft_[i] = std::make_unique<CudaFFT<T, 3>>(nx_arr, bc_arr);
            }
            else if (dim_ == 2)
            {
                std::array<int, 2> nx_arr = {cb->get_nx(0), cb->get_nx(1)};
                std::array<BoundaryCondition, 2> bc_arr = {bc_vec[0], bc_vec[2]};
                fft_[i] = std::make_unique<CudaFFT<T, 2>>(nx_arr, bc_arr);
            }
            else
            {
                std::array<int, 1> nx_arr = {cb->get_nx(0)};
                std::array<BoundaryCondition, 1> bc_arr = {bc_vec[0]};
                fft_[i] = std::make_unique<CudaFFT<T, 1>>(nx_arr, bc_arr);
            }
        }
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::fft_forward(int stream, CuDeviceData<T>* d_in, cuDoubleComplex* d_out, int batch)
{
    CUFFT_CHECK(cufft_exec_forward(plan_for_[stream][batch - 1], d_in, d_out));
}

template <typename T>
void CudaSolverPseudoBase<T>::fft_backward(int stream, cuDoubleComplex* d_in, CuDeviceData<T>* d_out, int batch)
{
    CUFFT_CHECK(cufft_exec_backward(plan_bak_[stream][batch - 1], d_in, d_out));
}

template <typename T>
void CudaSolverPseudoBase<T>::cleanup_crysfft()
{
    use_crysfft_ = false;
    crysfft_mode_ = CudaCrysFFTMode::None;
    crysfft_identity_map_ = false;
    crysfft_physical_size_ = 0;

    d_crysfft_phys_to_reduced_.release();
    d_crysfft_reduced_to_phys_.release();
    d_crysfft_orbit_counts_.release();
    for (int i = 0; i < MAX_STREAMS; i++)
    {
        crysfft_[i].reset();
        d_crysfft_in_[i].release();
        d_crysfft_out_[i].release();
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::set_space_group(
    SpaceGroup* sg,
    int* d_reduced_basis_indices,
    int* d_full_to_reduced_map,
    int n_basis)
{
    space_group_ = sg;
    d_reduced_basis_indices_ = d_reduced_basis_indices;
    d_full_to_reduced_map_ = d_full_to_reduced_map;
    n_basis_ = n_basis;

    cleanup_crysfft();

    // Full-grid work buffers around the FFT (freed again when the space group is removed)
    const int M = cb->get_total_grid();
    for(int i=0; i<n_streams; i++)
    {
        if (sg != nullptr)
        {
            if (d_q_full_in_[i].empty())
                d_q_full_in_[i].allocate(static_cast<size_t>(M) * q_full_in_factor_);
            if (d_q_full_out_[i].empty())
                d_q_full_out_[i].allocate(M);
        }
        else
        {
            d_q_full_in_[i].release();
            d_q_full_out_[i].release();
        }
    }

    // [type-specific algorithm] The CrysFFT fast path folds real fields on the
    // physical sub-grid; complex fields always use the full-grid reduced basis.
    if constexpr (std::is_same_v<T, double>)
    {
        if (space_group_ != nullptr && dim_ == 3 && is_periodic_)
            init_crysfft();
    }

    // The maps and CrysFFT setup above are snapshots: freeze the physical basis
    if (sg != nullptr)
        sg->mark_in_use();
}

template <typename T>
void CudaSolverPseudoBase<T>::init_crysfft()
{
    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    const auto angles = cb->get_angles();
    const bool z_axis_orthogonal =
        (angles.size() >= 3 &&
         std::abs(angles[0] - M_PI / 2.0) < 1e-10 &&
         std::abs(angles[1] - M_PI / 2.0) < 1e-10);
    const auto selection = select_crysfft_mode(
        space_group_, nx_logical, dim_, is_periodic_, cb->is_orthogonal(), z_axis_orthogonal);
    if (selection.mode == CrysFFTChoice::None && !selection.can_pmmm && !selection.can_oblique_z)
        return;

    const int M_reduced = space_group_->get_n_reduced_basis();
    const auto& full_to_reduced = space_group_->get_full_to_reduced_map();
    const bool use_m3_basis = space_group_->using_m3_physical_basis();
    const bool use_pmmm_basis = space_group_->using_pmmm_physical_basis();
    const bool use_hex_basis = space_group_->using_z_mirror_physical_basis();

    // Recursive 3m: octant sampled at even full-grid indices
    if (selection.mode == CrysFFTChoice::Recursive3m)
    {
        if (use_pmmm_basis)
            throw_with_line_number("Pmmm physical basis is enabled but recursive 3m CrysFFT is selected.");
        const auto map = build_crysfft_mapping_octant(nx_logical, full_to_reduced, M_reduced, /*even_indices=*/true);
        if (crysfft_mapping_usable(map, use_m3_basis, "M3 physical basis does not match recursive 3m CrysFFT grid ordering."))
            setup_crysfft(CudaCrysFFTMode::Recursive3m, map, selection.m3_translations);
    }

    // ObliqueZ: lower z half (z-mirror)
    if (!use_crysfft_ && selection.mode == CrysFFTChoice::ObliqueZ)
    {
        if (use_m3_basis || use_pmmm_basis)
            throw_with_line_number("ObliqueZ CrysFFT selected but Pmmm/M3 physical basis is enabled.");
        if (!selection.can_oblique_z)
            throw_with_line_number("ObliqueZ CrysFFT selected but z-mirror symmetry is unavailable.");
        const int z_shift = use_hex_basis ? space_group_->get_z_mirror_shift()
                                          : selection.oblique_z_shift;
        if (use_hex_basis && z_shift != selection.oblique_z_shift)
            throw_with_line_number("Z-mirror physical basis shift does not match CrysFFT selection.");
        const auto map = build_crysfft_mapping_z_mirror(nx_logical, full_to_reduced, M_reduced, z_shift);
        if (crysfft_mapping_usable(map, use_hex_basis, "Z-mirror physical basis does not match ObliqueZ CrysFFT grid ordering."))
            setup_crysfft(CudaCrysFFTMode::ObliqueZ, map, selection.m3_translations);
    }

    // Pmmm DCT: octant at consecutive indices
    if (!use_crysfft_ && selection.can_pmmm)
    {
        if (use_m3_basis)
            throw_with_line_number("M3 physical basis is enabled but recursive 3m CrysFFT is unavailable.");
        const auto map = build_crysfft_mapping_octant(nx_logical, full_to_reduced, M_reduced, /*even_indices=*/false);
        if (crysfft_mapping_usable(map, use_pmmm_basis, "Pmmm physical basis does not match Pmmm CrysFFT grid ordering."))
            setup_crysfft(CudaCrysFFTMode::PmmmDct, map, selection.m3_translations);
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::setup_crysfft(
    CudaCrysFFTMode mode, const CrysFFTBasisMapping& map, const std::array<double, 9>& m3_translations)
{
    crysfft_physical_size_ = map.M_phys;
    crysfft_identity_map_ = map.identity;

    // Index maps are needed only when the physical grid is not the reduced basis itself
    if (!crysfft_identity_map_)
    {
        d_crysfft_phys_to_reduced_.allocate(map.M_phys);
        d_crysfft_reduced_to_phys_.allocate(map.M_reduced);
        CUDA_CHECK(cudaMemcpy(d_crysfft_phys_to_reduced_, map.phys_to_reduced.data(),
                                   sizeof(int) * map.M_phys, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_crysfft_reduced_to_phys_, map.reduced_to_phys.data(),
                                   sizeof(int) * map.M_reduced, cudaMemcpyHostToDevice));
    }

    // Orbit counts for the CrysFFT stress fast path
    const auto& orbit_counts = space_group_->get_orbit_counts();
    d_crysfft_orbit_counts_.allocate(map.M_reduced);
    CUDA_CHECK(cudaMemcpy(d_crysfft_orbit_counts_, orbit_counts.data(),
                               sizeof(int) * map.M_reduced, cudaMemcpyHostToDevice));

    const auto nx = cb->get_nx();
    const std::array<int, 3> nx_logical = {nx[0], nx[1], nx[2]};
    const auto lx = cb->get_lx();
    const auto angles = cb->get_angles();
    const std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};

    for (int i = 0; i < n_streams; i++)
    {
        if (mode == CudaCrysFFTMode::Recursive3m)
            crysfft_[i] = std::make_unique<CudaCrysFFTRecursive3m>(nx_logical, cell_para, m3_translations);
        else if (mode == CudaCrysFFTMode::ObliqueZ)
            crysfft_[i] = std::make_unique<CudaCrysFFTObliqueZ>(nx_logical, cell_para);
        else
            crysfft_[i] = std::make_unique<CudaCrysFFT>(nx_logical, cell_para);

        d_crysfft_in_[i].allocate(map.M_phys);
        d_crysfft_out_[i].allocate(map.M_phys);
    }
    crysfft_mode_ = mode;
    use_crysfft_ = true;
}

template <typename T>
void CudaSolverPseudoBase<T>::update_laplacian_operator()
{
    try{
        // Update Pseudo Fourier basis arrays and boltz_bond for new box dimensions
        // Note: local_ds values are registered once in constructor via add_ds_value()
        // pseudo->update() recomputes boltz_bond for all registered ds values
        // Squeeze face BCs to one per dimension (see constructor note).
        auto bc_vec_upd = this->cb->get_boundary_conditions();
        std::vector<BoundaryCondition> bc_dim_upd;
        for (int d = 0; d < this->cb->get_dim(); ++d)
            bc_dim_upd.push_back(bc_vec_upd[2 * d]);
        pseudo->update(
            bc_dim_upd,
            this->molecules->get_bond_lengths(),
            this->cb->get_dx(),
            this->cb->get_recip_metric(),
            this->cb->get_recip_vec());

        if (use_crysfft_)
        {
            auto lx = cb->get_lx();
            auto angles = cb->get_angles();
            std::array<double, 6> cell_para = {lx[0], lx[1], lx[2], angles[0], angles[1], angles[2]};
            for (int i = 0; i < n_streams; ++i)
            {
                if (crysfft_[i] != nullptr)
                    crysfft_[i]->set_cell_para(cell_para);
            }
        }
    }
    catch(std::exception& exc)
    {
        throw_with_line_number(exc.what());
    }
}

template <typename T>
void CudaSolverPseudoBase<T>::update_dw_impl(
    const std::string& device, const std::map<std::string, const T*>& w_input,
    bool with_half, double factor, double factor_half)
{
    try{
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();
        const int M = cb->get_total_grid();

        const bool use_reduced_basis = (space_group_ != nullptr);
        const bool use_crysfft = (use_crysfft_ && space_group_ != nullptr);
        const int M_use = use_crysfft ? n_basis_ : M;

        // Get unique ds values from ContourLengthMapping
        const ContourLengthMapping& mapping = this->molecules->get_contour_length_mapping();
        const int n_unique_ds = mapping.get_n_unique_ds();

        cudaMemcpyKind cudaMemcpyInputToDevice;
        if (device == "gpu")
            cudaMemcpyInputToDevice = cudaMemcpyDeviceToDevice;
        else if(device == "cpu")
            cudaMemcpyInputToDevice = cudaMemcpyHostToDevice;
        else
        {
            throw_with_line_number("Invalid device \"" + device + "\".");
        }

        // Compute exp_dw for each ds_index and monomer type
        for (int ds_idx = 0; ds_idx < n_unique_ds; ++ds_idx)
        {
            const double local_ds = mapping.get_ds_from_index(ds_idx);
            for(const auto& item: w_input)
            {
                const std::string& monomer_type = item.first;
                const T *w = item.second;

                if (this->d_exp_dw[ds_idx].find(monomer_type) == this->d_exp_dw[ds_idx].end())
                    throw_with_line_number("monomer_type \"" + monomer_type + "\" is not in d_exp_dw[" + std::to_string(ds_idx) + "].");
                CuDeviceData<T>* d_full = this->d_exp_dw[ds_idx][monomer_type].get();
                CuDeviceData<T>* d_half = with_half ? this->d_exp_dw_half[ds_idx][monomer_type].get() : nullptr;

                if (use_reduced_basis && device == "gpu" && !use_crysfft)
                {
                    if constexpr (std::is_same_v<T, double>)
                    {
                        // Expand reduced basis -> full grid on device
                        ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_full, w, d_full_to_reduced_map_, M);
                        CUDA_CHECK_LAUNCH();
                        if (with_half)
                            ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS>>>(d_half, w, d_full_to_reduced_map_, M);
                        CUDA_CHECK_LAUNCH();
                    }
                    else
                    {
                        throw_with_line_number("Space group reduced basis is only supported for real fields.");
                    }
                }
                else
                {
                    const T* w_full = w;
                    std::vector<T> w_full_host;
                    if (use_reduced_basis && !use_crysfft)
                    {
                        if constexpr (std::is_same_v<T, double>)
                        {
                            w_full_host.resize(M);
                            space_group_->from_reduced_basis(w, w_full_host.data(), 1);
                            w_full = w_full_host.data();
                        }
                        else
                        {
                            throw_with_line_number("Space group reduced basis is only supported for real fields.");
                        }
                    }
                    // Copy field configurations to the device
                    CUDA_CHECK(cudaMemcpyAsync(d_full, w_full, sizeof(T)*M_use, cudaMemcpyInputToDevice));
                    if (with_half)
                        CUDA_CHECK(cudaMemcpyAsync(d_half, w_full, sizeof(T)*M_use, cudaMemcpyInputToDevice));
                }

                // d_exp_dw = exp(-w * local_ds * factor); d_exp_dw_half = exp(-w * local_ds * factor_half)
                ker_exp<<<N_BLOCKS, N_THREADS>>>(d_full, d_full, 1.0, -factor*local_ds, M_use);
                CUDA_CHECK_LAUNCH();
                if (with_half)
                    ker_exp<<<N_BLOCKS, N_THREADS>>>(d_half, d_half, 1.0, -factor_half*local_ds, M_use);
                CUDA_CHECK_LAUNCH();
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Stress of one segment (shared by RQM4, RK2 and the discrete chain)
//------------------------------------------------------------------------------
template <typename T>
const double* CudaSolverPseudoBase<T>::get_stress_boltz_bond(
    [[maybe_unused]] const std::string& monomer_type, [[maybe_unused]] bool is_half_bond_length) const
{
    return nullptr;   // continuous chains: the bond factor is already in the propagator
}

template <typename T>
void CudaSolverPseudoBase<T>::compute_single_segment_stress(
    const int STREAM,
    CuDeviceData<T> *d_q_pair, CuDeviceData<T> *d_segment_stress,
    std::string monomer_type, bool is_half_bond_length)
{
    try{
        // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            if (!is_periodic_)
                throw_with_line_number(
                    "Stress computation for complex fields with non-periodic "
                    "boundary conditions is not implemented.");
        }
        const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
        const int N_THREADS = CudaCommon::get_instance().get_n_threads();

        const int DIM = this->cb->get_dim();
        const int M = this->cb->get_total_grid();
        const int M_COMPLEX = pseudo->get_total_complex_grid();

        const auto& bond_lengths = this->molecules->get_bond_lengths();
        auto bond_it = bond_lengths.find(monomer_type);
        if (bond_it == bond_lengths.end())
            throw_with_line_number("monomer_type \"" + monomer_type + "\" not found in bond_lengths.");
        double bond_length_sq = bond_it->second * bond_it->second;
        if (is_half_bond_length)
            bond_length_sq *= 0.5;

        // Chain-model-specific Boltzmann bond factor (nullptr for continuous chains).
        // The discrete chain multiplies exp(-k^2 b^2 ds/6) * b^2 into the integrand
        // and weights the basis tables with 1; continuous chains weight them with b^2.
        const double* _d_boltz_bond = get_stress_boltz_bond(monomer_type, is_half_bond_length);
        const double basis_factor = (_d_boltz_bond != nullptr) ? 1.0 : bond_length_sq;

        const double* _d_fourier_basis_x = pseudo->get_fourier_basis_x();
        const double* _d_fourier_basis_y = pseudo->get_fourier_basis_y();
        const double* _d_fourier_basis_z = pseudo->get_fourier_basis_z();
        const double* _d_fourier_basis_xy = pseudo->get_fourier_basis_xy();
        const double* _d_fourier_basis_xz = pseudo->get_fourier_basis_xz();
        const double* _d_fourier_basis_yz = pseudo->get_fourier_basis_yz();
        const int* _d_negative_k_idx = pseudo->get_negative_frequency_mapping();

        // =====================================================================
        // CrysFFT stress fast path (reduced grid, no full-grid FFT)
        // Mirrors CpuSolverPseudoBase::compute_single_segment_stress.
        // ObliqueZ and all other cases fall through to the standard path.
        // =====================================================================
        // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
        if constexpr (std::is_same_v<T, double>)
        {
            // NOTE: The CrysFFT fast path computes only the diagonal sums
            // (V₁₁, V₂₂, V₃₃). This is fine because crystal systems eligible
            // for CrysFFT are orthogonal and never optimize angles. If the
            // force-off-diagonal flag is set (angle optimization active),
            // fall back to the standard path below — which can compute the
            // cross terms — instead of silently returning zeros for them.
            if (use_crysfft_ && space_group_ != nullptr && is_periodic_ && DIM == 3
                && this->cb->is_orthogonal()
                && !this->force_off_diagonal_stress_
                && (crysfft_mode_ == CudaCrysFFTMode::PmmmDct || crysfft_mode_ == CudaCrysFFTMode::Recursive3m))
            {
                const std::vector<double> lx_vec = this->cb->get_lx();

                CudaCrysFFTStressArgs args{};
                args.crysfft = crysfft_[STREAM].get();
                args.mode = crysfft_mode_;
                args.identity_map = crysfft_identity_map_;
                args.M_phys = crysfft_physical_size_;
                args.n_basis = n_basis_;
                args.M_full = M;
                args.d_phys_to_reduced = d_crysfft_phys_to_reduced_;
                args.d_reduced_to_phys = d_crysfft_reduced_to_phys_;
                args.d_orbit_counts = d_crysfft_orbit_counts_;
                args.d_phys_work = d_crysfft_in_[STREAM];
                args.d_phys_out = d_crysfft_out_[STREAM];
                args.d_reduce_buf = d_stress_sum[STREAM];
                args.d_cub_temp = d_temp_storage[STREAM];
                args.cub_temp_bytes = temp_storage_bytes[STREAM];
                args.space_group = space_group_;
                args.lx[0] = lx_vec[0];
                args.lx[1] = lx_vec[1];
                args.lx[2] = lx_vec[2];
                args.bond_length_sq = bond_length_sq;
                args.include_bond_factor = (_d_boltz_bond != nullptr);   // discrete chains
                args.global_ds = (_d_boltz_bond != nullptr) ? this->molecules->get_global_ds() : 0.0;
                args.stream = streams[STREAM][0];

                cuda_crysfft_compute_single_segment_stress(
                    args, d_q_pair, &d_q_pair[n_basis_], d_segment_stress);
                return;
            }
        }

        const StressScratch scratch = stress_scratch(STREAM);

        // Space group without CrysFFT fast path: the propagator pair arrives in
        // the reduced basis (packed at stride n_basis_). Expand both propagators
        // to the full grid into a contiguous 2*M buffer for the batched FFT.
        CuDeviceData<T>* d_q_pair_full = d_q_pair;
        if (space_group_ != nullptr)
        {
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (std::is_same_v<T, double>)
            {
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    scratch.pair_full, d_q_pair, d_full_to_reduced_map_, M);
                CUDA_CHECK_LAUNCH();
                ker_expand_reduced_basis<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(
                    &scratch.pair_full[M], &d_q_pair[n_basis_], d_full_to_reduced_map_, M);
                CUDA_CHECK_LAUNCH();
                d_q_pair_full = scratch.pair_full;
            }
            else
            {
                throw_with_line_number("Space group reduced basis is only supported for real fields.");
            }
        }

        if (is_periodic_)
        {
            // Periodic BC: batched forward FFT of both propagators
            fft_forward(STREAM, d_q_pair_full, scratch.qk_pair, 2);
            CUDA_CHECK_LAUNCH();

            // [type-specific algorithm] Real fields store the Hermitian half-spectrum, so the
            // conjugate product is formed directly; complex fields need the -k index map.
            // Multiply two propagators in the fourier spaces
            if constexpr (std::is_same<T, double>::value)
                ker_multi_complex_conjugate<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &scratch.qk_pair[0], &scratch.qk_pair[M_COMPLEX], M_COMPLEX);
            else
            {
                // Gather q2(-k) into a separate buffer: an in-place gather through the
                // k -> -k map (an involution) would race.
                ker_copy_data_with_idx<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_qk_conj_[STREAM], &scratch.qk_pair[M_COMPLEX], _d_negative_k_idx, M_COMPLEX);
                CUDA_CHECK_LAUNCH();
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], &scratch.qk_pair[0], d_qk_conj_[STREAM], 1.0, M_COMPLEX);
                CUDA_CHECK_LAUNCH();
            }
            CUDA_CHECK_LAUNCH();
        }
        else
        {
            // Non-periodic BC: use CudaFFT (DCT/DST)
            // [real-only feature] Space-group reduced basis, CrysFFT and non-periodic stress are implemented for real fields; complex fields take the generic path or raise.
            if constexpr (!std::is_same<T, double>::value)
            {
                throw_with_line_number("Stress computation with non-periodic BC is only supported for real fields (double).");
            }
            else
            {
                double* d_q1 = d_q_pair_full;
                double* d_q2 = &d_q_pair_full[M];

                fft_[STREAM]->forward_stream(d_q1, scratch.rk_1, streams[STREAM][0]);
                fft_[STREAM]->forward_stream(d_q2, scratch.rk_2, streams[STREAM][0]);

                // Multiply (real coefficients for non-periodic BC)
                // Per-mode Parseval weights (2 interior / 1 special modes) are folded
                // into the fourier_basis tables by Pseudo::update_weighted_fourier_basis_mixed.
                // CudaFFT's real-coefficient forward is 2x the textbook DCT-II/DST-II
                // convention per transformed dimension (CudaRealTransform matches FFTW's
                // unnormalized REDFT10/RODFT10, while the CPU FftwFFT/MklFFT scale by 0.5);
                // the coefficient product therefore carries 4^DIM, compensated here.
                // (All dimensions are r2r here: mixed periodic/non-periodic is rejected.)
                const double dct_scale = std::pow(0.25, DIM);
                ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], scratch.rk_1, scratch.rk_2, dct_scale, M_COMPLEX);
                CUDA_CHECK_LAUNCH();
            }
        }

        // Discrete chains: Boltzmann bond factor exp(-k²·b²·ds/6) and b² (already
        // halved for half-bond steps) into the integrand
        if (_d_boltz_bond != nullptr)
        {
            ker_multi<<<N_BLOCKS, N_THREADS, 0, streams[STREAM][0]>>>(d_q_multi[STREAM], d_q_multi[STREAM], _d_boltz_bond, bond_length_sq, M_COMPLEX);
            CUDA_CHECK_LAUNCH();
        }

        // fourier_basis arrays contain v⊗v dyad components of the deformation
        // vector v = 2π g⁻¹ m (units 1/L², NOT the Cartesian wavevector):
        // _d_fourier_basis_x = v₁², _d_fourier_basis_xy = v₁v₂, etc.
        // The accumulated sums V_ij = Σ kernel·v_i·v_j are converted to
        // lattice-parameter derivatives in compute_stress().
        // No cross-term corrections needed.

        // ============================================================================
        // PERFORMANCE-CRITICAL: Orthogonal box optimization
        // ============================================================================
        // For orthogonal boxes (all angles = 90°), the cross-term sums (V₁₂, V₁₃, V₂₃)
        // are not computed: angle optimization is inactive for orthogonal crystal
        // systems, so those slots are never consumed. This optimization
        // reduces the number of GPU kernel calls from 6 to 3 for 3D boxes,
        // providing ~10% speedup when box_is_altering=True.
        //
        // DO NOT REMOVE THIS OPTIMIZATION without benchmarking! We have experienced
        // performance regressions before when this check was accidentally removed.
        // See git history for commit 6d1dc54 which caused a regression by always
        // computing all 6 components.
        //
        // Explicit opt-in: force_off_diagonal_stress_ (set when angle optimization
        // is active) disables the skip so that angle derivatives are available
        // even when the box is at exactly 90° angles. The skip remains the default.
        // ============================================================================
        const bool is_orthogonal = this->cb->is_orthogonal() && !this->force_off_diagonal_stress_;

        // Dyad components V_c = Σ_k integrand(k)·basis_c(k)·basis_factor, written to
        // d_segment_stress[0..n_comp-1]
        StressBases bases{};
        int n_comp = 0;
        auto add_component = [&](const double* d_basis) { bases.basis[n_comp++] = d_basis; };
        if ( DIM == 3 )
        {
            add_component(_d_fourier_basis_x);   // V₁₁
            add_component(_d_fourier_basis_y);   // V₂₂
            add_component(_d_fourier_basis_z);   // V₃₃

            // Cross-terms: only compute for non-orthogonal boxes (triclinic lattices)
            if (!is_orthogonal)
            {
                add_component(_d_fourier_basis_xy);   // V₁₂
                add_component(_d_fourier_basis_xz);   // V₁₃
                add_component(_d_fourier_basis_yz);   // V₂₃
            }
        }
        if ( DIM == 2 )
        {
            // For non-periodic BC, 2D grid is mapped to y-z axes internally (tnx = {1, nx[0], nx[1]})
            // so stress data is stored in fourier_basis_y and fourier_basis_z
            add_component(is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_y);   // V₁₁
            add_component(is_periodic_ ? _d_fourier_basis_y : _d_fourier_basis_z);   // V₂₂

            // V₁₂: only compute for non-orthogonal boxes
            if (!is_orthogonal)
                add_component(is_periodic_ ? _d_fourier_basis_xy : _d_fourier_basis_yz);
        }
        if ( DIM == 1 )
        {
            // For non-periodic BC, 1D grid is mapped to z-axis internally (tnx = {1, 1, nx[0]})
            // so stress data is stored in fourier_basis_z
            add_component(is_periodic_ ? _d_fourier_basis_x : _d_fourier_basis_z);
        }

        // One pass over k for all components, then the block partials in block order
        launch_stress_dyads(d_segment_stress, d_stress_partial_[STREAM].get(), d_q_multi[STREAM].get(), bases, n_comp,
                            basis_factor, M_COMPLEX, streams[STREAM][0]);
        CUDA_CHECK_LAUNCH();
    }
    catch(std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiation
template class CudaSolverPseudoBase<double>;
template class CudaSolverPseudoBase<std::complex<double>>;

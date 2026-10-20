/**
 * @file ComputationBackend.h
 * @brief Platform interface used by the platform-neutral propagator computations.
 *
 * The propagator computation classes in src/common (ComputationContinuous,
 * ComputationDiscrete, ...) decide *what* is computed and in *which order*:
 * which propagators, which junctions, which segment pairs and weights, which
 * normalization. Every operation that touches grid data goes through a
 * ComputationBackend, which the CPU and CUDA platforms implement.
 *
 * **Granularity rule.** One backend operation is one physically meaningful
 * step on whole fields: "advance one propagator by one segment", "build the
 * junction product", "accumulate the concentration of one block". Do not add
 * array-level helpers (scale, add, multiply): they force one kernel launch per
 * call and keep the backend from fusing the work of a step.
 *
 * **Streams.** Operations that take a `stream` argument may run concurrently
 * with operations issued on other streams; on the CPU the stream index is the
 * OpenMP thread and the call completes before it returns, on CUDA the work is
 * queued on that CUDA stream. Call synchronize() before reading results that
 * were produced on several streams. Operations without a stream argument are
 * ordered after all work issued before them on any stream, and the scalars or
 * host arrays they return are complete when they return.
 *
 * **Memory.** Fields live in backend memory (host memory on the CPU, device
 * memory on CUDA) and are handled through BackendArray pointers. The type is
 * never defined, so the common code can pass the pointers around but cannot
 * read or write the data by accident; only the backend casts them back.
 * Every array holds get_n_basis() elements (the reduced basis when a space
 * group is set, otherwise the full grid).
 *
 * **Checkpoints.** The reduce-memory computations keep most propagator
 * segments as CheckpointArray, which the backend may place in slower memory
 * (pinned host memory on CUDA). A checkpoint is only stored to, loaded from,
 * or used as a junction source; every other operation takes BackendArray.
 *
 * @see CpuComputationBackend, CudaComputationBackend
 */

#ifndef COMPUTATION_BACKEND_H_
#define COMPUTATION_BACKEND_H_

#include <array>
#include <climits>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

/// One field in backend memory. Intentionally incomplete (see the file comment).
template <typename T>
struct BackendArray;

/// One stored propagator segment (reduce-memory mode). Intentionally incomplete.
template <typename T>
struct CheckpointArray;

/**
 * @brief One pair of propagators that meet at a segment, with its weight.
 *
 * Used for the concentration and stress integrals of a block, which are sums
 * over segments of weight * q_1 * q_2.
 */
template <typename T>
struct SegmentPair
{
    BackendArray<T>* q_1;              ///< Propagator coming from one end of the block
    BackendArray<T>* q_2;              ///< Propagator coming from the other end
    double weight;                     ///< Quadrature weight of this segment
    bool is_half_bond_length = false;  ///< Discrete chains: the pair is joined by a half bond
};

template <typename T>
class ComputationBackend
{
public:
    virtual ~ComputationBackend() {}

    // ==================== Execution ====================

    /// Number of propagators that can be computed concurrently.
    virtual int get_n_streams() const = 0;

    /// Wait until every operation issued so far has finished.
    virtual void synchronize() = 0;

protected:
    /// Leading integer of OMP_NUM_THREADS (it may be a nested list "4,2"), or the default.
    static int n_streams_from_env(int default_n_streams)
    {
        const char* env = std::getenv("OMP_NUM_THREADS");
        if (env == nullptr || *env == '\0')
            return default_n_streams;
        char* end = nullptr;
        const long value = std::strtol(env, &end, 10);
        if (end == env || value <= 0 || value > INT_MAX)
            return default_n_streams;
        return static_cast<int>(value);
    }

public:

    // ==================== Memory ====================

    /// Allocate one field of get_n_basis() elements, set to zero.
    virtual BackendArray<T>* allocate() = 0;
    virtual void deallocate(BackendArray<T>* a) = 0;

    /// dst = src, where src is a host array of get_n_basis() elements.
    virtual void copy_from_host(int stream, BackendArray<T>* dst, const T* src) = 0;

    /// dst = src, where dst is a host array of get_n_basis() elements.
    virtual void copy_to_host(T* dst, BackendArray<T>* src) = 0;

    /// Allocate one checkpoint of get_n_basis() elements (contents undefined).
    virtual CheckpointArray<T>* allocate_checkpoint() = 0;
    virtual void deallocate_checkpoint(CheckpointArray<T>* a) = 0;

    /// Checkpoint dst = src.
    virtual void store_checkpoint(int stream, CheckpointArray<T>* dst, BackendArray<T>* src) = 0;

    /// dst = checkpoint src.
    virtual void load_checkpoint(int stream, BackendArray<T>* dst, CheckpointArray<T>* src) = 0;

    // ==================== Solver ====================

    /// Recompute the diffusion (or bond) operator after the box changed.
    virtual void update_laplacian_operator() = 0;

    /// Recompute the Boltzmann factors from the host fields w_input.
    virtual void update_dw(std::map<std::string, const T*> w_input) = 0;

    virtual void set_force_off_diagonal_stress(bool force) = 0;

    /// q_out = propagator q_in advanced by one segment of this monomer type
    /// (q_in == q_out is allowed).
    virtual void advance_propagator(int stream, BackendArray<T>* q_in, BackendArray<T>* q_out,
                                    const std::string& monomer_type, int ds_index) = 0;

    /// Discrete chains: q_out = q_in convolved with a half bond (q_in == q_out is allowed).
    virtual void advance_propagator_half_bond_step(int stream, BackendArray<T>* q_in, BackendArray<T>* q_out,
                                                   const std::string& monomer_type) = 0;

    /// Discrete chains: exp(-w ds) of one full segment, valid after update_dw().
    virtual BackendArray<T>* boltzmann_factor(const std::string& monomer_type) = 0;

    // ==================== Propagator initial conditions ====================

    /// q = 1 (free chain end).
    virtual void set_unity(int stream, BackendArray<T>* q) = 0;

    /// a = 0.
    virtual void set_zero(int stream, BackendArray<T>* a) = 0;

    /// dst = src.
    virtual void copy(int stream, BackendArray<T>* dst, BackendArray<T>* src) = 0;

    /// q = Σ_i n_i q_i over sources (q_i, n_i): aggregated propagators.
    virtual void junction_sum(int stream, BackendArray<T>* q,
                              const std::vector<std::pair<BackendArray<T>*, int>>& sources) = 0;

    /// q = Π_i q_i^{n_i} over sources (q_i, n_i): propagators meeting at a junction.
    virtual void junction_product(int stream, BackendArray<T>* q,
                                  const std::vector<std::pair<BackendArray<T>*, int>>& sources) = 0;

    /// junction_sum() with checkpoints as sources.
    virtual void junction_sum(int stream, BackendArray<T>* q,
                              const std::vector<std::pair<CheckpointArray<T>*, int>>& sources) = 0;

    /// junction_product() with checkpoints as sources.
    virtual void junction_product(int stream, BackendArray<T>* q,
                                  const std::vector<std::pair<CheckpointArray<T>*, int>>& sources) = 0;

    /// q = q * mask (impenetrable regions); does nothing without a mask.
    virtual void apply_mask(int stream, BackendArray<T>* q) = 0;

    /// Discrete chains: q = q * exp(-w ds), the Boltzmann factor of one full segment.
    virtual void multiply_boltzmann_factor(int stream, BackendArray<T>* q, const std::string& monomer_type) = 0;

    // ==================== Statistics ====================

    /// ∫ a b dr over the grid (orbit-weighted in the reduced basis).
    virtual T inner_product(BackendArray<T>* a, BackendArray<T>* b) = 0;

    /// ∫ a b / w dr over the grid.
    virtual T inner_product_inverse_weight(BackendArray<T>* a, BackendArray<T>* b, BackendArray<T>* w) = 0;

    /// Concentration of a block, accumulated: phi += Σ_i weight_i q_1,i q_2,i.
    virtual void add_segment_products(int stream, BackendArray<T>* phi,
                                      const std::vector<SegmentPair<T>>& pairs) = 0;

    /**
     * @brief Normalize a block concentration: phi = norm * phi / divisor.
     * @param divisor Field to divide by before the normalization, or nullptr.
     */
    virtual void normalize_concentration(int stream, BackendArray<T>* phi, T norm, BackendArray<T>* divisor) = 0;

    /**
     * @brief Stress of one block: Σ_i weight_i * σ(q_1,i, q_2,i).
     *
     * σ is the solver's segment stress, the six v⊗v sums
     * [V₁₁, V₂₂, V₃₃, V₁₂, V₁₃, V₂₃] in the deformation-vector basis;
     * components the solver does not compute are zero.
     */
    virtual std::array<T, 6> block_stress(int stream, const std::vector<SegmentPair<T>>& pairs,
                                          const std::string& monomer_type) = 0;

    /**
     * @brief Solvent concentration phi = volume_fraction * exp(-w ds) / Q.
     * @return The solvent partition function Q = mean(exp(-w ds)).
     */
    virtual T solvent_concentration(BackendArray<T>* phi, BackendArray<T>* w,
                                    double ds, double volume_fraction) = 0;

    /// out = Σ_i weight_i a_i, where out is a host array of get_n_basis() elements.
    virtual void sum_to_host(T* out, const std::vector<BackendArray<T>*>& arrays,
                             const std::vector<T>& weights) = 0;
};

#endif

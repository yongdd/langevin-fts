/**
 * @file CrysFFTPmmmBase.h
 * @brief CRTP base class for crystallographic FFT with Pmmm symmetry.
 *
 * Provides common functionality for DCT-based Pmmm symmetry FFT implementations.
 * Derived classes (FFTW, MKL) implement the actual DCT forward/backward operations.
 *
 * **Common Functionality:**
 *
 * - Grid dimension validation and setup
 * - Per-thread Boltzmann factor caching (owned by the object)
 * - Cell parameter management
 * - Contour step management
 *
 * **CRTP Pattern:**
 *
 * Derived classes must implement:
 * - `get_thread_local_buffers()` - Per-thread (io, temp) work buffers
 * - `void dct_forward_impl(double* io, double* temp)` - DCT-II forward
 * - `void dct_backward_impl(double* io, double* temp)` - DCT-III backward
 *
 * @see FftwCrysFFTPmmm for FFTW implementation
 * @see MklCrysFFTPmmm for MKL implementation
 */

#ifndef CRYS_FFT_PMMM_BASE_H_
#define CRYS_FFT_PMMM_BASE_H_

#include <array>
#include <atomic>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>
#include <numbers>

#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @class CrysFFTPmmmBase
 * @brief CRTP base class for Pmmm crystallographic FFT.
 *
 * @tparam Derived The derived class type (CRTP pattern)
 */
template <typename Derived>
class CrysFFTPmmmBase
{
protected:
    std::array<int, 3> nx_logical_;    ///< Logical grid dimensions (Nx, Ny, Nz)
    std::array<int, 3> nx_physical_;   ///< Physical grid: (Nx/2, Ny/2, Nz/2)
    int M_logical_;                     ///< Total logical grid size
    int M_physical_;                    ///< Total physical grid size

    // Cell parameters (Lx, Ly, Lz, alpha, beta, gamma)
    std::array<double, 6> cell_para_;

    // Normalization factor for DCT-II/III round trip: 1/M_logical
    double norm_factor_;

    // Work buffers (allocated by derived class)
    double* io_buffer_;
    double* temp_buffer_;

    //--------------------------------------------------------------------------
    // Per-thread Boltzmann factor caching (owned by the object)
    //--------------------------------------------------------------------------

    struct ThreadState
    {
        std::map<double, std::unique_ptr<double[]>> boltzmann;
        const double* boltz_current = nullptr;
        double ds_current = std::numeric_limits<double>::quiet_NaN();
        uint64_t epoch = 0;
    };

    mutable std::atomic<uint64_t> cache_epoch_{1};   ///< bumped to invalidate every thread's cache
    /// Unique per object, never reused: lets a thread recognise the object its
    /// remembered cache slot belongs to (see get_thread_state()).
    inline static std::atomic<uint64_t> next_instance_id_{1};
    const uint64_t instance_id_{next_instance_id_.fetch_add(1, std::memory_order_relaxed)};
    mutable std::mutex thread_states_mutex_;
    /// Per-calling-thread caches (the propagator loop runs in OpenMP threads),
    /// owned by this object so that they are released together with it.
    mutable std::unordered_map<std::thread::id, ThreadState> thread_states_;

    /// Cache of the calling thread (owned by this object, see thread_states_).
    ThreadState& get_thread_state() const
    {
        // Each thread remembers the slot it used last, tagged with the object's
        // id, so the usual case (one object per thread) takes no lock. The slot
        // is looked up under the lock only when the thread switches object; the
        // map keeps references valid across insertions, and a stale remembered
        // pointer is never followed because ids are not reused.
        struct LastSlot { uint64_t instance_id = 0; ThreadState* state = nullptr; };
        thread_local LastSlot last;
        if (last.instance_id != instance_id_)
        {
            std::lock_guard<std::mutex> lock(thread_states_mutex_);
            last.state = &thread_states_[std::this_thread::get_id()];
            last.instance_id = instance_id_;
        }
        ThreadState* state = last.state;
        const uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
        if (state->epoch != epoch)
        {
            state->boltzmann.clear();
            state->boltz_current = nullptr;
            state->ds_current = std::numeric_limits<double>::quiet_NaN();
            state->epoch = epoch;
        }
        return *state;
    }

    /// Generate Boltzmann factors for given ds.
    std::unique_ptr<double[]> generateBoltzmann(double ds) const
    {
        auto boltz = std::make_unique<double[]>(M_physical_);

        double Lx = cell_para_[0];
        double Ly = cell_para_[1];
        double Lz = cell_para_[2];

        int Nx2 = nx_physical_[0];
        int Ny2 = nx_physical_[1];
        int Nz2 = nx_physical_[2];

        const double PI = M_PI;

        int idx = 0;
        for (int ix = 0; ix < Nx2; ++ix)
        {
            double kx = ix * 2.0 * PI / Lx;
            double kx2 = kx * kx;

            for (int iy = 0; iy < Ny2; ++iy)
            {
                double ky = iy * 2.0 * PI / Ly;
                double ky2 = ky * ky;

                for (int iz = 0; iz < Nz2; ++iz)
                {
                    double kz = iz * 2.0 * PI / Lz;
                    double kz2 = kz * kz;

                    boltz[idx++] = std::exp(-(kx2 + ky2 + kz2) * ds);
                }
            }
        }

        return boltz;
    }

    /// Invalidate all Boltzmann factor caches.
    void freeBoltzmann()
    {
        cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
    }

    /// Access derived class (CRTP).
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

public:
    /**
     * @brief Construct CrysFFTPmmmBase for given grid.
     *
     * @param nx_logical Logical grid dimensions (must be even)
     * @param cell_para  Cell parameters [Lx, Ly, Lz, alpha, beta, gamma]
     */
    CrysFFTPmmmBase(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para)
        : nx_logical_(nx_logical),
          cell_para_(cell_para),
          io_buffer_(nullptr),
          temp_buffer_(nullptr)
    {
        // Validate even grid sizes
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] % 2 != 0)
            {
                throw_with_line_number("CrysFFTPmmm requires even grid dimensions. "
                    "Dimension " + std::to_string(d) + " has size " + std::to_string(nx_logical_[d]));
            }
            if (nx_logical_[d] <= 0)
            {
                throw_with_line_number("CrysFFTPmmm requires positive grid dimensions.");
            }
        }

        // Physical grid: (N/2) in each dimension
        nx_physical_ = {
            nx_logical_[0] / 2,
            nx_logical_[1] / 2,
            nx_logical_[2] / 2
        };

        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

        // DCT-II/III normalization: 1 / M_logical for round-trip
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);
    }

    /// Virtual destructor.
    virtual ~CrysFFTPmmmBase() = default;

    /// Update cell parameters.
    void set_cell_para(const std::array<double, 6>& cell_para)
    {
        if (cell_para[0] == cell_para_[0] &&
            cell_para[1] == cell_para_[1] &&
            cell_para[2] == cell_para_[2])
        {
            return;
        }

        cell_para_ = cell_para;
        freeBoltzmann();
    }

    /// Set contour step size and prepare Boltzmann factors.
    void set_contour_step(double ds)
    {
        ThreadState& state = get_thread_state();
        if (state.boltz_current != nullptr && state.ds_current == ds)
            return;

        auto it = state.boltzmann.find(ds);
        if (it == state.boltzmann.end())
        {
            std::unique_ptr<double[]> boltz = generateBoltzmann(ds);
            it = state.boltzmann.emplace(ds, std::move(boltz)).first;
        }

        state.ds_current = ds;
        state.boltz_current = it->second.get();
    }

    /**
     * @brief Apply diffusion operator using DCT-II/III.
     *
     * Computes: q_out = DCT-III[ exp(-k²ds) * DCT-II[q_in] ]
     */
    void diffusion(double* q_in, double* q_out)
    {
        ThreadState& state = get_thread_state();
        if (!state.boltz_current)
        {
            throw_with_line_number("CrysFFTPmmm::set_contour_step must be called before diffusion().");
        }

        // Get thread-local buffers from derived class
        auto [io_local, temp_local] = derived().get_thread_local_buffers();

        // Step 1: Copy input
        std::memcpy(io_local, q_in, sizeof(double) * M_physical_);

        // Step 2: Forward DCT-II
        derived().dct_forward_impl(io_local, temp_local);

        // Step 3: Apply Boltzmann factor
        for (int i = 0; i < M_physical_; ++i)
            io_local[i] *= state.boltz_current[i];

        // Step 4: Backward DCT-III
        derived().dct_backward_impl(io_local, temp_local);

        // Step 5: Apply normalization and copy result
        for (int i = 0; i < M_physical_; ++i)
            q_out[i] = io_local[i] * norm_factor_;
    }

    /// Apply custom multiplier in Fourier space.
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier)
    {
        // Get thread-local buffers from derived class
        auto [io_local, temp_local] = derived().get_thread_local_buffers();

        // Step 1: Copy input
        std::memcpy(io_local, q_in, sizeof(double) * M_physical_);

        // Step 2: Forward DCT-II
        derived().dct_forward_impl(io_local, temp_local);

        // Step 3: Apply multiplier
        for (int i = 0; i < M_physical_; ++i)
            io_local[i] *= multiplier[i];

        // Step 4: Backward DCT-III
        derived().dct_backward_impl(io_local, temp_local);

        // Step 5: Apply normalization and copy result
        for (int i = 0; i < M_physical_; ++i)
            q_out[i] = io_local[i] * norm_factor_;
    }

    // Accessors
    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif  // CRYS_FFT_PMMM_BASE_H_

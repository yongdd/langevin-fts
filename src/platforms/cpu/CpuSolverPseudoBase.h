/**
 * @file CpuSolverPseudoBase.h
 * @brief Common base class for CPU pseudo-spectral solvers.
 *
 * This header provides CpuSolverPseudoBase, a common base class that
 * consolidates shared functionality between CpuSolverPseudoRQM4
 * and CpuSolverPseudoDiscrete, including:
 *
 * - FFT object management (creation/destruction)
 * - Pseudo-spectral operator helper
 * - Boundary condition handling
 * - Transform dispatch methods
 * - Laplacian operator updates
 * - Stress computation (with customizable coefficient function)
 *
 * **Design Decision:**
 *
 * All simulations are conducted in fixed dimensions (1D, 2D, or 3D).
 * The FFT object is stored as `std::unique_ptr<FFT<T>> fft_` which points to FftwFFT<T, DIM>
 * or MklFFT<T, DIM> based on the backend and the dimensionality. This avoids
 * maintaining separate fft_1d, fft_2d, fft_3d pointers when only one is ever
 * used per solver.
 *
 * **Stress Array Convention (Voigt Notation):**
 *
 * - Index 0-2: Diagonal components (σ₁, σ₂, σ₃) for length optimization
 * - Index 3-5: Off-diagonal components (σ₁₂, σ₁₃, σ₂₃) for angle optimization
 *
 * @see CpuSolverPseudoRQM4 for continuous chain implementation
 * @see CpuSolverPseudoDiscrete for discrete chain implementation
 * @see docs/theory/StressTensor.md for detailed derivation
 */

#ifndef CPU_SOLVER_PSEUDO_BASE_H_
#define CPU_SOLVER_PSEUDO_BASE_H_

#include <string>
#include <vector>
#include <map>
#include <complex>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "Exception.h"
#include "Molecules.h"
#include "ComputationBox.h"
#include "CpuSolver.h"
#include "Pseudo.h"
#include "FFTFactory.h"
#include "SpaceGroup.h"
#include "CrysFFTBasisMapping.h"

#ifdef USE_CPU_FFTW
#include "FftwCrysFFTPmmm.h"
#include "FftwCrysFFTRecursive3m.h"
#include "FftwCrysFFTObliqueZ.h"
#endif

#ifdef USE_CPU_MKL
#include "MklCrysFFTPmmm.h"
#include "MklCrysFFTRecursive3m.h"
#include "MklCrysFFTObliqueZ.h"
#endif

/**
 * @class CpuSolverPseudoBase
 * @brief Common base class for CPU pseudo-spectral propagator solvers.
 *
 * Consolidates shared code between continuous and discrete chain solvers,
 * including FFT management, transform dispatch, and stress computation.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Shared Functionality:**
 *
 * - FFT object creation/destruction with dimension dispatch
 * - Forward/backward transform methods
 * - Laplacian operator updates
 * - Stress computation framework
 *
 * **Virtual Methods:**
 *
 * Derived classes must implement:
 * - update_dw(): Different Boltzmann factor formulas
 * - advance_propagator(): Different propagation algorithms
 * - advance_propagator_half_bond_step(): Discrete-specific
 * - get_stress_boltz_bond() (optional): bond factor of the stress integrand; the base returns nullptr
 */
template <typename T>
class CpuSolverPseudoBase : public CpuSolver<T>
{
protected:
    /**
     * @brief Work arrays of one calling thread.
     *
     * The propagators run in OpenMP threads, so every thread needs its own
     * arrays. They are owned by the solver and released with it (thread_local
     * arrays would stay allocated until the threads end). Each step function
     * and the stress routine use them from start to end of one call; nothing
     * nests, so one set per thread is enough.
     */
    struct WorkArrays
    {
        std::vector<T> q_1, q_2;                     ///< Step results (RQM4 full step and half steps)
        std::vector<T> q_full_in, q_full_out;        ///< Full-grid copies of reduced-basis propagators
        std::vector<double> k_1, k_2;                ///< Spectral coefficients
        std::vector<double> phys_in, phys_out;       ///< CrysFFT physical grid
        std::vector<T> pair_1, pair_2;               ///< Stress: full-grid propagator pair
        std::vector<double> phys_1, phys_2, phys_3;  ///< Stress: CrysFFT physical grid
        std::vector<double> reduced, boltz, multiplier;  ///< Stress: reduced-basis sum, bond factor, multiplier
    };

    /// Work arrays of the calling thread (created on first use)
    WorkArrays& work_arrays() const
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        return work_arrays_[std::this_thread::get_id()];
    }

    ComputationBox<T>* cb;       ///< Computation box for grid info
    Molecules* molecules;        ///< Molecules container
    std::string chain_model;     ///< Chain model identifier

    /**
     * @brief Spectral transform object for FFT/DCT/DST (handles all BCs).
     *
     * Points to FftwFFT<T, DIM> or MklFFT<T, DIM> based on the selected
     * backend, where DIM is 1, 2, or 3 based on the simulation dimension.
     * All simulations are conducted in fixed dimensions, so only one
     * transform object is needed per solver instance.
     *
     * Uses an FFT<T> base class pointer to enable polymorphic calls without
     * dimension-specific casting or backend-specific dispatch.
     */
    std::unique_ptr<FFT<T>> fft_;

    FFTBackend fft_backend_;  ///< Selected FFT backend (FFTW or MKL)

    std::unique_ptr<Pseudo<T>> pseudo;   ///< Pseudo-spectral operator helper

    bool is_periodic_;           ///< True if all BCs are periodic
    int dim_;                    ///< Number of dimensions (1, 2, or 3)

    /// Space group for reduced basis operations (nullptr if not used).
    SpaceGroup* space_group_;

    enum class CrysFFTMode
    {
        None,
        PmmmDct,
        Recursive3m,
        ObliqueZ
    };

#ifdef USE_CPU_FFTW
    std::unique_ptr<FftwCrysFFTPmmm> crysfft_pmmm_fftw_;
    std::unique_ptr<FftwCrysFFTRecursive3m> crysfft_recursive_fftw_;
    std::unique_ptr<FftwCrysFFTObliqueZ> crysfft_oblique_fftw_;
#endif
#ifdef USE_CPU_MKL
    std::unique_ptr<MklCrysFFTPmmm> crysfft_pmmm_mkl_;
    std::unique_ptr<MklCrysFFTRecursive3m> crysfft_recursive_mkl_;
    std::unique_ptr<MklCrysFFTObliqueZ> crysfft_oblique_mkl_;
#endif
    CrysFFTMode crysfft_mode_ = CrysFFTMode::None;
    std::map<int, double> ds_values_;

    std::vector<int> crysfft_full_indices_;
    std::vector<int> crysfft_reduced_indices_;
    std::vector<double> crysfft_kx2_;
    std::vector<double> crysfft_ky2_;
    std::vector<double> crysfft_kz2_;
    std::array<double, 3> crysfft_k_cache_lx_ = {{-1.0, -1.0, -1.0}};
    bool crysfft_identity_map_ = false;  ///< True when reduced basis matches physical grid

    bool use_crysfft() const { return crysfft_mode_ != CrysFFTMode::None; }
    bool use_crysfft_recursive() const { return crysfft_mode_ == CrysFFTMode::Recursive3m; }
    bool use_crysfft_pmmm() const { return crysfft_mode_ == CrysFFTMode::PmmmDct; }
    bool use_crysfft_oblique() const { return crysfft_mode_ == CrysFFTMode::ObliqueZ; }
    bool use_crysfft_identity_map() const { return crysfft_identity_map_; }

    int get_crysfft_physical_size() const;
    void crysfft_set_cell_para(const std::array<double, 6>& cell_para);
    void crysfft_set_contour_step(double coeff);
    void crysfft_diffusion(double* q_in, double* q_out) const;
    void update_crysfft_k_cache();

    /**
     * @brief Perform forward transform.
     *
     * Delegates to fft_->forward() which handles all boundary conditions.
     *
     * @param rdata Real-space data input
     * @param cdata Fourier/DCT coefficient output
     */
    void transform_forward(T* rdata, double* cdata) { fft_->forward(rdata, cdata); }

    /**
     * @brief Perform backward transform.
     *
     * Delegates to fft_->backward() which handles all boundary conditions.
     *
     * @param cdata Fourier/DCT coefficient input
     * @param rdata Real-space data output
     */
    void transform_backward(double* cdata, T* rdata) { fft_->backward(cdata, rdata); }

    /**
     * @brief Whether spectral coefficients are stored as complex pairs.
     *
     * Periodic BCs always use complex coefficients. For complex fields
     * (CL-FTS) the DCT/DST coefficients of the real and imaginary parts
     * are stored interleaved as complex pairs as well — the transforms
     * are real-linear, so each part transforms independently.
     */
    bool complex_coeffs() const
    {
        if constexpr (std::is_same<T, std::complex<double>>::value)
            return true;
        else
            return is_periodic_;
    }

    /**
     * @brief Multiply Fourier coefficients by a factor array.
     *
     * Handles both periodic BC (complex coefficients) and non-periodic BC
     * (real coefficients from DCT/DST).
     *
     * @param k_data    Fourier coefficient array (modified in-place)
     * @param factor    Multiplicative factor array
     * @param n_complex Number of complex elements (M_COMPLEX)
     */
    void multiply_fourier_coeffs(double* k_data, const double* factor, int n_complex)
    {
        if (complex_coeffs())
        {
            std::complex<double>* k_complex = reinterpret_cast<std::complex<double>*>(k_data);
            for (int i = 0; i < n_complex; ++i)
                k_complex[i] *= factor[i];
        }
        else
        {
            for (int i = 0; i < n_complex; ++i)
                k_data[i] *= factor[i];
        }
    }

    /**
     * @brief Linear combination: k_out = a*k_in1 + b*k_in2
     *
     * Handles periodic (complex) and non-periodic (real) coefficient types.
     */
    void fourier_linear_combination_2(
        double* k_out, const double* a, const double* k_in1,
        const double* b, const double* k_in2, int n_complex)
    {
        if (complex_coeffs())
        {
            std::complex<double>* out = reinterpret_cast<std::complex<double>*>(k_out);
            const std::complex<double>* in1 = reinterpret_cast<const std::complex<double>*>(k_in1);
            const std::complex<double>* in2 = reinterpret_cast<const std::complex<double>*>(k_in2);
            for (int i = 0; i < n_complex; ++i)
                out[i] = a[i] * in1[i] + b[i] * in2[i];
        }
        else
        {
            for (int i = 0; i < n_complex; ++i)
                k_out[i] = a[i] * k_in1[i] + b[i] * k_in2[i];
        }
    }

    /**
     * @brief Linear combination: k_out = a*k_in1 + b*k_in2 + c*k_in3
     *
     * Handles periodic (complex) and non-periodic (real) coefficient types.
     */
    void fourier_linear_combination_3(
        double* k_out, const double* a, const double* k_in1,
        const double* b, const double* k_in2,
        const double* c, const double* k_in3, int n_complex)
    {
        if (complex_coeffs())
        {
            std::complex<double>* out = reinterpret_cast<std::complex<double>*>(k_out);
            const std::complex<double>* in1 = reinterpret_cast<const std::complex<double>*>(k_in1);
            const std::complex<double>* in2 = reinterpret_cast<const std::complex<double>*>(k_in2);
            const std::complex<double>* in3 = reinterpret_cast<const std::complex<double>*>(k_in3);
            for (int i = 0; i < n_complex; ++i)
                out[i] = a[i] * in1[i] + b[i] * in2[i] + c[i] * in3[i];
        }
        else
        {
            for (int i = 0; i < n_complex; ++i)
                k_out[i] = a[i] * k_in1[i] + b[i] * k_in2[i] + c[i] * k_in3[i];
        }
    }

    /**
     * @brief Linear combination: k_out += a*(k_in1 - k_in2)
     *
     * Handles periodic (complex) and non-periodic (real) coefficient types.
     */
    void fourier_add_scaled_diff(
        double* k_out, const double* a,
        const double* k_in1, const double* k_in2, int n_complex)
    {
        if (complex_coeffs())
        {
            std::complex<double>* out = reinterpret_cast<std::complex<double>*>(k_out);
            const std::complex<double>* in1 = reinterpret_cast<const std::complex<double>*>(k_in1);
            const std::complex<double>* in2 = reinterpret_cast<const std::complex<double>*>(k_in2);
            for (int i = 0; i < n_complex; ++i)
                out[i] += a[i] * (in1[i] - in2[i]);
        }
        else
        {
            for (int i = 0; i < n_complex; ++i)
                k_out[i] += a[i] * (k_in1[i] - k_in2[i]);
        }
    }

    /**
     * @brief Initialize shared components in constructor.
     *
     * Sets up:
     * - Boundary condition checking (is_periodic_)
     * - FFT object creation (fft_) using specified backend
     * - Pseudo-spectral operator (pseudo)
     *
     * @param cb Computation box
     * @param molecules Molecules container
     * @param backend FFT backend to use (FFTW or MKL)
     */
    void init_shared(ComputationBox<T>* cb, Molecules* molecules, FFTBackend backend);


    /**
     * @brief Get bond factor for stress computation.
     *
     * Returns the appropriate multiplier for stress coefficient:
     * - Continuous (default): Returns nullptr (no bond factor in stress)
     * - Discrete (override): Returns bond function ĝ(k) or half-bond ĝ^(1/2)(k)
     *
     * @param monomer_type Monomer type
     * @param is_half_bond_length Whether using half bond length
     * @return Pointer to bond function array, or nullptr if not applicable
     */
    virtual const double* get_stress_boltz_bond(
        std::string monomer_type, bool is_half_bond_length) const;

    // ---- space group / CrysFFT setup (set_space_group) ----------------------
    /// Drop every CrysFFT object and index map.
    void reset_crysfft();
    /// Select and build the CrysFFT fast path for the current space group (3D periodic, real fields).
    void init_crysfft();
    /// Store the physical-grid <-> reduced-basis map of the chosen CrysFFT (see plan_crysfft()).
    void adopt_crysfft_mapping(const CrysFFTBasisMapping& map, bool physical_basis_enabled);

    // ---- update_dw() helper --------------------------------------------------
    /**
     * @brief Bring the update_dw() input fields onto the grid the Boltzmann factors live on.
     *
     * With a space group and the standard FFT path the reduced-basis fields are
     * expanded to the full grid into `w_full_cache` (real fields only; complex
     * fields throw). Without a space group, or on the CrysFFT path where exp_dw
     * stays in the reduced basis, the input pointers are used as they are.
     * `w_use` maps each monomer type to the field pointer to read, and the
     * return value is the number of grid points each exp_dw array must hold.
     *
     * Serial use only: called from update_dw(), never from the OpenMP
     * propagator loops.
     */
    int expand_w_for_update_dw(const std::map<std::string, const T*>& w_input,
                               std::map<std::string, std::vector<T>>& w_full_cache,
                               std::map<std::string, const T*>& w_use) const;

    // ---- compute_single_segment_stress() helpers ----------------------------
    /**
     * @brief Diagonal stress sums through the CrysFFT fast path.
     *
     * Applies to real fields on orthogonal 3D periodic cells with a Pmmm or
     * recursive-3m CrysFFT and no angle optimization. Returns false when the
     * fast path does not apply; `stress` is then untouched.
     */
    bool compute_stress_crysfft(const T* q_1, const T* q_2, double bond_length_sq,
                                const double* boltz_bond, std::vector<T>& stress);
    /// phys_out = DCT^-1[ k_axis^2 (boltz) DCT[phys_in] ] through the Pmmm CrysFFT (boltz may be nullptr).
    void crysfft_pmmm_axis_multiplier(double* phys_in, double* phys_out, int axes, const double* boltz);
    /// Same through the recursive-3m CrysFFT; returns false when no recursive object exists.
    bool crysfft_recursive_axis_multiplier(double* phys_in, double* phys_out, int axes, bool with_bond, double coeff);
    /// Accumulate Σ_k kernel(k) v_i v_j from complex Fourier coefficients (periodic BC).
    void accumulate_stress_periodic(const double* qk_1, const double* qk_2, double bond_length_sq,
                                    const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const;
    /// Accumulate Σ_k kernel(k) v_i v_j from real DCT/DST coefficients (non-periodic BC).
    void accumulate_stress_nonperiodic(const double* qk_1, const double* qk_2, double bond_length_sq,
                                       const double* boltz_bond, bool is_orthogonal, std::vector<T>& stress) const;

public:
    /// Default constructor.
    CpuSolverPseudoBase() : is_periodic_(true), dim_(0), space_group_(nullptr) {}

    /// Virtual destructor. Frees the shared components (FFT, Pseudo).
    virtual ~CpuSolverPseudoBase() = default;   // fft_ and pseudo are owning members

    /**
     * @brief Set space group for reduced basis operations.
     *
     * When set, propagator input/output uses reduced basis and the solver
     * internally handles expand/reduce around FFT operations.
     *
     * @param sg Space group pointer (nullptr to disable reduced basis)
     */
    void set_space_group(SpaceGroup* sg);

    /// Register contour step size for a ds index.
    void register_ds_value(int ds_index, double ds) { ds_values_[ds_index] = ds; }

    /// Get the contour step size for a ds index.
    double get_ds_value(int ds_index) const;

    /// Compute effective diffusion coefficient (b^2 * ds / 6).
    double get_effective_diffusion_coeff(
        const std::string& monomer_type, int ds_index, bool half_step) const;

    /**
     * @brief Update Fourier-space operators.
     *
     * Recomputes exp(-b²|k|²ds/6) for each monomer type when box
     * dimensions change. Required after box size updates.
     *
     * - Continuous chains: diffusion propagator
     * - Discrete chains: bond function ĝ(k)
     *
     * Shared by all three pseudo-spectral solvers (none overrides it).
     */
    void update_laplacian_operator() override;

protected:
    /**
     * @brief Copy a reduced-basis field onto the CrysFFT physical grid, and back.
     *
     * Index-map copies (crysfft_reduced_to_phys_ / crysfft_phys_to_reduced_).
     */
    void fill_crysfft_from_reduced(const double* reduced_in, double* phys_out) const;
    void reduce_crysfft_to_reduced(const double* phys_in, double* reduced_out) const;

    /**
     * @brief Compute stress contribution from a single segment.
     *
     * Computes ∂ln(Q)/∂θ in Fourier space by multiplying forward and backward
     * propagators with weighted basis functions. Cross-term corrections for
     * non-orthogonal boxes are included.
     *
     * Chain model difference:
     * - Continuous: Φ(k) = 1 (bond factor already in propagator)
     * - Discrete: Φ(k) = exp(-b²k²Δs/6) or exp(-b²k²Δs/12) for half-bond
     *
     * @param q_1                Forward propagator
     * @param q_2                Backward propagator
     * @param monomer_type       Monomer type for segment length
     * @param is_half_bond_length Whether using half bond (discrete model)
     *
     * @return Stress components in Voigt notation:
     *         [σ₁, σ₂, σ₃, σ₁₂, σ₁₃, σ₂₃] (6 components)
     *         For 2D: [σ₁, σ₂, σ₁₂, 0, 0, 0]
     *         For 1D: only index 0 is meaningful
     */
    std::vector<T> compute_single_segment_stress(
        T* q_1, T* q_2, std::string monomer_type, bool is_half_bond_length) override;

private:
    mutable std::mutex work_mutex_;
    mutable std::unordered_map<std::thread::id, WorkArrays> work_arrays_;   ///< Per-thread work arrays (see WorkArrays)
};

#endif

/**
 * @file FourierTransform.h
 * @brief Fourier multipliers on whole fields: backward(Σ M(k) · forward(fields)).
 *
 * A FourierTransform applies linear operators that are diagonal in the
 * wavevector, such as smearing kernels, Green's functions and the propagators
 * of linear Langevin dynamics, to fields that live on the host. The CPU version
 * transforms with FFTW or MKL; the CUDA version uploads the fields, transforms
 * and multiplies on the device and downloads the result, with the multipliers
 * stored on the device once.
 *
 * **Coefficients** follow the layout of the platform FFT (FFT<T>):
 *
 * - Periodic axes, real fields: half spectrum of the r2c transform, shape
 *   (nx[0], ..., nx[d-1]/2 + 1), complex.
 * - Periodic axes, complex fields: full spectrum of the c2c transform, shape nx.
 * - Wall axes: cosine (DCT-II) or sine (DST-II) series, shape nx; real for
 *   real fields, complex for complex fields (real and imaginary parts are
 *   transformed independently).
 *
 * forward() is unnormalized, F(m) = Σ_j f_j φ_m(x_j) with φ_m = exp(-2πimj/N),
 * cos(πm(j+½)/N) or sin(π(m+1)(j+½)/N) per axis, and backward() includes the
 * normalization, so backward(forward(f)) = f. Periodic and wall axes cannot be
 * mixed.
 *
 * **Multipliers** are stored once (create_multiplier()) and applied with apply():
 *
 *     out_i = backward( Σ_terms Σ_j M_ij(k) · forward(in_j)(k) )
 *
 * A matrix multiplier holds n_out × n_in functions of k; a diagonal one holds
 * a single function applied to each field of its term. With real fields and
 * periodic axes the result is real: only the half spectrum of M is read, and on
 * its planes that hold both k and -k (last index 0 and, for an even length, n/2)
 * M is replaced by (M(k) + conj M(-k))/2, the operator of a real field (an odd
 * derivative i·k loses its Nyquist mode). The inverse r2c transforms read
 * inconsistent values on these planes differently, so without this the result
 * would depend on the backend; backward() treats its coefficients the same way.
 * With real fields and wall axes the coefficients are real and so must be M.
 *
 * get_k_squared() gives |k|² in the coefficient layout from the current box,
 * with the alias convention of the propagator solvers (Pseudo): the minimal
 * |k|² representative on oblique cells, k = 2πm/L on periodic axes and
 * k = πm/L (cosine) or π(m+1)/L (sine) on wall axes.
 *
 * Arrays passed to and returned by these functions are host arrays, field
 * after field. A transform runs one call at a time.
 *
 * @see CpuFourierTransform, CudaFourierTransform
 */

#ifndef FOURIER_TRANSFORM_H_
#define FOURIER_TRANSFORM_H_

#include <complex>
#include <memory>
#include <vector>

#include "ComputationBox.h"

template <typename T>
class FourierTransform;

/**
 * @class FourierMultiplier
 * @brief Handle of a multiplier stored by a FourierTransform.
 *
 * The values live in the platform's memory (device memory on CUDA) and are
 * freed with the handle. A handle is used only with the transform that
 * created it.
 */
template <typename T>
class FourierMultiplier
{
public:
    virtual ~FourierMultiplier() = default;

    /// Number of output fields (for a diagonal multiplier: 1, see is_diagonal())
    int get_n_out() const { return n_out_; }
    /// Number of input fields (for a diagonal multiplier: 1, see is_diagonal())
    int get_n_in() const { return n_in_; }
    /// One function of k applied to every field of a term
    bool is_diagonal() const { return diagonal_; }
    /// Transform that created the handle
    const FourierTransform<T>* get_owner() const { return owner_; }

protected:
    FourierMultiplier(const FourierTransform<T>* owner, int n_out, int n_in, bool diagonal)
        : owner_(owner), n_out_(n_out), n_in_(n_in), diagonal_(diagonal) {}

private:
    const FourierTransform<T>* owner_;
    int n_out_;
    int n_in_;
    bool diagonal_;
};

/**
 * @brief One term of FourierTransform::apply(): a multiplier and its input fields.
 *
 * `fields` holds n_fields host fields of n_grid values each: n_in of the
 * multiplier, or any number for a diagonal multiplier (which then gives as many
 * output fields).
 */
template <typename T>
struct FourierTerm
{
    const FourierMultiplier<T>* multiplier;
    const T* fields;
    int n_fields;
};

/**
 * @class FourierTransform
 * @brief Forward/backward transforms and Fourier multipliers of whole fields.
 *
 * @tparam T Field type (double or std::complex<double>)
 */
template <typename T>
class FourierTransform
{
public:
    /**
     * @param cb Computation box (grid and cell); must outlive the transform. The
     *           cosine and sine series need an orthogonal cell and keep it so
     *           (ComputationBox::hold_orthogonal_cell()) while they exist.
     * @param bc Transform of each axis: PERIODIC for all axes, or REFLECTING
     *           (cosine) / ABSORBING (sine) for all axes
     */
    FourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc);
    /// Releases the orthogonal cell held by the cosine and sine series
    virtual ~FourierTransform();

    FourierTransform(const FourierTransform&) = delete;
    FourierTransform& operator=(const FourierTransform&) = delete;

    /// Number of grid points of a field
    int get_n_grid() const { return n_grid_; }
    /// Number of coefficients of a field
    int get_n_coefficients() const { return n_coefficients_; }
    /// Shape of the coefficients of a field
    const std::vector<int>& get_coefficient_shape() const { return coefficient_shape_; }
    /// Coefficients are complex (false only for real fields with wall axes)
    bool has_complex_coefficients() const { return complex_coefficients_; }
    /// Transform of each axis
    const std::vector<BoundaryCondition>& get_boundary_conditions() const { return bc_; }

    /// |k|² of every coefficient for the current box
    std::vector<double> get_k_squared() const;

    /**
     * @brief Store a multiplier.
     *
     * @param values   n_out × n_in × n_coefficients values, k fastest (a diagonal
     *                 multiplier: n_coefficients values, n_out = n_in = 1)
     * @param n_out    Number of output fields
     * @param n_in     Number of input fields
     * @param diagonal One function applied to each field
     */
    std::unique_ptr<FourierMultiplier<T>> create_multiplier(
        const std::complex<double>* values, int n_out, int n_in, bool diagonal);

    /**
     * @brief out = backward(Σ_terms M · forward(fields)).
     *
     * @param terms Multipliers and their input fields
     * @param n_out Number of output fields (every term must give n_out fields)
     * @param out   n_out × n_grid values
     */
    void apply(const std::vector<FourierTerm<T>>& terms, int n_out, T* out);

    /**
     * @brief Coefficients of n_fields fields.
     *
     * @param fields       n_fields × n_grid values
     * @param n_fields     Number of fields
     * @param coefficients n_fields × n_coefficients values, complex numbers as
     *                     interleaved (real, imaginary) pairs when
     *                     has_complex_coefficients()
     */
    void forward(const T* fields, int n_fields, double* coefficients);

    /// Inverse of forward() (same layouts)
    void backward(const double* coefficients, int n_fields, T* fields);

protected:
    ComputationBox<T>* cb_;
    std::vector<BoundaryCondition> bc_;
    int dim_;
    bool periodic_;
    int n_grid_;
    int n_coefficients_;
    std::vector<int> coefficient_shape_;
    bool complex_coefficients_;
    bool holds_orthogonal_cell_ = false;

    /// Make n_blocks half spectra Hermitian on the planes that hold both k and -k
    /// (real fields with periodic axes only; see the class description)
    void make_hermitian(std::complex<double>* values, long long n_blocks) const;

    /// Doubles per field of coefficients (2·n_coefficients when complex)
    long long coefficient_stride() const
    {
        return (complex_coefficients_ ? 2LL : 1LL) * n_coefficients_;
    }

    /// Platform storage of validated values (complex, or real parts when the coefficients are real)
    virtual std::unique_ptr<FourierMultiplier<T>> store_multiplier(
        const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal) = 0;
    virtual void apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out) = 0;
    virtual void forward_impl(const T* fields, int n_fields, double* coefficients) = 0;
    virtual void backward_impl(const double* coefficients, int n_fields, T* fields) = 0;
};

#endif

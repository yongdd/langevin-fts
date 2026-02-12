/**
 * @file AndersonMixing.cpp
 * @brief Implementation of AndersonMixing base class.
 *
 * Provides the constructor and SVD-based least-squares solver for the
 * Anderson mixing algorithm. The actual mixing computation is implemented
 * in platform-specific derived classes (CpuAndersonMixing, CudaAndersonMixing).
 *
 * **Anderson Mixing Algorithm:**
 *
 * Given field history {w_i} and residual history {d_i}, find coefficients
 * {a_i} that minimize the linear combination of residuals:
 *
 *     min || Σ a_i d_i ||²  subject to Σ a_i = 1
 *
 * This leads to a least-squares problem solved via SVD (LAPACKE_dgelsd/zgelsd).
 *
 * **Template Instantiations:**
 *
 * - AndersonMixing<double>: Real field mixing
 * - AndersonMixing<std::complex<double>>: Complex field mixing
 */

#include <complex>
#include <vector>
#include <type_traits>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

#if __has_include(<lapacke.h>)
#include <lapacke.h>
#elif __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#error "LAPACKE header not found. Install with: conda install -c conda-forge lapack"
#endif

// RAII helper to suppress stderr (LAPACK xerbla warnings)
struct SuppressStderr {
    int saved_fd;
    SuppressStderr() {
        fflush(stderr);
        saved_fd = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    ~SuppressStderr() {
        fflush(stderr);
        dup2(saved_fd, STDERR_FILENO);
        close(saved_fd);
    }
};

#include "AndersonMixing.h"

/**
 * @brief Construct Anderson mixing with given parameters.
 *
 * @param n_var       Number of field variables
 * @param max_hist    Maximum history length for mixing
 * @param start_error Error threshold to begin Anderson mixing
 * @param mix_min     Minimum mixing parameter (prevents instability)
 * @param mix_init    Initial mixing parameter
 */
template <typename T>
AndersonMixing<T>::AndersonMixing(int n_var, int max_hist,
    double start_error, double mix_min, double mix_init)
{
    // The number of variables to be determined
    this->n_var = n_var;
    // Anderson mixing begin if error level becomes less then start_anderson_error
    this->start_error = start_error;
    // Maximum number of histories to calculate new field when using Anderson mixing
    this->max_hist = max_hist;
    // Minimum mixing parameter
    this->mix_min = mix_min;
    // Initialize mixing parameter
    this->mix = mix_init;
    this->mix_init = mix_init;
}

/**
 * @brief Solve least-squares problem Ua = v via SVD (LAPACKE_dgelsd/zgelsd).
 *
 * Uses SVD decomposition for robust handling of ill-conditioned normal
 * equations (e.g., compressible model with condition number ~100).
 *
 * @param u Input matrix U (n × n), stored as pointer-to-pointer rows
 * @param v Input vector (size n)
 * @param a Output solution vector (size n)
 * @param n System size (number of history entries used)
 */
template <typename T>
void AndersonMixing<T>::find_an(T **u, T *v, T *a, int n)
{
    if (n <= 0) return;

    // Handle n=1 case directly (avoids LAPACK overhead and DLASCL warnings)
    if (n == 1)
    {
        if (std::abs(u[0][0]) > 1e-20)
            a[0] = v[0] / u[0][0];
        else
            a[0] = static_cast<T>(0.0);
        return;
    }

    // Check max element to avoid LAPACK warnings on zero/tiny matrices
    double max_elem = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            max_elem = std::max(max_elem, std::abs(u[i][j]));
    if (max_elem < 1e-20)
    {
        for (int i = 0; i < n; i++)
            a[i] = static_cast<T>(0.0);
        return;
    }

    std::vector<double> s(n);  // singular values
    lapack_int rank;
    lapack_int info;

    if constexpr (std::is_same_v<T, double>)
    {
        // Copy to contiguous row-major array for LAPACKE
        std::vector<double> A(n * n);
        std::vector<double> b(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A[i * n + j] = u[i][j];
            b[i] = v[i];
        }

        {
            SuppressStderr guard;
            info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, n, n, 1,
                                  A.data(), n, b.data(), 1,
                                  s.data(), 1e-10, &rank);
        }

        for (int i = 0; i < n; i++)
            a[i] = (info == 0) ? b[i] : 0.0;
    }
    else  // std::complex<double>
    {
        std::vector<std::complex<double>> A(n * n);
        std::vector<std::complex<double>> b(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A[i * n + j] = u[i][j];
            b[i] = v[i];
        }

        {
            SuppressStderr guard;
            info = LAPACKE_zgelsd(LAPACK_ROW_MAJOR, n, n, 1,
                                  reinterpret_cast<lapack_complex_double*>(A.data()), n,
                                  reinterpret_cast<lapack_complex_double*>(b.data()), 1,
                                  s.data(), 1e-10, &rank);
        }

        for (int i = 0; i < n; i++)
            a[i] = (info == 0) ? b[i] : std::complex<double>(0.0, 0.0);
    }
}

// Explicit template instantiation
template class AndersonMixing<double>;
template class AndersonMixing<std::complex<double>>;

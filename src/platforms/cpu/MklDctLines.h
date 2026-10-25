/**
 * @file MklDctLines.h
 * @brief DCT-II and DCT-III of the lines of a grid along one axis, through a
 *        same-length real FFT of MKL DFTI with pre- and post-twiddles.
 *
 * Shared by MklFFT (reflecting boundaries) and MklCrysFFTPmmm. The algorithm
 * follows cuHelmholtz (Makhoul 1980). The lines are the num_transforms * stride
 * lines of length n with element spacing stride, starting at
 * batch * n * stride + s; temp is scratch of the same size, and the result is
 * copied back into data. cos_tbl and sin_tbl hold cos(k pi / (2n)) and
 * sin(k pi / (2n)) for k = 0..n/2.
 */

#ifndef MKL_DCT_LINES_H_
#define MKL_DCT_LINES_H_

#include <complex>
#include <cstring>
#include <vector>

#include "mkl_dfti.h"
#include "MklStatusCheck.h"

/**
 * @brief post_scale times FFTW's unnormalized REDFT10 of every line:
 *        X_k = 2 post_scale sum_j x_j cos(pi (j + 1/2) k / n).
 *
 * fft_backward is a committed length-n real DFTI descriptor (complex-complex
 * conjugate-even storage, not in place); its backward transform is used.
 */
inline void mkl_dct2_lines(
    double* data, double* temp, int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_backward, const double* cos_tbl, const double* sin_tbl, double post_scale)
{
    if (n == 1)
    {
        // REDFT10 of a singleton is 2*x; no FFT or neighboring sample is needed.
        for (int i = 0; i < num_transforms * stride; ++i)
            data[i] *= 2.0 * post_scale;
        return;
    }
    const int complex_size = n / 2 + 1;

    thread_local std::vector<double> slice;
    thread_local std::vector<std::complex<double>> fft_in;
    thread_local std::vector<double> fft_out;
    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(complex_size);
        fft_out.resize(n);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            const int offset = batch * n * stride + s;

            // Extract 1D slice (strided)
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-II preprocessing: rearrange into complex format (cuHelmholtz preOp_dct2)
            fft_in[0] = std::complex<double>(slice[0], 0.0);
            for (int k = 1; k < complex_size - 1; ++k)
            {
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }
            if (n % 2 == 0)
            {
                fft_in[n / 2] = std::complex<double>(slice[n - 1], 0.0);
            }
            else
            {
                int k = n / 2;
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }

            // Inverse FFT (c2r)
            mkl_dfti_check(DftiComputeBackward(fft_backward,
                                               reinterpret_cast<double*>(fft_in.data()),
                                               fft_out.data()),
                           "DftiComputeBackward (DCT-II)");

            // DCT-II postprocessing: twiddle factors (cuHelmholtz postOp_dct2)
            temp[offset] = (2.0 * post_scale) * fft_out[0];
            for (int k = 1; k <= n / 2; ++k)
            {
                double Ta = fft_out[k] + fft_out[n - k];
                double Tb = fft_out[k] - fft_out[n - k];

                double result_k = (Ta * cos_tbl[k] + Tb * sin_tbl[k]) * post_scale;
                double result_nk = (Ta * sin_tbl[k] - Tb * cos_tbl[k]) * post_scale;

                temp[offset + k * stride] = result_k;
                if (k < n - k)
                    temp[offset + (n - k) * stride] = result_nk;
            }
        }
    }

    std::memcpy(data, temp, static_cast<size_t>(num_transforms) * n * stride * sizeof(double));
}

/**
 * @brief post_scale times FFTW's unnormalized REDFT01 of every line:
 *        x_j = post_scale (X_0 + 2 sum_{k>=1} X_k cos(pi k (j + 1/2) / n)).
 *
 * fft_forward is a committed length-n real DFTI descriptor (complex-complex
 * conjugate-even storage, not in place); its forward transform is used.
 */
inline void mkl_dct3_lines(
    double* data, double* temp, int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_forward, const double* cos_tbl, const double* sin_tbl, double post_scale)
{
    if (n == 1)
    {
        // REDFT01 of a singleton is the identity
        for (int i = 0; i < num_transforms * stride; ++i)
            data[i] *= post_scale;
        return;
    }
    const int complex_size = n / 2 + 1;

    thread_local std::vector<double> slice;
    thread_local std::vector<double> fft_in;
    thread_local std::vector<std::complex<double>> fft_out;
    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(n);
        fft_out.resize(complex_size);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            const int offset = batch * n * stride + s;

            // Extract 1D slice (strided)
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-III preprocessing: twiddle factors (cuHelmholtz preOp_dct3)
            fft_in[0] = slice[0];
            for (int k = 0; k < n / 2; ++k)
            {
                double val_k = slice[k + 1];
                double val_nk = slice[n - k - 1];

                double Ta = val_k + val_nk;
                double Tb = val_k - val_nk;

                fft_in[k + 1] = Ta * sin_tbl[k + 1] + Tb * cos_tbl[k + 1];
                fft_in[n - k - 1] = Ta * cos_tbl[k + 1] - Tb * sin_tbl[k + 1];
            }

            // Forward FFT (r2c)
            mkl_dfti_check(DftiComputeForward(fft_forward, fft_in.data(), fft_out.data()),
                           "DftiComputeForward (DCT-III)");

            // DCT-III postprocessing: rearrange complex to real (cuHelmholtz postOp_dct3)
            temp[offset] = fft_out[0].real() * post_scale;
            for (int k = 1; k <= n / 2; ++k)
            {
                double re = fft_out[k].real();
                double im = fft_out[k].imag();

                temp[offset + (2 * k - 1) * stride] = (re - im) * post_scale;
                if (2 * k < n)
                    temp[offset + (2 * k) * stride] = (re + im) * post_scale;
            }
        }
    }

    std::memcpy(data, temp, static_cast<size_t>(num_transforms) * n * stride * sizeof(double));
}

#endif  // MKL_DCT_LINES_H_

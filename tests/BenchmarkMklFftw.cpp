/**
 * @file BenchmarkMklFftw.cpp
 * @brief Benchmark comparing FFTW vs MKL for DFT and all DCT/DST types (1-4).
 *        Tests both accuracy and performance.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <complex>
#include <vector>
#include <string>
#include <algorithm>

#include <fftw3.h>
#include <mkl_dfti.h>
#include "MklRealTransform.h"

using namespace std;
using Clock = chrono::high_resolution_clock;

constexpr int NUM_WARMUP = 10;
constexpr int NUM_ITERATIONS = 25;
constexpr double ACCURACY_TOL = 1e-8;

//==============================================================================
// DFT (Real-to-Complex FFT) Benchmarks
//==============================================================================

// Test DFT accuracy: compare FFTW and MKL r2c outputs
double test_dft_accuracy_3d(int Nx, int Ny, int Nz)
{
    int total_real = Nx * Ny * Nz;
    int total_complex = Nx * Ny * (Nz / 2 + 1);

    vector<double> input(total_real);

    // Initialize with deterministic random data
    mt19937 gen(5678 + Nx * 100 + Ny * 10 + Nz);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < total_real; ++i)
        input[i] = dist(gen);

    // FFTW r2c
    vector<double> fftw_in = input;
    vector<complex<double>> fftw_out(total_complex);
    fftw_plan plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz,
                                          fftw_in.data(),
                                          reinterpret_cast<fftw_complex*>(fftw_out.data()),
                                          FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // MKL DFTI r2c
    vector<double> mkl_in = input;
    vector<complex<double>> mkl_out(total_complex);

    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG dims[3] = {Nx, Ny, Nz};
    DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    MKL_LONG ostrides[4] = {0, Ny * (Nz/2+1), Nz/2+1, 1};
    DftiSetValue(handle, DFTI_OUTPUT_STRIDES, ostrides);
    DftiCommitDescriptor(handle);
    DftiComputeForward(handle, mkl_in.data(), mkl_out.data());
    DftiFreeDescriptor(&handle);

    // Compare
    double max_err = 0.0;
    for (int i = 0; i < total_complex; ++i)
    {
        double err = abs(fftw_out[i] - mkl_out[i]);
        max_err = max(max_err, err);
    }
    return max_err;
}

// Benchmark FFTW DFT r2c
double benchmark_fftw_dft_3d(int Nx, int Ny, int Nz, int num_iter)
{
    int total_real = Nx * Ny * Nz;
    int total_complex = Nx * Ny * (Nz / 2 + 1);

    vector<double> rdata(total_real);
    vector<complex<double>> cdata(total_complex);

    mt19937 gen(42);
    normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < total_real; ++i)
        rdata[i] = dist(gen);

    fftw_plan plan = fftw_plan_dft_r2c_3d(Nx, Ny, Nz,
                                          rdata.data(),
                                          reinterpret_cast<fftw_complex*>(cdata.data()),
                                          FFTW_ESTIMATE);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; ++i)
        fftw_execute(plan);

    auto t_start = Clock::now();
    for (int i = 0; i < num_iter; ++i)
        fftw_execute(plan);
    auto t_end = Clock::now();

    fftw_destroy_plan(plan);

    return chrono::duration<double, micro>(t_end - t_start).count() / num_iter;
}

// Benchmark MKL DFTI r2c
double benchmark_mkl_dft_3d(int Nx, int Ny, int Nz, int num_iter)
{
    int total_real = Nx * Ny * Nz;
    int total_complex = Nx * Ny * (Nz / 2 + 1);

    vector<double> rdata(total_real);
    vector<complex<double>> cdata(total_complex);

    mt19937 gen(42);
    normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < total_real; ++i)
        rdata[i] = dist(gen);

    DFTI_DESCRIPTOR_HANDLE handle;
    MKL_LONG dims[3] = {Nx, Ny, Nz};
    DftiCreateDescriptor(&handle, DFTI_DOUBLE, DFTI_REAL, 3, dims);
    DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    MKL_LONG ostrides[4] = {0, Ny * (Nz/2+1), Nz/2+1, 1};
    DftiSetValue(handle, DFTI_OUTPUT_STRIDES, ostrides);
    DftiCommitDescriptor(handle);

    // Warmup
    for (int i = 0; i < NUM_WARMUP; ++i)
        DftiComputeForward(handle, rdata.data(), cdata.data());

    auto t_start = Clock::now();
    for (int i = 0; i < num_iter; ++i)
        DftiComputeForward(handle, rdata.data(), cdata.data());
    auto t_end = Clock::now();

    DftiFreeDescriptor(&handle);

    return chrono::duration<double, micro>(t_end - t_start).count() / num_iter;
}

//==============================================================================
// DCT/DST Benchmarks
//==============================================================================

fftw_r2r_kind get_fftw_kind(MklTransformType type)
{
    switch (type)
    {
        case MKL_DCT_1: return FFTW_REDFT00;
        case MKL_DCT_2: return FFTW_REDFT10;
        case MKL_DCT_3: return FFTW_REDFT01;
        case MKL_DCT_4: return FFTW_REDFT11;
        case MKL_DST_1: return FFTW_RODFT00;
        case MKL_DST_2: return FFTW_RODFT10;
        case MKL_DST_3: return FFTW_RODFT01;
        case MKL_DST_4: return FFTW_RODFT11;
        default: return FFTW_REDFT00;
    }
}

double max_abs_diff(const vector<double>& a, const vector<double>& b)
{
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        err = max(err, abs(a[i] - b[i]));
    return err;
}

double test_accuracy_3d(int Nx, int Ny, int Nz, MklTransformType type)
{
    int total = Nx * Ny * Nz;
    vector<double> input(total);

    mt19937 gen(1234 + Nx * 100 + Ny * 10 + Nz + static_cast<int>(type));
    uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < total; ++i)
        input[i] = dist(gen);

    // FFTW
    vector<double> fftw_out = input;
    fftw_r2r_kind kind = get_fftw_kind(type);
    fftw_r2r_kind kinds[3] = {kind, kind, kind};
    int dims[3] = {Nx, Ny, Nz};
    fftw_plan plan = fftw_plan_r2r(3, dims, fftw_out.data(), fftw_out.data(), kinds, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // MKL
    vector<double> mkl_out = input;
    MklRealTransform3D transform(Nx, Ny, Nz, type);
    transform.execute(mkl_out.data());

    return max_abs_diff(fftw_out, mkl_out);
}

double benchmark_fftw_3d(int Nx, int Ny, int Nz, MklTransformType type, int num_iter)
{
    int total = Nx * Ny * Nz;
    vector<double> data(total);

    mt19937 gen(42);
    normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < total; ++i)
        data[i] = dist(gen);

    fftw_r2r_kind kind = get_fftw_kind(type);
    fftw_r2r_kind kinds[3] = {kind, kind, kind};
    int dims[3] = {Nx, Ny, Nz};

    fftw_plan plan = fftw_plan_r2r(3, dims, data.data(), data.data(), kinds, FFTW_ESTIMATE);

    for (int i = 0; i < NUM_WARMUP; ++i)
        fftw_execute(plan);

    auto t_start = Clock::now();
    for (int i = 0; i < num_iter; ++i)
        fftw_execute(plan);
    auto t_end = Clock::now();

    fftw_destroy_plan(plan);

    return chrono::duration<double, micro>(t_end - t_start).count() / num_iter;
}

double benchmark_mkl_3d(int Nx, int Ny, int Nz, MklTransformType type, int num_iter)
{
    int total = Nx * Ny * Nz;
    vector<double> data(total);

    mt19937 gen(42);
    normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < total; ++i)
        data[i] = dist(gen);

    MklRealTransform3D transform(Nx, Ny, Nz, type);

    for (int i = 0; i < NUM_WARMUP; ++i)
        transform.execute(data.data());

    auto t_start = Clock::now();
    for (int i = 0; i < num_iter; ++i)
        transform.execute(data.data());
    auto t_end = Clock::now();

    return chrono::duration<double, micro>(t_end - t_start).count() / num_iter;
}

//==============================================================================
// Main
//==============================================================================

int main()
{
    cout << string(80, '=') << endl;
    cout << "Transform Benchmark: FFTW vs MKL (DFT + DCT/DST 1-4)" << endl;
    cout << "Tests: Accuracy (tol=" << scientific << setprecision(0) << ACCURACY_TOL << ") and Performance" << endl;
    cout << "Iterations: " << NUM_ITERATIONS << " (warmup: " << NUM_WARMUP << ")" << endl;
    cout << string(80, '=') << endl << endl;

    vector<int> grid_sizes = {32, 48, 64};

    bool all_passed = true;

    // Print header
    cout << setw(8) << "Grid"
         << setw(10) << "Type"
         << setw(12) << "MaxErr"
         << setw(14) << "FFTW (us)"
         << setw(14) << "MKL (us)"
         << setw(10) << "Speedup"
         << endl;
    cout << string(68, '-') << endl;

    // DFT benchmarks
    for (int N : grid_sizes)
    {
        string grid_str = to_string(N) + "^3";

        double accuracy_err = test_dft_accuracy_3d(N, N, N);
        double fftw_time = benchmark_fftw_dft_3d(N, N, N, NUM_ITERATIONS);
        double mkl_time = benchmark_mkl_dft_3d(N, N, N, NUM_ITERATIONS);

        double speedup = fftw_time / mkl_time;
        bool acc_pass = (accuracy_err < ACCURACY_TOL);
        if (!acc_pass)
            all_passed = false;

        cout << setw(8) << grid_str
             << setw(10) << "DFT"
             << setw(12) << scientific << setprecision(1) << accuracy_err
             << setw(14) << fixed << setprecision(1) << fftw_time
             << setw(14) << fixed << setprecision(1) << mkl_time
             << setw(9) << fixed << setprecision(2) << speedup << "x"
             << endl;
    }
    cout << string(68, '-') << endl;

    // DCT/DST benchmarks
    struct TypeInfo {
        MklTransformType type;
        string name;
    };
    vector<TypeInfo> types = {
        {MKL_DCT_1, "DCT-1"},
        {MKL_DCT_2, "DCT-2"},
        {MKL_DCT_3, "DCT-3"},
        {MKL_DCT_4, "DCT-4"},
        {MKL_DST_1, "DST-1"},
        {MKL_DST_2, "DST-2"},
        {MKL_DST_3, "DST-3"},
        {MKL_DST_4, "DST-4"},
    };

    for (int N : grid_sizes)
    {
        string grid_str = to_string(N) + "^3";

        for (const auto& t : types)
        {
            double accuracy_err = 0.0;
            double fftw_time = 0.0;
            double mkl_time = 0.0;
            bool error_occurred = false;

            try {
                accuracy_err = test_accuracy_3d(N, N, N, t.type);
                fftw_time = benchmark_fftw_3d(N, N, N, t.type, NUM_ITERATIONS);
                mkl_time = benchmark_mkl_3d(N, N, N, t.type, NUM_ITERATIONS);
            } catch (const exception& e) {
                cerr << "Error for " << t.name << " at " << grid_str << ": " << e.what() << endl;
                error_occurred = true;
            }

            if (!error_occurred) {
                double speedup = fftw_time / mkl_time;
                bool acc_pass = (accuracy_err < ACCURACY_TOL);

                if (!acc_pass)
                    all_passed = false;

                cout << setw(8) << grid_str
                     << setw(10) << t.name
                     << setw(12) << scientific << setprecision(1) << accuracy_err
                     << setw(14) << fixed << setprecision(1) << fftw_time
                     << setw(14) << fixed << setprecision(1) << mkl_time
                     << setw(9) << fixed << setprecision(2) << speedup << "x"
                     << endl;
            } else {
                all_passed = false;
                cout << setw(8) << grid_str
                     << setw(10) << t.name
                     << "  (error)" << endl;
            }
        }
        cout << string(68, '-') << endl;
    }

    cout << endl;
    cout << "Note: Time is for one forward transform." << endl;
    cout << "Speedup > 1.0 means MKL is faster than FFTW." << endl;
    cout << string(80, '=') << endl;

    if (all_passed) {
        cout << "BenchmarkMklFftw: PASSED" << endl;
        return 0;
    } else {
        cout << "BenchmarkMklFftw: FAILED" << endl;
        return 1;
    }
}

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <fftw3.h>

#include "MklRealTransform.h"

static void fill_random(std::vector<double>& data, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& v : data)
        v = dist(rng);
}

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b)
{
    double err = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

static fftw_r2r_kind fftw_kind(MklTransformType type)
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
        default:        return FFTW_REDFT00;
    }
}

static void check_1d(int N, MklTransformType type, double tol)
{
    std::vector<double> in(N);
    fill_random(in, 1000 + N + static_cast<int>(type));

    std::vector<double> fftw_out = in;
    std::vector<double> mkl_out = in;

    fftw_plan plan = fftw_plan_r2r_1d(N,
                                      fftw_out.data(),
                                      fftw_out.data(),
                                      fftw_kind(type),
                                      FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    MklRealTransform1D mkl(N, type);
    mkl.execute(mkl_out.data());

    double err = max_abs_diff(fftw_out, mkl_out);
    if (err > tol)
    {
        std::cerr << "1D mismatch (" << getTransformName(type)
                  << ", N=" << N << "): max error " << err << "\n";
        std::exit(1);
    }
}

static void check_2d(int Nx, int Ny, MklTransformType type, double tol)
{
    const int total = Nx * Ny;
    std::vector<double> in(total);
    fill_random(in, 2000 + Nx * 10 + Ny + static_cast<int>(type));

    std::vector<double> fftw_out = in;
    std::vector<double> mkl_out = in;

    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny,
                                      fftw_out.data(),
                                      fftw_out.data(),
                                      fftw_kind(type),
                                      fftw_kind(type),
                                      FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    MklRealTransform2D mkl(Nx, Ny, type);
    mkl.execute(mkl_out.data());

    double err = max_abs_diff(fftw_out, mkl_out);
    if (err > tol)
    {
        std::cerr << "2D mismatch (" << getTransformName(type)
                  << ", Nx=" << Nx << ", Ny=" << Ny
                  << "): max error " << err << "\n";
        std::exit(1);
    }
}

static void check_3d(int Nx, int Ny, int Nz, MklTransformType type, double tol)
{
    const int total = Nx * Ny * Nz;
    std::vector<double> in(total);
    fill_random(in, 3000 + Nx * 100 + Ny * 10 + Nz + static_cast<int>(type));

    std::vector<double> fftw_out = in;
    std::vector<double> mkl_out = in;

    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz,
                                      fftw_out.data(),
                                      fftw_out.data(),
                                      fftw_kind(type),
                                      fftw_kind(type),
                                      fftw_kind(type),
                                      FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    MklRealTransform3D mkl(Nx, Ny, Nz, type);
    mkl.execute(mkl_out.data());

    double err = max_abs_diff(fftw_out, mkl_out);
    if (err > tol)
    {
        std::cerr << "3D mismatch (" << getTransformName(type)
                  << ", Nx=" << Nx << ", Ny=" << Ny << ", Nz=" << Nz
                  << "): max error " << err << "\n";
        std::exit(1);
    }
}

int main()
{
    const double tol = 1e-8;

    const std::vector<MklTransformType> types = {
        MKL_DCT_1, MKL_DCT_2, MKL_DCT_3, MKL_DCT_4,
        MKL_DST_1, MKL_DST_2, MKL_DST_3, MKL_DST_4
    };

    for (auto type : types)
    {
        check_1d(17, type, tol);
        check_2d(8, 10, type, tol);
        check_3d(6, 5, 4, type, tol);
    }

    std::cout << "TestMklFftwAccuracyAllTypes: PASSED\n";
    return 0;
}

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "MklRealTransform.h"

static void fill_random(std::vector<double>& data, int seed_offset)
{
    std::mt19937 rng(12345 + seed_offset);
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

static void check_roundtrip_1d(int N,
                               MklTransformType fwd_type,
                               MklTransformType bwd_type,
                               double norm,
                               const std::string& label)
{
    std::vector<double> data(N);
    fill_random(data, static_cast<int>(fwd_type) * 10 + N);
    std::vector<double> original = data;

    MklRealTransform1D fwd(N, fwd_type);
    MklRealTransform1D bwd(N, bwd_type);

    fwd.execute(data.data());
    bwd.execute(data.data());

    for (double& v : data)
        v /= norm;

    double err = max_abs_diff(data, original);
    if (err > 1e-10)
    {
        std::cerr << "1D round-trip failed: " << label
                  << " (N=" << N << ", err=" << err << ")\n";
        std::exit(1);
    }
}

static void check_roundtrip_2d(int Nx, int Ny,
                               MklTransformType fwd_x,
                               MklTransformType fwd_y,
                               MklTransformType bwd_x,
                               MklTransformType bwd_y,
                               double norm,
                               const std::string& label)
{
    const int total = Nx * Ny;
    std::vector<double> data(total);
    fill_random(data, static_cast<int>(fwd_x) * 100 + static_cast<int>(fwd_y) * 10 + Nx + Ny);
    std::vector<double> original = data;

    MklRealTransform2D fwd(Nx, Ny, fwd_x, fwd_y);
    MklRealTransform2D bwd(Nx, Ny, bwd_x, bwd_y);

    fwd.execute(data.data());
    bwd.execute(data.data());

    for (double& v : data)
        v /= norm;

    double err = max_abs_diff(data, original);
    if (err > 1e-10)
    {
        std::cerr << "2D round-trip failed: " << label
                  << " (Nx=" << Nx << ", Ny=" << Ny << ", err=" << err << ")\n";
        std::exit(1);
    }
}

static void check_roundtrip_3d(int Nx, int Ny, int Nz,
                               MklTransformType type,
                               double norm,
                               const std::string& label)
{
    const int total = Nx * Ny * Nz;
    std::vector<double> data(total);
    fill_random(data, static_cast<int>(type) * 100 + Nx + Ny + Nz);
    std::vector<double> original = data;

    MklRealTransform3D fwd(Nx, Ny, Nz, type);
    MklRealTransform3D bwd(Nx, Ny, Nz, type);

    fwd.execute(data.data());
    bwd.execute(data.data());

    for (double& v : data)
        v /= norm;

    double err = max_abs_diff(data, original);
    if (err > 1e-10)
    {
        std::cerr << "3D round-trip failed: " << label
                  << " (Nx=" << Nx << ", Ny=" << Ny << ", Nz=" << Nz
                  << ", err=" << err << ")\n";
        std::exit(1);
    }
}

int main()
{
    const int N = 9;

    // 1D: DCT-1..4 and DST-1..4
    check_roundtrip_1d(N, MKL_DCT_1, MKL_DCT_1, 2.0 * (N - 1), "DCT-1");
    check_roundtrip_1d(N, MKL_DCT_2, MKL_DCT_3, 2.0 * N, "DCT-2->3");
    check_roundtrip_1d(N, MKL_DCT_3, MKL_DCT_2, 2.0 * N, "DCT-3->2");
    check_roundtrip_1d(N, MKL_DCT_4, MKL_DCT_4, 2.0 * N, "DCT-4");

    check_roundtrip_1d(N, MKL_DST_1, MKL_DST_1, 2.0 * (N + 1), "DST-1");
    check_roundtrip_1d(N, MKL_DST_2, MKL_DST_3, 2.0 * N, "DST-2->3");
    check_roundtrip_1d(N, MKL_DST_3, MKL_DST_2, 2.0 * N, "DST-3->2");
    check_roundtrip_1d(N, MKL_DST_4, MKL_DST_4, 2.0 * N, "DST-4");

    // 2D mixed: DCT-2 (x) + DST-3 (y), inverse DCT-3 (x) + DST-2 (y)
    {
        const int Nx = 8;
        const int Ny = 6;
        const double norm = (2.0 * Nx) * (2.0 * Ny);
        check_roundtrip_2d(Nx, Ny,
                           MKL_DCT_2, MKL_DST_3,
                           MKL_DCT_3, MKL_DST_2,
                           norm, "2D mixed DCT2/DST3");
    }

    // 3D: DCT-4 self-inverse
    {
        const int Nx = 6;
        const int Ny = 5;
        const int Nz = 4;
        const double norm = (2.0 * Nx) * (2.0 * Ny) * (2.0 * Nz);
        check_roundtrip_3d(Nx, Ny, Nz, MKL_DCT_4, norm, "3D DCT-4");
    }

    std::cout << "TestMklRealTransform: PASSED\n";
    return 0;
}

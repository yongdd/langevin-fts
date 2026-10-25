/**
 * @file Recursive3mHelpers.h
 * @brief Host-side pieces of the recursive 2x2y2z (3m) crystallographic FFT
 *        shared by the CPU (CrysFFTRecursive3mBase) and CUDA
 *        (CudaCrysFFTRecursive3m) engines.
 *
 * The algorithm is described in "Accelerated pseudo-spectral method of
 * self-consistent field theory via crystallographic fast Fourier transform",
 * Macromolecules (2020). Both engines build their k-space tables with these
 * functions, so the fold, its twiddle factors and the stress multipliers have
 * one definition; the translation-parity permutation of the fold lives in
 * Recursive3mFoldParity.h.
 */

#ifndef RECURSIVE_3M_HELPERS_H_
#define RECURSIVE_3M_HELPERS_H_

#include <array>
#include <cmath>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

namespace crys_recursive_3m {

constexpr double kPi = 3.14159265358979323846;

/// Phase gradients of the seven non-identity twiddle factors for the m3
/// generator translations g (3 x 3, row per generator) on the grid s.
inline std::vector<std::array<double, 3>> generate_m3_operations(
    const std::array<double, 9>& g, const std::array<int, 3>& s)
{
    return {
        {-2.0 * kPi * (g[0]),                              -2.0 * kPi * (g[1]),                              -2.0 * kPi * (g[2] - 1.0 / s[2])},
        {-2.0 * kPi * (g[3]),                              -2.0 * kPi * (g[4] - 1.0 / s[1]),                 -2.0 * kPi * (g[5])},
        {-2.0 * kPi * (g[0] + g[3]),                       -2.0 * kPi * (-g[1] + g[4] - 1.0 / s[1]),         -2.0 * kPi * (g[2] + g[5] - 1.0 / s[2])},
        {-2.0 * kPi * (g[6] - 1.0 / s[0]),                 -2.0 * kPi * (g[7]),                              -2.0 * kPi * (g[8])},
        {-2.0 * kPi * (g[6] - g[0] - 1.0 / s[0]),          -2.0 * kPi * (g[1] + g[7]),                       -2.0 * kPi * (g[2] + g[8] - 1.0 / s[2])},
        {-2.0 * kPi * (-g[3] + g[6] - 1.0 / s[0]),         -2.0 * kPi * (g[4] + g[7] - 1.0 / s[1]),          -2.0 * kPi * (g[5] + g[8])},
        {-2.0 * kPi * (-g[0] - g[3] + g[6] - 1.0 / s[0]),  -2.0 * kPi * (-g[1] + g[4] + g[7] - 1.0 / s[1]),  -2.0 * kPi * (g[2] + g[5] + g[8] - 1.0 / s[2])}
    };
}

/// cos and sin of ix*p0 + iy*p1 + iz*p2 over range, for each (cos, sin, p) task.
inline void twiddle_factor(
    const std::vector<std::tuple<double*, double*, std::array<double, 3>>>& data,
    const std::array<int, 3>& range)
{
    const int Nx = range[0];
    const int Ny = range[1];
    const int Nz = range[2];

    for (size_t itr_task = 0; itr_task < data.size(); ++itr_task)
    {
        double* dstcos = std::get<0>(data[itr_task]);
        double* dstsin = std::get<1>(data[itr_task]);
        auto paras = std::get<2>(data[itr_task]);
        for (int ix = 0; ix < Nx; ++ix)
        {
            for (int iy = 0; iy < Ny; ++iy)
            {
                size_t base = static_cast<size_t>(ix * Ny + iy) * Nz;
                for (int iz = 0; iz < Nz; ++iz)
                {
                    double angle = ix * paras[0] + iy * paras[1] + iz * paras[2];
                    dstcos[base + iz] = std::cos(angle);
                    dstsin[base + iz] = std::sin(angle);
                }
            }
        }
    }
}

/// Split the grid src of the given size into its eight octants dst[part]
/// (part bits: 4 = upper x half, 2 = upper y half, 1 = upper z half).
inline void mat_split(
    const double* src,
    const std::array<double*, 8>& dst,
    const std::array<int, 3>& size)
{
    const int Nx = size[0];
    const int Ny = size[1];
    const int Nz = size[2];

    const int Nx2 = Nx / 2;
    const int Ny2 = Ny / 2;
    const int Nz2 = Nz / 2;

    for (int part = 0; part < 8; ++part)
    {
        const double* local_src = src
            + ((part & 4) >> 2) * Nx2 * Ny * Nz
            + ((part & 2) >> 1) * Ny2 * Nz
            + (part & 1) * Nz2;
        double* local_dst = dst[part];

        for (int ix = 0; ix < Nx2; ++ix)
        {
            for (int iy = 0; iy < Ny2; ++iy)
            {
                const double* src_row = local_src + (ix * Ny + iy) * Nz;
                double* dst_row = local_dst + (ix * Ny2 + iy) * Nz2;
                std::memcpy(dst_row, src_row, sizeof(double) * Nz2);
            }
        }
    }
}

/// In-place butterflies (a, b) -> (a + b, a - b) over the octant pairs of seq.
inline void add_sub_by_seq(
    const std::array<double*, 8>& dst,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        double* dst0 = dst[seq[itr_seq].first];
        double* dst1 = dst[seq[itr_seq].second];
        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                double temp = dst0[base + itr_z] + dst1[base + itr_z];
                dst1[base + itr_z] = dst0[base + itr_z] - dst1[base + itr_z];
                dst0[base + itr_z] = temp;
            }
        }
    }
}

/// dst = src_mul * src (real and imaginary parts), followed by the butterflies
/// of add_sub_by_seq, over the octant pairs of seq.
inline void mul_add_sub_by_seq(
    const std::array<double*, 8>& dst_re,
    const std::array<double*, 8>& dst_im,
    const std::array<const double*, 8>& src_re,
    const std::array<const double*, 8>& src_im,
    const std::array<const double*, 8>& src_mul,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        int i0 = seq[itr_seq].first;
        int i1 = seq[itr_seq].second;
        double* dst_re0 = dst_re[i0];
        double* dst_re1 = dst_re[i1];
        double* dst_im0 = dst_im[i0];
        double* dst_im1 = dst_im[i1];
        const double* src_re0 = src_re[i0];
        const double* src_re1 = src_re[i1];
        const double* src_im0 = src_im[i0];
        const double* src_im1 = src_im[i1];
        const double* mul0 = src_mul[i0];
        const double* mul1 = src_mul[i1];

        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                size_t idx = base + itr_z;
                dst_re0[idx] = mul0[idx] * src_re0[idx];
                dst_im0[idx] = mul0[idx] * src_im0[idx];
                dst_re1[idx] = mul1[idx] * src_re1[idx];
                dst_im1[idx] = mul1[idx] * src_im1[idx];

                double tempre = dst_re0[idx] + dst_re1[idx];
                dst_re1[idx] = dst_re0[idx] - dst_re1[idx];
                dst_re0[idx] = tempre;

                double tempim = dst_im0[idx] + dst_im1[idx];
                dst_im1[idx] = dst_im0[idx] - dst_im1[idx];
                dst_im0[idx] = tempim;
            }
        }
    }
}

/// Stress multipliers: k_d² summed over some axes (Kxy2: x and y, K2: all),
/// times exp(-k² coeff) for the Exp types (discrete chains).
enum class MultiplierType
{
    Kx2,
    Ky2,
    Kz2,
    ExpKx2,
    ExpKy2,
    ExpKz2,
    Kxy2,
    K2,
    ExpKxy2,
    ExpK2
};

/// Axes a multiplier sums k_d² over (bit 0: x, 1: y, 2: z).
inline int multiplier_axes(MultiplierType type)
{
    switch (type)
    {
        case MultiplierType::Kx2:  case MultiplierType::ExpKx2:  return 1;
        case MultiplierType::Ky2:  case MultiplierType::ExpKy2:  return 2;
        case MultiplierType::Kz2:  case MultiplierType::ExpKz2:  return 4;
        case MultiplierType::Kxy2: case MultiplierType::ExpKxy2: return 3;
        case MultiplierType::K2:   case MultiplierType::ExpK2:   return 7;
    }
    return 0;
}

/// Whether a multiplier carries the bond factor exp(-k² coeff).
inline bool multiplier_has_bond(MultiplierType type)
{
    return type == MultiplierType::ExpKx2 || type == MultiplierType::ExpKy2 || type == MultiplierType::ExpKz2
        || type == MultiplierType::ExpKxy2 || type == MultiplierType::ExpK2;
}

/**
 * @brief value_at(kx², ky², kz²) / M_logical on the logical grid nx of the
 *        orthogonal cell (lengths cell[0..2]), k_d = 2π m_d / L_d with the
 *        centered index |m_d| <= N_d/2.
 *
 * The array the fold splits into the eight octant tables, for the Boltzmann
 * factors and the stress multipliers alike.
 */
template <typename ValueAt>
inline std::vector<double> logical_k_table(
    const std::array<int, 3>& nx, const std::array<double, 6>& cell, ValueAt&& value_at)
{
    std::vector<double> kx(nx[0]);
    std::vector<double> ky(nx[1]);
    std::vector<double> kz(nx[2]);

    double factor_Lx = 2.0 * kPi / cell[0];
    factor_Lx *= factor_Lx;
    double factor_Ly = 2.0 * kPi / cell[1];
    factor_Ly *= factor_Ly;
    double factor_Lz = 2.0 * kPi / cell[2];
    factor_Lz *= factor_Lz;

    for (int ix = 0; ix < nx[0]; ++ix)
    {
        int temp = (ix > nx[0] / 2) ? (nx[0] - ix) : ix;
        kx[ix] = temp * temp * factor_Lx;
    }
    for (int iy = 0; iy < nx[1]; ++iy)
    {
        int temp = (iy > nx[1] / 2) ? (nx[1] - iy) : iy;
        ky[iy] = temp * temp * factor_Ly;
    }
    for (int iz = 0; iz < nx[2]; ++iz)
    {
        int temp = (iz > nx[2] / 2) ? (nx[2] - iz) : iz;
        kz[iz] = temp * temp * factor_Lz;
    }

    const size_t M_logical = static_cast<size_t>(nx[0]) * nx[1] * nx[2];
    std::vector<double> tempmat(M_logical);
    const double factor = 1.0 / static_cast<double>(M_logical);
    for (int ix = 0; ix < nx[0]; ++ix)
    {
        for (int iy = 0; iy < nx[1]; ++iy)
        {
            size_t base = static_cast<size_t>(ix * nx[1] + iy) * nx[2];
            for (int iz = 0; iz < nx[2]; ++iz)
                tempmat[base + iz] = value_at(kx[ix], ky[iy], kz[iz]) * factor;
        }
    }
    return tempmat;
}

}  // namespace crys_recursive_3m

#endif  // RECURSIVE_3M_HELPERS_H_

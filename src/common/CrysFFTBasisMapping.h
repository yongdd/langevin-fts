/**
 * @file CrysFFTBasisMapping.h
 * @brief Index maps between a CrysFFT physical sub-grid and a space-group reduced basis.
 *
 * The crystallographic FFT fast paths work on a physical sub-grid of the full
 * mesh: the (Nx/2, Ny/2, Nz/2) octant for the Pmmm DCT and the recursive 3m
 * transforms (sampled at consecutive or at even full-grid indices), or the
 * lower half in z for the z-mirror (ObliqueZ) transform. The maps built here
 * relate every physical point to its full-grid index and to the reduced-basis
 * (orbit) index of the space group. They are platform independent; the CPU and
 * CUDA solvers keep them in their own storage layouts.
 */

#ifndef CRYSFFT_BASIS_MAPPING_H_
#define CRYSFFT_BASIS_MAPPING_H_

#include <array>
#include <string>
#include <vector>

#include "Exception.h"

struct CrysFFTBasisMapping
{
    int M_phys = 0;                    ///< Number of physical grid points
    int M_reduced = 0;                 ///< Number of reduced-basis points
    std::vector<int> phys_to_full;     ///< physical index -> full-grid index
    std::vector<int> phys_to_reduced;  ///< physical index -> reduced-basis index
    std::vector<int> reduced_to_phys;  ///< reduced index -> first physical index mapped to it (-1: none)
    bool covers_all_reduced = false;   ///< Every reduced-basis point is reached from the physical grid
    bool identity = false;             ///< M_phys == M_reduced and phys_to_reduced[i] == i for all i
};

namespace crysfft_mapping_detail
{
/// Derive the reduced-basis maps and the flags from phys_to_full.
inline void finish(CrysFFTBasisMapping& map, const std::vector<int>& full_to_reduced)
{
    map.M_phys = static_cast<int>(map.phys_to_full.size());
    map.phys_to_reduced.resize(map.M_phys);
    map.reduced_to_phys.assign(map.M_reduced, -1);
    std::vector<int> coverage(map.M_reduced, 0);
    for (int i = 0; i < map.M_phys; ++i)
    {
        const int r = full_to_reduced[map.phys_to_full[i]];
        map.phys_to_reduced[i] = r;
        if (map.reduced_to_phys[r] < 0)
            map.reduced_to_phys[r] = i;
        coverage[r] += 1;
    }
    map.covers_all_reduced = true;
    for (int r = 0; r < map.M_reduced; ++r)
    {
        if (coverage[r] == 0)
        {
            map.covers_all_reduced = false;
            break;
        }
    }
    map.identity = (map.M_phys == map.M_reduced);
    for (int i = 0; map.identity && i < map.M_phys; ++i)
    {
        if (map.phys_to_reduced[i] != i)
            map.identity = false;
    }
}
}  // namespace crysfft_mapping_detail

/**
 * @brief Octant physical grid (Nx/2, Ny/2, Nz/2).
 *
 * @param even_indices true: sample the full grid at even indices (recursive 3m
 *                     transform); false: consecutive indices (Pmmm DCT).
 */
inline CrysFFTBasisMapping build_crysfft_mapping_octant(
    const std::array<int, 3>& nx, const std::vector<int>& full_to_reduced,
    int M_reduced, bool even_indices)
{
    CrysFFTBasisMapping map;
    map.M_reduced = M_reduced;
    const int Nx2 = nx[0] / 2, Ny2 = nx[1] / 2, Nz2 = nx[2] / 2;
    map.phys_to_full.reserve(static_cast<size_t>(Nx2) * Ny2 * Nz2);
    for (int ix = 0; ix < Nx2; ++ix)
    {
        const int fx = even_indices ? (2 * ix) : ix;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            const int fy = even_indices ? (2 * iy) : iy;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                const int fz = even_indices ? (2 * iz) : iz;
                map.phys_to_full.push_back((fx * nx[1] + fy) * nx[2] + fz);
            }
        }
    }
    crysfft_mapping_detail::finish(map, full_to_reduced);
    return map;
}

/**
 * @brief Lower z half (Nx, Ny, Nz/2) for the z-mirror (ObliqueZ) transform.
 *
 * @param z_shift Offset of the physical half along z (with wrap-around); the
 *                mirror plane sits at z_shift for a glide translation of c/2.
 */
inline CrysFFTBasisMapping build_crysfft_mapping_z_mirror(
    const std::array<int, 3>& nx, const std::vector<int>& full_to_reduced,
    int M_reduced, int z_shift)
{
    CrysFFTBasisMapping map;
    map.M_reduced = M_reduced;
    const int Nz2 = nx[2] / 2;
    map.phys_to_full.reserve(static_cast<size_t>(nx[0]) * nx[1] * Nz2);
    for (int ix = 0; ix < nx[0]; ++ix)
    {
        for (int iy = 0; iy < nx[1]; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + z_shift;
                if (iz_full >= nx[2])
                    iz_full -= nx[2];
                map.phys_to_full.push_back((ix * nx[1] + iy) * nx[2] + iz_full);
            }
        }
    }
    crysfft_mapping_detail::finish(map, full_to_reduced);
    return map;
}

/**
 * @brief Whether a mapping can be used for the CrysFFT path.
 *
 * With a physical basis enabled the space group already stores its fields on
 * the physical grid, so the mapping must be the identity; a mismatch is a
 * configuration error and throws `mismatch_message`. Without a physical basis
 * the mapping is usable when it reaches every reduced-basis point.
 */
inline bool crysfft_mapping_usable(
    const CrysFFTBasisMapping& map, bool physical_basis_enabled, const char* mismatch_message)
{
    if (physical_basis_enabled)
    {
        if (!map.identity)
            throw_with_line_number(std::string(mismatch_message));
        return true;
    }
    return map.covers_all_reduced;
}

/**
 * @brief Axis groups of the CrysFFT stress fast path, as bit masks (1: x, 2: y, 4: z).
 *
 * A per-axis multiplier k_d² is not invariant under point-group operations
 * that permute axes (cubic: x, y, z; tetragonal: x, y), so on the reduced basis
 * only its sum over such a group of axes is well defined. The stress routines
 * apply one multiplier Σ_{d in group} k_d² per group and give every axis of the
 * group the group sum divided by the group size; the kernel q1·q2 is symmetric,
 * so this is the exact value of each diagonal component.
 */
inline std::vector<int> crysfft_stress_axis_groups(const std::string& crystal_system)
{
    if (crystal_system == "Cubic")
        return {7};
    if (crystal_system == "Tetragonal")
        return {3, 4};
    return {1, 2, 4};
}

#endif  // CRYSFFT_BASIS_MAPPING_H_

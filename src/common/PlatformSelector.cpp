/**
 * @file PlatformSelector.cpp
 * @brief Implementation of platform selection factory.
 *
 * Implements the abstract factory pattern for selecting computational
 * platforms at runtime. Available platforms are determined by compile-time
 * preprocessor macros (USE_CPU_MKL, USE_CUDA).
 *
 * **Supported Platforms:**
 *
 * - cpu-mkl: Intel MKL-based CPU implementation (if USE_CPU_MKL defined)
 * - cuda: NVIDIA CUDA GPU implementation (if USE_CUDA defined)
 *
 * **Factory Creation:**
 *
 * The create_factory_real() and create_factory_complex() methods return
 * platform-specific AbstractFactory instances that can create all
 * computational objects (ComputationBox, Pseudo, AndersonMixing, etc.).
 *
 * @see AbstractFactory for factory interface
 * @see MklFactory for CPU implementation
 * @see CudaFactory for GPU implementation
 */

#include <iostream>
#include <vector>
#include <string>
#include "Exception.h"

#ifdef USE_CPU_MKL
#include "MklFactory.h"
#endif
#ifdef USE_CUDA
#include "CudaFactory.h"
#include "CudaCommon.h"
#endif
#include "PlatformSelector.h"

/**
 * @brief Get list of available computational platforms.
 *
 * Queries compile-time configuration to determine which platforms
 * are available for use.
 *
 * @return Vector of platform names (e.g., {"cpu-mkl", "cuda"})
 * @throws Exception if no platforms are available
 */
std::vector<std::string> PlatformSelector::avail_platforms()
{
    std::vector<std::string> names;
#ifdef USE_CPU_MKL
    names.push_back("cpu-mkl");
#endif
#ifdef USE_CUDA
    names.push_back("cuda");
#endif
    if(names.size() == 0)
        throw_with_line_number("No available platform");
    return names;
}

/**
 * @brief Create factory for real-valued field computations.
 *
 * Instantiates a platform-specific factory for simulations with
 * real-valued fields (standard SCFT/L-FTS with periodic boundaries).
 *
 * @param platform            Platform name ("cpu-mkl" or "cuda")
 * @param reduce_memory_usage Enable memory-saving mode (CUDA only)
 *
 * @return Pointer to platform-specific AbstractFactory<double>
 * @throws Exception if platform not found or not compiled
 */
AbstractFactory<double>* PlatformSelector::create_factory_real(std::string platform, bool reduce_memory_usage)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory<double>(reduce_memory_usage);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory<double>(reduce_memory_usage);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}

/**
 * @brief Create factory for complex-valued field computations.
 *
 * Instantiates a platform-specific factory for simulations with
 * complex-valued fields (required for non-periodic boundaries or
 * certain field transformations).
 *
 * @param platform            Platform name ("cpu-mkl" or "cuda")
 * @param reduce_memory_usage Enable memory-saving mode (CUDA only)
 *
 * @return Pointer to platform-specific AbstractFactory<std::complex<double>>
 * @throws Exception if platform not found or not compiled
 */
AbstractFactory<std::complex<double>>* PlatformSelector::create_factory_complex(std::string platform, bool reduce_memory_usage)
{
#ifdef USE_CPU_MKL
    if (platform == "cpu-mkl")
        return new MklFactory<std::complex<double>>(reduce_memory_usage);
#endif
#ifdef USE_CUDA
    if (platform == "cuda")
        return new CudaFactory<std::complex<double>>(reduce_memory_usage);
#endif
    throw_with_line_number("Could not find platform '" + platform + "'");
    return NULL;
}
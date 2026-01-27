/**
 * @file PlatformSelector.h
 * @brief Platform detection and factory instantiation for multi-backend support.
 *
 * This header provides the PlatformSelector class which detects available
 * computational platforms and creates the appropriate AbstractFactory instance.
 * It abstracts away the platform selection logic, allowing client code to
 * work with a unified interface regardless of whether CPU or GPU is used.
 *
 * **Supported Platforms:**
 *
 * - **cpu-fftw**: FFTW-based CPU implementation
 *   - Uses OpenMP for parallelization
 *   - Available when compiled with FFTW support
 *
 * - **cuda**: NVIDIA CUDA GPU implementation
 *   - Uses cuFFT for FFT operations
 *   - Requires CUDA-capable GPU
 *   - Available when compiled with CUDA support
 *
 * @see AbstractFactory for the factory interface
 * @see FftwFactory for CPU implementation details
 * @see CudaFactory for GPU implementation details
 *
 * @example
 * @code
 * // Check available platforms
 * std::vector<std::string> platforms = PlatformSelector::avail_platforms();
 * for (const auto& p : platforms) {
 *     std::cout << "Available: " << p << std::endl;
 * }
 * // Output: "cuda", "cpu-fftw"
 *
 * // Create factory for CUDA platform
 * auto* factory = PlatformSelector::create_factory_real("cuda", false);
 *
 * // Create factory for CPU with memory reduction (ignored for CPU)
 * auto* factory_cpu = PlatformSelector::create_factory_real("cpu-fftw", false);
 *
 * // Create factory for complex fields (for certain advanced applications)
 * auto* factory_complex = PlatformSelector::create_factory_complex("cuda", false);
 * @endcode
 */

#ifndef PLATFORM_SELECTOR_H_
#define PLATFORM_SELECTOR_H_

#include <string>
#include <vector>
#include "AbstractFactory.h"

/**
 * @class PlatformSelector
 * @brief Static utility class for platform detection and factory creation.
 *
 * PlatformSelector provides static methods to:
 * 1. Query which computational platforms are available
 * 2. Create AbstractFactory instances for a specified platform
 *
 * **Platform Selection Strategy:**
 *
 * The Python interface typically auto-selects the platform:
 * - 2D/3D simulations: Prefer "cuda" if available, fall back to "cpu-fftw"
 * - 1D simulations: Use "cpu-fftw" (GPU overhead not worthwhile)
 *
 * **Build Configuration:**
 *
 * Available platforms depend on CMake build configuration:
 * - CUDA support: Requires CUDA Toolkit and compatible GPU
 * - FFTW support: Requires FFTW library
 *
 * @note This class has only static methods; do not instantiate it.
 */
class PlatformSelector
{
public:
    /**
     * @brief Get list of available computational platforms.
     *
     * Returns a vector of platform name strings that can be passed to
     * create_factory_real() or create_factory_complex().
     *
     * @return Vector of available platform names, e.g., {"cuda", "cpu-fftw"}
     *
     * @note The order indicates preference (first = fastest/recommended).
     *
     * @example
     * @code
     * auto platforms = PlatformSelector::avail_platforms();
     * if (std::find(platforms.begin(), platforms.end(), "cuda") != platforms.end()) {
     *     std::cout << "CUDA is available" << std::endl;
     * }
     * @endcode
     */
    static std::vector<std::string> avail_platforms();

    /**
     * @brief Create an AbstractFactory for real-valued fields.
     *
     * Creates a platform-specific factory for simulations using real-valued
     * fields (standard SCFT and L-FTS calculations).
     *
     * @param platform          Platform name: "cuda" or "cpu-fftw"
     * @param reduce_memory If true, store only propagator checkpoints instead
     *                          of full histories, recomputing as needed.
     *                          Reduces memory usage but increases computation time.
     *
     * @return AbstractFactory<double>* pointer to the created factory
     *
     * @throws Exception if platform is not available or invalid
     *
     * @note Caller is responsible for deleting the returned factory.
     *
     * @example
     * @code
     * // Create CUDA factory
     * auto* factory = PlatformSelector::create_factory_real("cuda", false);
     *
     * // Create solver with specific numerical method
     * auto* solver = factory->create_propagator_computation(cb, mols, optimizer, "rqm4");
     * // Or for real-space methods:
     * // auto* solver = factory->create_propagator_computation(cb, mols, optimizer, "cn-adi2");
     * @endcode
     */
    static AbstractFactory<double>* create_factory_real(
        std::string platform, bool reduce_memory);

    /**
     * @brief Create an AbstractFactory for complex-valued fields.
     *
     * Creates a platform-specific factory for simulations using complex-valued
     * fields (certain advanced applications with complex order parameters).
     *
     * @param platform          Platform name: "cuda" or "cpu-fftw"
     * @param reduce_memory If true, store only propagator checkpoints instead
     *                          of full histories, recomputing as needed.
     *
     * @return AbstractFactory<std::complex<double>>* pointer to the created factory
     *
     * @throws Exception if platform is not available or invalid
     *
     * @note Complex field support is less commonly used. Most simulations
     *       use real-valued fields via create_factory_real().
     */
    static AbstractFactory<std::complex<double>>* create_factory_complex(
        std::string platform, bool reduce_memory);
};

#endif

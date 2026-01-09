/**
 * @file TemplateInstantiations.h
 * @brief Centralized template instantiation macros for consistent instantiation.
 *
 * This header provides macros to ensure consistent template instantiation
 * across the codebase. All template classes should use these macros at the
 * end of their implementation files.
 *
 * **Supported Types:**
 * - double: Real-valued fields (standard SCFT/L-FTS)
 * - std::complex<double>: Complex-valued fields (advanced applications)
 *
 * **Usage:**
 *
 * At the end of each .cpp/.cu implementation file:
 * @code
 * #include "TemplateInstantiations.h"
 * INSTANTIATE_CLASS(MyTemplateClass);
 * @endcode
 *
 * For classes with multiple template parameters (e.g., FFT with dimension):
 * @code
 * INSTANTIATE_FFT_CLASS(MklFFT);
 * @endcode
 *
 * **Maintenance:**
 *
 * To add a new supported type (e.g., float):
 * 1. Add to INSTANTIATE_CLASS macro
 * 2. Update this documentation
 * 3. Rebuild all affected targets
 *
 * @see PropagatorComputation for usage example
 */

#ifndef TEMPLATE_INSTANTIATIONS_H
#define TEMPLATE_INSTANTIATIONS_H

#include <complex>

/**
 * @brief Instantiate a single-parameter template class for standard types.
 *
 * Creates explicit instantiations for:
 * - ClassName<double>
 * - ClassName<std::complex<double>>
 *
 * @param ClassName The template class to instantiate
 *
 * Example:
 * @code
 * // At end of MyClass.cpp
 * INSTANTIATE_CLASS(MyClass);
 * // Equivalent to:
 * // template class MyClass<double>;
 * // template class MyClass<std::complex<double>>;
 * @endcode
 */
#define INSTANTIATE_CLASS(ClassName) \
    template class ClassName<double>; \
    template class ClassName<std::complex<double>>;

/**
 * @brief Instantiate FFT classes for all dimension and type combinations.
 *
 * Creates explicit instantiations for:
 * - ClassName<double, 1>, ClassName<double, 2>, ClassName<double, 3>
 * - ClassName<std::complex<double>, 1>, ClassName<std::complex<double>, 2>,
 *   ClassName<std::complex<double>, 3>
 *
 * @param ClassName The FFT template class to instantiate
 *
 * Example:
 * @code
 * // At end of MklFFT.cpp
 * INSTANTIATE_FFT_CLASS(MklFFT);
 * @endcode
 */
#define INSTANTIATE_FFT_CLASS(ClassName) \
    template class ClassName<double, 1>; \
    template class ClassName<double, 2>; \
    template class ClassName<double, 3>; \
    template class ClassName<std::complex<double>, 1>; \
    template class ClassName<std::complex<double>, 2>; \
    template class ClassName<std::complex<double>, 3>;

/**
 * @brief Instantiate a class template for real types only.
 *
 * Use when complex instantiation is not needed or not yet implemented.
 *
 * @param ClassName The template class to instantiate
 */
#define INSTANTIATE_CLASS_REAL_ONLY(ClassName) \
    template class ClassName<double>;

/**
 * @brief Instantiate FFT classes for real types only.
 *
 * Use when complex instantiation is not needed or not yet implemented.
 *
 * @param ClassName The FFT template class to instantiate
 */
#define INSTANTIATE_FFT_CLASS_REAL_ONLY(ClassName) \
    template class ClassName<double, 1>; \
    template class ClassName<double, 2>; \
    template class ClassName<double, 3>;

/* Future: Add float support when needed
#define INSTANTIATE_CLASS_ALL(ClassName) \
    template class ClassName<float>; \
    template class ClassName<double>; \
    template class ClassName<std::complex<float>>; \
    template class ClassName<std::complex<double>>;
*/

#endif // TEMPLATE_INSTANTIATIONS_H

/**
 * @file MklDftiLayout.h
 * @brief Direction-aware DFTI layout parameters for old and new MKL releases.
 */

#ifndef MKL_DFTI_LAYOUT_H_
#define MKL_DFTI_LAYOUT_H_

#include <mkl_dfti.h>
#include <mkl_version.h>

namespace mkl_dfti
{
// MKL 2026.2 replaces input/output parameters with forward/backward-domain
// parameters. For a backward transform, the input is in the backward domain
// and the output is in the forward domain. Keep that distinction at call sites.
#if INTEL_MKL_VERSION >= 20260200
inline constexpr auto forward_input_strides = DFTI_FWD_STRIDES;
inline constexpr auto forward_output_strides = DFTI_BWD_STRIDES;
inline constexpr auto backward_input_strides = DFTI_BWD_STRIDES;
inline constexpr auto backward_output_strides = DFTI_FWD_STRIDES;
inline constexpr auto forward_input_distance = DFTI_FWD_DISTANCE;
inline constexpr auto forward_output_distance = DFTI_BWD_DISTANCE;
inline constexpr auto backward_input_distance = DFTI_BWD_DISTANCE;
inline constexpr auto backward_output_distance = DFTI_FWD_DISTANCE;
#else
inline constexpr auto forward_input_strides = DFTI_INPUT_STRIDES;
inline constexpr auto forward_output_strides = DFTI_OUTPUT_STRIDES;
inline constexpr auto backward_input_strides = DFTI_INPUT_STRIDES;
inline constexpr auto backward_output_strides = DFTI_OUTPUT_STRIDES;
inline constexpr auto forward_input_distance = DFTI_INPUT_DISTANCE;
inline constexpr auto forward_output_distance = DFTI_OUTPUT_DISTANCE;
inline constexpr auto backward_input_distance = DFTI_INPUT_DISTANCE;
inline constexpr auto backward_output_distance = DFTI_OUTPUT_DISTANCE;
#endif
}  // namespace mkl_dfti

#endif  // MKL_DFTI_LAYOUT_H_

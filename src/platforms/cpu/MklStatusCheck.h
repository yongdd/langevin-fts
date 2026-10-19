/**
 * @file MklStatusCheck.h
 * @brief Status check for Intel MKL DFTI calls.
 *
 * Every DftiCreateDescriptor / DftiSetValue / DftiCommitDescriptor /
 * DftiComputeForward / DftiComputeBackward call in the MKL backend is wrapped
 * in mkl_dfti_check(), which throws with MKL's own error text when the call
 * did not succeed. Only compiled in the MKL build (USE_CPU_MKL).
 */

#ifndef MKL_STATUS_CHECK_H_
#define MKL_STATUS_CHECK_H_

#include <string>
#include "mkl_dfti.h"
#include "Exception.h"

/**
 * @brief Throw when an MKL DFTI call returned an error status.
 *
 * @param status Return value of the DFTI call
 * @param what   Short description of the call for the error message
 */
inline void mkl_dfti_check(MKL_LONG status, const char* what)
{
    if (status != 0 && !DftiErrorClass(status, DFTI_NO_ERROR))
        throw_with_line_number(std::string(what) + " failed: " + DftiErrorMessage(status));
}

#endif  // MKL_STATUS_CHECK_H_

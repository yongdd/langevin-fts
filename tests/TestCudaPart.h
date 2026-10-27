/**
 * @file TestCudaPart.h
 * @brief Fail a platform-loop test whose CUDA part did not run.
 *
 * PlatformSelector::avail_platforms() lists "cuda" only when a CUDA device is
 * visible. A test that loops over those platforms would, in a build with the
 * CUDA backend on a node without a GPU, run only its CPU parts and pass, which
 * reads as a passed CUDA test. Such a test runs its CPU parts and then ends
 * with `return cuda_part_missing() ? 1 : 0;`, like the TestCuda* tests and the
 * Python tests (tests/cuda_part.py).
 */

#ifndef TEST_CUDA_PART_H_
#define TEST_CUDA_PART_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "PlatformSelector.h"

/// True, after printing why, if this build has the CUDA backend but no CUDA device is visible
inline bool cuda_part_missing()
{
#ifdef USE_CUDA
    std::vector<std::string> platforms = PlatformSelector::avail_platforms();
    if (std::find(platforms.begin(), platforms.end(), "cuda") == platforms.end())
    {
        std::cout << "FAILED: this build has the CUDA backend, but no CUDA device is visible: "
                  << "the CUDA part of this test did not run." << std::endl;
        return true;
    }
#endif
    return false;
}

#endif

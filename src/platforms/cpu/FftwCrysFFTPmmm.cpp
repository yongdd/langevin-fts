/**
 * @file FftwCrysFFTPmmm.cpp
 * @brief FFTW implementation of crystallographic FFT for Pmmm symmetry.
 *
 * Uses native FFTW3 DCT-II/III (REDFT10/REDFT01) plans.
 */

#include "FftwCrysFFTPmmm.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
FftwCrysFFTPmmm::FftwCrysFFTPmmm(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part */)
    : Base(nx_logical, cell_para),
      plan_forward_(nullptr),
      plan_backward_(nullptr)
{
    // Allocate work buffers for plan creation
    io_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));
    temp_buffer_ = static_cast<double*>(fftw_malloc(sizeof(double) * M_physical_));

    if (!io_buffer_ || !temp_buffer_)
    {
        throw_with_line_number("Failed to allocate work buffer memory");
    }

    // DCT-II (REDFT10) forward, DCT-III (REDFT01) backward
    plan_forward_ = fftw_plan_r2r_3d(
        nx_physical_[0], nx_physical_[1], nx_physical_[2],
        io_buffer_, temp_buffer_,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10,
        FFTW_MEASURE);

    plan_backward_ = fftw_plan_r2r_3d(
        nx_physical_[0], nx_physical_[1], nx_physical_[2],
        io_buffer_, temp_buffer_,
        FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01,
        FFTW_MEASURE);

    if (!plan_forward_ || !plan_backward_)
    {
        throw_with_line_number("Failed to create FFTW DCT-II/III plans");
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
FftwCrysFFTPmmm::~FftwCrysFFTPmmm()
{
    if (plan_forward_) fftw_destroy_plan(plan_forward_);
    if (plan_backward_) fftw_destroy_plan(plan_backward_);
    if (io_buffer_) fftw_free(io_buffer_);
    if (temp_buffer_) fftw_free(temp_buffer_);
}

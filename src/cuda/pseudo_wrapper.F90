!--------------------------------------------------------------------
! this is a wrapper module for cuda version of the Pseudo-spectral method.
module pseudo

  use constants
  use simulation_box
  use polymer_chain
  implicit none

! fft_II = the number of grid of expfactor in the x direction
!  integer, protected :: fft_II
! expfactor = exponential factor in the Fourier space
! expfactor_half = half exponential factor for the 4th order (in s direction) pseudospectral.
  real(kind=rp), private, allocatable :: expfactor(:,:,:), expfactor_half(:,:,:)
! the number of grid in the each direction
  integer, private :: II, JJ, KK
! the number of additional grid to make neutral boundary
  integer, private :: MX, MY, MZ

contains
!-------------- pseudo_initialize ---------------
  subroutine pseudo_initialize(new_process_idx)
!   visible device id
    integer, optional, intent(in)  :: new_process_idx
    integer :: fft_II, process_idx
    integer :: num_block, threads_per_block

    if( PRESENT(new_process_idx)) then
        process_idx = new_process_idx
    else
        process_idx = 0
    end if

    II = x%hi-x%lo+1
    JJ = y%hi-y%lo+1
    KK = z%hi-z%lo+1

!   define matrix size
    allocate(expfactor     (0:II/2, 0:JJ-1,0:KK-1))
    allocate(expfactor_half(0:II/2, 0:JJ-1,0:KK-1))

!   calculate the Fourier components of distribution function
    call init_gaussian_factor(II,JJ,KK,expfactor,expfactor_half)

    if(.not. pp_get("gpu.num_block", num_block)) num_block = 512
    if(.not. pp_get("gpu.threads_per_block", threads_per_block)) threads_per_block = 256

    call pseudo_cuda_initialize(expfactor, expfactor_half, seg, volume, &
      II, JJ, KK, NN, NNf, num_block, threads_per_block, process_idx)

  end subroutine
!-------------- pseudo_run ---------------
! "subroutine pseudo_cuda_run" is implmented in "pseudo.cu" file.
  subroutine pseudo_run(phia, phib, QQ, &
    q1_init,q2_init,wa, wb)

    real(kind=rp), intent(out) :: phia   (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(out) :: phib   (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in)  :: wa     (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in)  :: wb     (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in)  :: q1_init(0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in)  :: q2_init(0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(out) :: QQ

    real(kind=rp) :: expdwa      (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp) :: expdwa_half (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp) :: expdwb      (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp) :: expdwb_half (0:II-1,0:JJ-1,0:KK-1)

    expdwa     (:,:,:) = exp(-wa(:,:,:)*ds*0.5d0)
    expdwb     (:,:,:) = exp(-wb(:,:,:)*ds*0.5d0)
    expdwa_half(:,:,:) = exp(-wa(:,:,:)*ds*0.25d0)
    expdwb_half(:,:,:) = exp(-wb(:,:,:)*ds*0.25d0)

    call pseudo_cuda_run(phia, phib, QQ, &
      q1_init,q2_init,expdwa,expdwa_half, &
      expdwb,expdwb_half)

  end subroutine
!-------------- pseudo_finalize ---------------
  subroutine pseudo_finalize

    deallocate(expfactor)
    deallocate(expfactor_half)

    call pseudo_cuda_finalize()

  end subroutine
end module pseudo
!--------------------------------------------------------------------

!--------------------------------------------------------------------
! this module defines parameters and subroutines to conduct fast
! Fourier transform (FFT) using math kernel library(MKL).
module fft
  use constants, only : rp
  use MKL_DFTI
  use MKL_TRIG_TRANSFORMS
  implicit none

! fft_normal_factor = nomalization factor FFT
  real(kind=rp), private :: fft_normal_factor

! the number of grid in the each direction
  integer, private :: II, JJ, KK

! pointers for forward and backward transform
  type(DFTI_DESCRIPTOR), private, pointer :: hand_forward, hand_backward

! if it has already initilized
  logical, private :: initilized = .False.

contains
!-------------- fft_initialize ---------------
  subroutine fft_initialize(box_dim, new_II, new_JJ, new_KK)
    integer, intent(in) :: box_dim
    integer, intent(in) :: new_II, new_JJ, new_KK

    ! Execution status
    integer :: status = 0
    ! Strides define data layout for real and complex domain
    integer, allocatable :: cstrides(:), rstrides(:)

    if(initilized) then
      return
    end if

    if(box_dim /=3) then
      write(*,*) "Only 3 dimensional FFT is implemented"
      stop (1)
    end if

    II = new_II
    JJ = new_JJ
    KK = new_KK

!   define types of FFT
    hand_forward => null()
    hand_backward => null()

    allocate(cstrides(4), rstrides(4))
    cstrides= [0, 1, II/2+1, (int(II/2.0d0)+1)*JJ]
    rstrides= [0, 1, II,     II*JJ]

#if USE_SINGLE_PRECISION == 1
    status = DftiCreateDescriptor(hand_forward, DFTI_SINGLE, DFTI_REAL, 3, [II,JJ,KK])
    status = DftiCreateDescriptor(hand_backward, DFTI_SINGLE, DFTI_REAL, 3, [II,JJ,KK])
#else
    status = DftiCreateDescriptor(hand_forward, DFTI_DOUBLE, DFTI_REAL, 3, [II,JJ,KK])
    status = DftiCreateDescriptor(hand_backward, DFTI_DOUBLE, DFTI_REAL, 3, [II,JJ,KK])
#endif

    status = DftiSetValue(hand_forward, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    status = DftiSetValue(hand_forward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
    status = DftiSetValue(hand_forward, DFTI_INPUT_STRIDES, rstrides)
    status = DftiSetValue(hand_forward, DFTI_OUTPUT_STRIDES, cstrides)
    status = DftiCommitDescriptor(hand_forward)

    status = DftiSetValue(hand_backward, DFTI_PLACEMENT, DFTI_NOT_INPLACE)
    status = DftiSetValue(hand_backward, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX)
    status = DftiSetValue(hand_backward, DFTI_INPUT_STRIDES, cstrides)
    status = DftiSetValue(hand_backward, DFTI_OUTPUT_STRIDES, rstrides)
    status = DftiCommitDescriptor(hand_backward)

    if(allocated(cstrides)) deallocate(cstrides)
    if(allocated(rstrides)) deallocate(rstrides)

!   compute a normalization factor
    fft_normal_factor = II*JJ*KK

    initilized = .True.

  end subroutine

!-------------- fft_r2c_3d ---------------
  subroutine fft_forward(rdata, cdata)
    ! compute 3D forward fourier transform
    real(kind=rp), intent(inout)    :: rdata(:,:,:)
    complex(kind=rp), intent(inout) :: cdata(:,:,:)
    integer :: status

    status = DftiComputeForward(hand_forward, rdata(:,1,1), cdata(:,1,1))

  end subroutine
!-------------- fft_c2r_3d ---------------
  subroutine fft_backward(cdata, rdata)
    ! compute 3D backward fourier transform
    real(kind=rp), intent(inout)    :: rdata(:,:,:)
    complex(kind=rp), intent(inout) :: cdata(:,:,:)
    integer :: status

    status = DftiComputeBackward(hand_backward, cdata(:,1,1), rdata(:,1,1))
    rdata(:,:,:) = rdata(:,:,:)/fft_normal_factor

  end subroutine
!-------------- fft_finalize ---------------
  subroutine fft_finalize()
    integer :: status

    if(.not. initilized) then
      return
    end if

    status = DftiFreeDescriptor(hand_forward)
    status = DftiFreeDescriptor(hand_backward)

    initilized = .False.
  end subroutine
end module fft
!--------------------------------------------------------------------

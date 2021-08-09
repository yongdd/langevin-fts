!--------------------------------------------------------------------
! this module defines parameters and subroutines related to pseudo-spectral method.
module pseudo

  use constants, only : rp
  use simulation_box
  use polymer_chain
  use fft
  implicit none

! expfactor = exponential factor in the Fourier space
! expfactor_half = half exponential factor for the 4th order (in s direction) pseudospectral.
  real(kind=rp), protected, allocatable :: expfactor(:,:,:), expfactor_half(:,:,:)

! the number of grid in the each direction
  integer, private :: II, JJ, KK

! the number of additional grid to make neutral boundary
  integer, private :: MX, MY, MZ

contains
!-------------- pseudo_initialize ---------------
  subroutine pseudo_initialize(new_process_idx)
!   visible device id
!   dummy variable for consistent with CUDA pseudo
    integer, optional, intent(in)  :: new_process_idx

    II = x%hi-x%lo+1
    JJ = y%hi-y%lo+1
    KK = z%hi-z%lo+1

!   initialize fft module
    call fft_initialize(BOX_DIM, &
      II, JJ, KK)

!   define matrix size
    allocate(expfactor     (0:II/2,0:JJ-1,0:KK-1))
    allocate(expfactor_half(0:II/2,0:JJ-1,0:KK-1))

!   calculate the Fourier components of distribution function
    call init_gaussian_factor(II,JJ,KK,expfactor,expfactor_half)

  end subroutine pseudo_initialize
!--------------------------------------------------------------------
! this subroutine integrates the block copolymer partition function
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
    real(kind=rp) :: q1          (0:II-1,0:JJ-1,0:KK-1,0:NN)
    real(kind=rp) :: q2          (0:II-1,0:JJ-1,0:KK-1,0:NN)
    integer :: n,i,j,k

    expdwa     (:,:,:) = exp(-wa(:,:,:)*ds*0.5d0)
    expdwb     (:,:,:) = exp(-wb(:,:,:)*ds*0.5d0)
    expdwa_half(:,:,:) = exp(-wa(:,:,:)*ds*0.25d0)
    expdwb_half(:,:,:) = exp(-wb(:,:,:)*ds*0.25d0)

    !$OMP PARALLEL SECTIONS private(n) num_threads(2)
    !$OMP SECTION
    q1(:,:,:,0) = 0.0d0
    q1(0:II-1,0:JJ-1,0:KK-1,0) = q1_init(0:II-1,0:JJ-1,0:KK-1)
! diffusion of A chain
    do n=1,NNf
      call pseudo_onestep(q1(:,:,:,n-1),q1(:,:,:,n),&
        expdwa,expdwa_half)
    end do
! diffusion of B chain
    do n=NNf+1,NN
      call pseudo_onestep(q1(:,:,:,n-1),q1(:,:,:,n),&
        expdwb,expdwb_half)
    end do
    !$OMP SECTION
    q2(:,:,:,NN) = 0.0d0
    q2(0:II-1,0:JJ-1,0:KK-1,NN) = q2_init(0:II-1,0:JJ-1,0:KK-1)
!   diffusion of B chain
    do n=NN,NNf+1,-1
      call pseudo_onestep(q2(:,:,:,n),q2(:,:,:,n-1),&
        expdwb,expdwb_half)
    end do
!   diffusion of A chain
    do n=NNf,1,-1
      call pseudo_onestep(q2(:,:,:,n),q2(:,:,:,n-1),&
        expdwa,expdwa_half)
    end do
    !$OMP END PARALLEL SECTIONS

!   compute segment concentration with Simpson quadratrue.
!   segment concentration. only half contribution from the end
    phia= q2(:,:,:,0)/2
    do n=1,NNf-1
      phia = phia + q1(:,:,:,n)*q2(:,:,:,n)
    end do
!   the junction is half A and half B
    phib = q1(:,:,:,NNf)*q2(:,:,:,NNf)/2
    phia = phia + phib
    do n=NNf+1,NN-1
      phib = phib + q1(:,:,:,n)*q2(:,:,:,n)
    end do
!   only half contribution from the end
    phib = phib + q1(:,:,:,NN)/2
!   calculates the total partition function
    QQ = sum(q1(:,:,:,NNf)*q2(:,:,:,NNf)*seg(:,:,:))

!   normalize the concentration
    phia = phia*volume/QQ/NN
    phib = phib*volume/QQ/NN

  end subroutine pseudo_run
!--------------------------------------------------------------------
! this subroutine integrates the block copolymer partition function, q,
! one time-step, ds, forward where expdw = exp(-w*ds/2), expdw_half =
! exp(-w*ds/4) and w is the field.
  subroutine pseudo_onestep(qin,qout,expdw,expdw_half)
    real(kind=rp), intent(in) ::          qin(0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(out)::         qout(0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in) ::        expdw(0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp), intent(in) ::   expdw_half(0:II-1,0:JJ-1,0:KK-1)

    real(kind=rp) :: qout1 (0:II-1,0:JJ-1,0:KK-1)
    real(kind=rp) :: qout2 (0:II-1,0:JJ-1,0:KK-1)

    complex(kind=rp) :: k_qin1 (0:II/2,0:JJ-1,0:KK-1)
    complex(kind=rp) :: k_qin2 (0:II/2,0:JJ-1,0:KK-1)

    !$OMP PARALLEL SECTIONS num_threads(2)
    !$OMP SECTION
!   step 1
    qout1(:,:,:) = expdw(:,:,:)*qin(:,:,:)

    call fft_forward(qout1,k_qin1)        ! 3D fourier discrete transform, forward and inplace
    k_qin1 = expfactor*k_qin1             ! multiply e^(-k^2 ds/6) in fourier space, in all 3 directions
    call fft_backward(k_qin1,qout1)       ! 3D fourier discrete transform, backword and inplace
    qout1(:,:,:) = expdw(:,:,:)*qout1(:,:,:) ! normalization calculation and evaluate e^(-w*ds/2) in real space
    !$OMP SECTION
!   step 2
    ! evaluate e^(-w*ds/4) in real space
    qout2(:,:,:) = expdw_half(:,:,:)*qin(:,:,:)
    call fft_forward(qout2,k_qin2)    ! 3D fourier discrete transform, forward and inplace
    k_qin2 = expfactor_half*k_qin2    ! multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
    call fft_backward(k_qin2,qout2)   ! 3D fourier discrete transform, backword and inplace
    qout2(:,:,:) = expdw(:,:,:)*qout2(:,:,:) ! normalization calculation and evaluate e^(-w*ds/2) in real space

    call fft_forward(qout2,k_qin2)    ! 3D fourier discrete transform, forward and inplace
    k_qin2 = expfactor_half*k_qin2    ! multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
    call fft_backward(k_qin2,qout2)   ! 3D fourier discrete transform, backword and inplace
    qout2(:,:,:) = expdw_half(:,:,:)*qout2(:,:,:)    ! normalization calculation and evaluate e^(-w*ds/4) in real space

    !$OMP END PARALLEL SECTIONS
    qout(:,:,:) = (4.0d0*qout2(:,:,:) - qout1(:,:,:))/3.0d0

  end subroutine pseudo_onestep

!-------------- pseudo_finalize ---------------
  subroutine pseudo_finalize

    deallocate(expfactor)
    deallocate(expfactor_half)
    call fft_finalize()

  end subroutine

end module pseudo
!--------------------------------------------------------------------

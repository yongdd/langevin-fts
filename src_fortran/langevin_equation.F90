
!--------------------------------------------------------------------
! this is a module for the Langevin equation of the exchange field W-
module langevin_equation
!
  use, intrinsic :: ieee_arithmetic, only : ieee_is_normal
  use param_parser
  use simulation_box
  use polymer_chain, only : NN
  use pseudo
  use random_gaussian

  implicit none

! delta_tau = langevin time step
! rho_a3 = segment density*a^3
! normal_sigma = sigma value of normal noise
! invarint_Nbar = invariant polymerization index
  real(kind=8), protected :: delta_tau, rho_a3, normal_sigma, invarint_Nbar

! input and output fields
  real(kind=8), private, allocatable :: wminus_out(:,:,:), wminus_star(:,:,:)

! Lambda and random noise function
  real(kind=8), private, allocatable :: lambda(:,:,:), lambda_star(:,:,:)
  real(kind=8), private, allocatable :: normal_noise(:,:,:)

contains
!-------------- le_initialize ---------------
  subroutine le_initialize()

!   initialize Gaussian random generator with seed
    call rng_initialize(777)

!   Langevin noise parameters
    if(.not. pp_get("langevin.delta_tau", delta_tau)) delta_tau = 0.20d0
    if(.not. pp_get("langevin.rho_a3",    rho_a3))    rho_a3 = 8.0d0
    invarint_Nbar = rho_a3**2 * NN
!   standard deviation of normal noise
    normal_sigma = sqrt(2*delta_tau/(dx*dy*dz*rho_a3*real(NN,8)**1.5d0))

!------------------ allocate arrays  -------------------------------
    allocate(wminus_out  (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(wminus_star (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(lambda      (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(lambda_star (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(normal_noise(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

  end subroutine

!---------------- le_update -----------------
  subroutine le_update(phia, phib, q1_init,q2_init, wplus, wminus, ext_w)

    real(kind=8), intent(inout) :: phia  (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(inout) :: phib  (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(in) :: q1_init  (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(in) :: q2_init  (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(inout) :: wplus (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(inout) :: wminus(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(in) ::  ext_w   (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)

    real(kind=8) :: QQ

!   call random number generator
    do
      call rng_gaussian(totalMM, 0.0d0, normal_sigma, normal_noise)
      if (ieee_is_normal(sum(normal_noise))) then
        exit
      else
        write(*,'(A)') "Infinity in normal_noise, retry..."
      end if
    end do

!   update wminus using Lanvegin equation
    call pseudo_run(phia, phib, QQ, q1_init,q2_init, wplus+wminus,wplus-wminus)
    lambda(:,:,:) = (phia(:,:,:) - phib(:,:,:)) + 2/chiN *(wminus(:,:,:)-ext_w(:,:,:))
    wminus(:,:,:) = wminus(:,:,:) - lambda(:,:,:)*delta_tau*NN + normal_noise(:,:,:)*NN
    call zeromean(wminus(:,:,:))

  end subroutine
!-------------- le_finalize ---------------
  subroutine le_finalize()

! finalize Gaussian random generator
    call rng_finalize()

  end subroutine
end module langevin_equation

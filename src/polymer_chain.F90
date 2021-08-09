!--------------------------------------------------------------------
! this module defines polyer chain parameters and provides
! subroutines that initialize bond interaction parameters
module polymer_chain

  use constants
  use param_parser
  use simulation_box
  implicit none

! NN = number of time steps
! NNf = number of time steps for A fraction
  integer, protected :: NN, NNf
! f = A fraction (1-f is the B fraction)
! ds = discrete step sizes
  real(kind=rp), protected :: f, ds
! chiN = interaction parameter between A and B Monomers
  real(kind=rp), protected :: chiN

contains
!----------------- chain_initialize -----------------------------
  subroutine chain_initialize()

    if(.not. pp_get("chain.a_fraction", f)) then
      write(*,*) "chain.a_fraction is not specified."
      stop (1)
    end if
    if(.not. pp_get("chain.contour_step", NN)) then
      write(*,*) "chain.contour_step is not specified."
      stop (1)
    end if
    if(.not. pp_get("chain.chiN", chiN)) then
      write(*,*) "chain.chiN is not specified."
      stop (1)
    end if

    ! grid number for A fraction
    NNf = nint(NN*f)
    if( abs(NNf-NN*f) > 1.d-6) then
      write(*,*) "NN*f is not an integer"
      stop (1)
    end if

    ! grid sizes contour direction
    ds = 1.0d0/NN
  end subroutine

!----------------- set_chiN ------------------------------
  subroutine chain_set_chiN(new_chiN)
    real(kind=rp), intent(in) :: new_chiN
    chiN = new_chiN
  end subroutine
!----------------- init_gaussian_factor ------------------------------
! Gaussian chain
  subroutine init_gaussian_factor(II,JJ,KK,expfactor,expfactor_half)
    integer, intent(in) :: II, JJ, KK
    real(kind=rp), intent(out) :: expfactor     (0:II/2,0:JJ-1,0:KK-1)
    real(kind=rp), intent(out) :: expfactor_half(0:II/2,0:JJ-1,0:KK-1)
    integer :: i, j, k, itemp, jtemp, ktemp
    real(kind=rp) :: xfactor, yfactor, zfactor

!   calculate the exponential factor
    xfactor = -(2*PI/(II*dx))**2*ds/6.0d0
    yfactor = -(2*PI/(JJ*dy))**2*ds/6.0d0
    zfactor = -(2*PI/(KK*dz))**2*ds/6.0d0

    do i=0,II/2
      itemp = i
      do j=0,JJ-1
        if( j > JJ/2) then
          jtemp = JJ-j
        else
          jtemp = j
        end if
        do k=0,KK-1
          if( k > KK/2) then
            ktemp = KK-k
          else
            ktemp = k
          end if
          expfactor(i,j,k) = exp(itemp**2*xfactor+jtemp**2*yfactor+ktemp**2*zfactor)
          expfactor_half(i,j,k) = exp((itemp**2*xfactor+jtemp**2*yfactor+ktemp**2*zfactor)/2)
        end do
      end do
    end do

  end subroutine

!----------------- chain_finalize ------------------------------------
  subroutine chain_finalize()
  end subroutine
end module

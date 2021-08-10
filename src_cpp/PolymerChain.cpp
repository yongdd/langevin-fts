!--------------------------------------------------------------------
! this module defines polyer chain parameters and provides
! subroutines that initialize bond interaction parameters
module polymer_chain

  implicit none

! NN = number of time steps
! NNf = number of time steps for A fraction
  integer, protected :: NN, NNf
! f = A fraction (1-f is the B fraction)
! ds = discrete step sizes
  real(kind=8), protected :: f, ds
! chiN = interaction parameter between A and B Monomers
  real(kind=8), protected :: chiN
  
  real(kind=8), parameter, private :: PI = 3.14159265358979323846d0

contains
!----------------- chain_initialize -----------------------------
  subroutine chain_initialize(new_f, new_NN, new_chiN)
    real(kind=8), intent(in) :: new_f, new_chiN
    integer, intent(in) :: new_NN

    f = new_f
    NN = new_NN
    chiN = new_chiN  

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
    real(kind=8), intent(in) :: new_chiN
    chiN = new_chiN
  end subroutine
!----------------- init_gaussian_factor ------------------------------
! Gaussian chain
  subroutine init_gaussian_factor(II,JJ,KK,dx,dy,dz,ds,expfactor,expfactor_half)
    integer, intent(in) :: II, JJ, KK
    real(kind=8), intent(in) :: dx, dy, dz, ds
    real(kind=8), intent(out) :: expfactor     (0:II/2,0:JJ-1,0:KK-1)
    real(kind=8), intent(out) :: expfactor_half(0:II/2,0:JJ-1,0:KK-1)
    integer :: i, j, k, itemp, jtemp, ktemp
    real(kind=8) :: xfactor, yfactor, zfactor

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

!--------------------------------------------------------------------
! this module defines simulation box parameters defined once and used repeatedly,
! and provides subroutines that compute inner product in a given geometry.
module simulation_box
  implicit none

! lo = lower bound of each grid
  integer, protected :: x_lo, y_lo, z_lo
! hi = upper bound of each grid
  integer, protected :: x_hi, y_hi, z_hi
! the number of total grid
  integer, protected :: totalMM
! Lx, Ly, Lz = length of the block copolymer in each direction (in units of aN^1/2)
! dx, dy, dz = discrete step sizes
  real(kind=8), protected :: Lx, Ly, Lz
  real(kind=8), protected :: dx, dy, dz

! seg = simple integral weight, dV
  real(kind=8), protected, allocatable :: seg(:,:,:)
! volume = volume of the system.
  real(kind=8), protected :: volume

contains
!----------------- box_initialize -----------------------------
  subroutine box_initialize(new_x_hi, new_y_hi, new_z_hi, new_Lx, new_Ly, new_Lz)
    integer, intent(in) :: new_x_hi, new_y_hi, new_z_hi 
    real(kind=8), intent(in) :: new_Lx, new_Ly, new_Lz
    integer :: i, j, k

    x_hi = new_x_hi - 1
    y_hi = new_y_hi - 1
    z_hi = new_z_hi - 1

    x_lo = 0
    y_lo = 0
    z_lo = 0

    Lx = new_Lx
    Ly = new_Ly
    Lz = new_lz

! the number of total grids
    totalMM = (x_hi-x_lo+1)*(y_hi-y_lo+1)*(z_hi-z_lo+1)

! grid sizes in x, z direction
    dx = Lx/(x_hi+1)
    dy = Ly/(y_hi+1)
    dz = Lz/(z_hi+1)

! define matrix size
    allocate(seg(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

! weight factor for integral
    seg(:,:,:) = dx*dy*dz
!  system polymer
    volume = Lx*Ly*Lz

  end subroutine
!-------------- box_finalize ---------------
  subroutine box_finalize
    deallocate(seg)
  end subroutine
!--------------------------------------------------------------------
! this function calculates integral g*h
  real(kind=8) function dot(g,h)
    real(kind=8), intent(in) :: g(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi), h(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    dot = sum(seg(:,:,:)*g(:,:,:)*h(:,:,:))
  end function dot
!--------------------------------------------------------------------
! this function calculates multiple integrals in
  real(kind=8) function multidot(n_comp,g,h)
    integer, intent(in) :: n_comp
    real(kind=8), intent(in) :: g(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi,1:n_comp)
    real(kind=8), intent(in) :: h(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi,1:n_comp)
    integer :: i
    multidot = 0.0d0
    do i = 1,n_comp
      multidot = multidot + sum(seg(:,:,:)*g(:,:,:,i)*h(:,:,:,i))
    end do
  end function multidot
!--------------------------------------------------------------------
! This subroutine makes the input a zero-meaned matrix
  subroutine zeromean(w)
    real(kind=8), intent(inout) :: w(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8) :: tot
    tot = sum(seg(:,:,:)*w(:,:,:))
    w(:,:,:) = w(:,:,:) - tot/volume
  end subroutine zeromean

end module simulation_box

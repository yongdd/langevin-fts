!--------------------------------------------------------------------
! this module defines simulation box parameters defined once and used repeatedly,
! and provides subroutines that compute inner product in a given geometry.
module simulation_box

  use param_parser
  implicit none

! BOX_DIM = dimension of simulation box
! if BOX_DIM == 1, input parameters related to y and z directions will be ignored.
! if BOX_DIM == 2, input parameters related to       z direction  will be ignored.
#if BOX_DIMENSION == 1
  integer, parameter :: BOX_DIM = 1
#elif BOX_DIMENSION == 2
  integer, parameter :: BOX_DIM = 2
#else
  integer, parameter :: BOX_DIM = 3
#endif

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
  subroutine box_initialize()
    integer :: i, j, k

    if(.not. pp_get("geometry.grids", x_hi, 1) .and. BOX_DIM >= 1) then
      write(*,*) "geometry.grids[1] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", y_hi, 2) .and. BOX_DIM >= 2) then
      write(*,*) "geometry.grids[2] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", z_hi, 3) .and. BOX_DIM >= 3) then
      write(*,*) "geometry.grids[3] is not specified."
      stop (1)
    end if

    if(.not. pp_get("geometry.box_size", Lx, 1) .and. BOX_DIM >= 1) then
      write(*,*) "geometry.box_size[1] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.box_size", Ly, 2) .and. BOX_DIM >= 2) then
      write(*,*) "geometry.box_size[2] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.box_size", Lz, 3) .and. BOX_DIM >= 3) then
      write(*,*) "geometry.box_size[3] is not specified."
      stop (1)
    end if

    x_lo = 1
    y_lo = 1
    z_lo = 1
    if(BOX_DIM <= 1) then
      Ly = 1.0d0
      y_hi = 1
      y_lo = 1
    end if
    if(BOX_DIM <= 2) then
      Lz = 1.0d0
      z_hi = 1
      z_lo = 1
    end if

! the number of total grids
    totalMM = (x_hi-x_lo+1)*(y_hi-y_lo+1)*(z_hi-z_lo+1)

! grid sizes in x, z direction
    dx = Lx/x_hi
    dy = Ly/y_hi
    dz = Lz/z_hi

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

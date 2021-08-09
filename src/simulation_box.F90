!--------------------------------------------------------------------
! this module defines simulation box parameters defined once and used repeatedly,
! and provides subroutines that compute inner product in a given geometry.
module simulation_box

  use constants
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

! grid parameters
  type :: geometry
!   lo = lower bound of each grid
!   hi = upper bound of each grid
    integer :: lo, hi
  end type

  type(geometry), protected :: x, y, z
! the number of total grid
  integer, protected :: totalMM
! Lx, Ly, Lz = length of the block copolymer in each direction (in units of aN^1/2)
! dx, dy, dz = discrete step sizes
  real(kind=rp), protected :: Lx, Ly, Lz
  real(kind=rp), protected :: dx, dy, dz

! seg = simple integral weight, dV
  real(kind=rp), protected, allocatable :: seg(:,:,:)
! volume = volume of the system.
  real(kind=rp), protected :: volume

contains
!----------------- box_initialize -----------------------------
  subroutine box_initialize()
    integer :: i, j, k

    if(.not. pp_get("geometry.grids", x%hi, 1) .and. BOX_DIM >= 1) then
      write(*,*) "geometry.grids[1] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", y%hi, 2) .and. BOX_DIM >= 2) then
      write(*,*) "geometry.grids[2] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", z%hi, 3) .and. BOX_DIM >= 3) then
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

    x%lo = 1
    y%lo = 1
    z%lo = 1
    if(BOX_DIM <= 1) then
      Ly = 1.0d0
      y%hi = 1
      y%lo = 1
    end if
    if(BOX_DIM <= 2) then
      Lz = 1.0d0
      z%hi = 1
      z%lo = 1
    end if

! the number of total grids
    totalMM = (x%hi-x%lo+1)*(y%hi-y%lo+1)*(z%hi-z%lo+1)

! grid sizes in x, z direction
    dx = Lx/x%hi
    dy = Ly/y%hi
    dz = Lz/z%hi

! define matrix size
    allocate(seg(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))

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
  real(kind=rp) function dot(g,h)
    real(kind=rp), intent(in) :: g(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi), h(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
    dot = sum(seg(:,:,:)*g(:,:,:)*h(:,:,:))
  end function dot
!--------------------------------------------------------------------
! this function calculates multiple integrals in
  real(kind=rp) function multidot(n_comp,g,h)
    integer, intent(in) :: n_comp
    real(kind=rp), intent(in) :: g(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi,1:n_comp)
    real(kind=rp), intent(in) :: h(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi,1:n_comp)
    integer :: i
    multidot = 0.0d0
    do i = 1,n_comp
      multidot = multidot + sum(seg(:,:,:)*g(:,:,:,i)*h(:,:,:,i))
    end do
  end function multidot
!--------------------------------------------------------------------
! This subroutine makes the input a zero-meaned matrix
  subroutine zeromean(w)
    real(kind=rp), intent(inout) :: w(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
    real(kind=rp) :: tot
    tot = sum(seg(:,:,:)*w(:,:,:))
    w(:,:,:) = w(:,:,:) - tot/volume
  end subroutine zeromean

end module simulation_box

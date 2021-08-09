!------------------------------------------------------------------
! A circular buffer is a data structure that uses a single,
! fixed-size buffer as if it were connected end-to-end.
! Each elements are 1-dmensional real array.
!------------------------------------------------------------------
module circular_buffer_module

  use constants, only : rp
  implicit none

  ! This is a pointer to an real array.
  ! It is the only way to make array of pointer.
  ! Ponter is introduced to avoid array copy.
  type real_array_pt
    real(kind=rp), pointer, dimension(:) :: p
  end type real_array_pt

  type circular_buffer
    integer :: length  ! maximum number of elements
    integer :: width   ! size of each elements
    integer :: start   ! index of oldest elements
    integer :: n_items  ! the number of stored items
    type(real_array_pt), dimension(:), allocatable :: elems
  end type circular_buffer

contains
  subroutine cbInit(this, length, width)
    type(circular_buffer), intent(inout) :: this
    integer, intent(in) :: length, width
    integer :: i
    this%length = length
    this%width = width
    this%start = 0
    this%n_items = 0
    allocate( this%elems(0:length-1) )
    do i = 0, length-1
      allocate(this%elems(i)%p(0:width-1))
    end do
    
  end subroutine

  subroutine cbReset(this)
    type(circular_buffer), intent(inout) :: this
    this%start = 0
    this%n_items = 0
  end subroutine

  subroutine cbFree(this)
    type(circular_buffer), intent(inout) :: this
    integer :: i
    do i = 0, this%length-1
      deallocate(this%elems(i)%p)
    end do
    deallocate( this%elems )
  end subroutine

  subroutine cbInsert(this, elem)
    type(circular_buffer), intent(inout) :: this
    real(kind=rp), intent(in) :: elem(0:this%width-1)
    this%elems(mod(this%start + this%n_items, this%length))%p = elem
    if (this%n_items == this%length) then
      this%start = mod(this%start+1, this%length)
    end if
    this%n_items  = min(this%n_items+1, this%length)
  end subroutine

  function cbGetArray(this, n)
    type(circular_buffer), intent(in) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(0:this%width-1) :: cbGetArray
    cbGetArray = this%elems(mod(this%start+n,this%length))%p
  end function

  function cbGetArrayPt(this, n)
    type(circular_buffer), intent(in) :: this
    integer, intent(in) :: n
    type(real_array_pt) :: cbGetArrayPt
    cbGetArrayPt = this%elems(mod(this%start+n,this%length))
  end function

  real(kind=rp) function cbGet(this, n, m)
    type(circular_buffer), intent(in) :: this
    integer, intent(in) :: n, m
    cbGet = this%elems(mod(this%start+n,this%length))%p(m)
  end function

  real(kind=rp) function cbGetSym(this, n, m)
    type(circular_buffer), intent(in) :: this
    integer, intent(in) :: n, m
    cbGetSym = this%elems(mod(this%start+max(n,m),this%length))%p(abs(n-m))
  end function

end module circular_buffer_module

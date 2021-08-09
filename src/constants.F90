!--------------------------------------------------------------------
! this module defines some constant variables
module constants
  implicit none

  integer, parameter :: sp = selected_real_kind(6)
  integer, parameter :: dp = selected_real_kind(15)

! precision of real
#if USE_SINGLE_PRECISION == 1
  integer, parameter :: rp = sp
#else
  integer, parameter :: rp = dp
#endif

! PI = 3.141592..
  real(kind=dp), parameter :: PI = 4.0d0 * atan(1.0d0)

end module constants



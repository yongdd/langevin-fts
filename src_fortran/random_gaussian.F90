!--------------------------------------------------------------------
!this module provides Gaussian random generators
module random_gaussian
  implicit none
  
  ! function overloading
  interface rng_initialize
    module procedure rng_initialize_with_input
    module procedure rng_initialize_without_input
  end interface
  
  real(kind=8), parameter, private :: PI = 3.14159265358979323846d0
  
  public :: rng_set_seed, rng_gaussian
  public :: rng_can_get_seed, rng_get_seed, rng_finalize
  private :: rng_initialize_with_input, rng_initialize_without_input
  
contains
!----------------- rng_initialize -----------------------------
  subroutine rng_initialize_with_input(input_seed)
    integer :: n, input_seed
    integer, allocatable :: seed(:)
    call random_seed(size = n)
    allocate(seed(n))
    seed = input_seed
    call random_seed(put=seed)
    if (allocated(seed)) deallocate(seed)
  end subroutine
  
  subroutine rng_initialize_without_input()
    call random_seed()
  end subroutine
!----------------- rng_set_seed -----------------------------
  subroutine rng_set_seed(seed_array, n)
    integer, intent(in) :: n
    integer, intent(inout) :: seed_array(n)
    call random_seed(put=seed_array)
  end subroutine
!----------------- rng_gaussian -----------------------------
  subroutine rng_gaussian(totalMM, mean, std, normal_noise)

    ! The Box-Muller transform is implemented.
    integer, intent(in) :: totalMM
    real(kind=8), intent(in) :: mean, std
    real(kind=8), dimension(totalMM) :: normal_noise
    ! uniform random array
    real(kind=8), allocatable :: u1(:) ! this array is declared as double precision to reduce singularity.
    real(kind=8), allocatable :: u2(:)
    integer :: uniformMM
    uniformMM = (totalMM + 1)/2
    
    allocate(u1(1:uniformMM))
    allocate(u2(1:uniformMM))
    call random_number(u1)
    call random_number(u2)
      
    normal_noise(1:uniformMM) = mean + sqrt(-2*log(u1))*cos(2*PI*u2)*std
    normal_noise(totalMM-uniformMM+1:totalMM) = mean + sqrt(-2*log(u1))*sin(2*PI*u2)*std
    ! If totalMM is odd, normal_noise(uniformMM) will be overwritten.
    
    deallocate(u1, u2)
    
  end subroutine
!----------------- rng_can_get_seed -----------------------------
  logical function rng_can_get_seed()
    rng_can_get_seed = .True.
  end function
!----------------- rng_get_seed -----------------------------
  function rng_get_seed()
    integer :: n
    integer, allocatable :: seed(:)
    integer, allocatable :: rng_get_seed(:)
    call random_seed(size = n)
    allocate(seed(n))
    call random_seed(get=seed)
    rng_get_seed = seed
    if (allocated(seed)) deallocate(seed)
  end function
!----------------- rng_finalize -----------------------------
  subroutine rng_finalize()
  end subroutine
  
end module random_gaussian

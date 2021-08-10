!-----------------------------------------------------------
! The main program begins here
program langevin_fts_block_copolymer_3d

  use, intrinsic :: ieee_arithmetic, only : ieee_is_normal
  use random_gaussian
  !use omp_lib

  implicit none
  real(kind=8), parameter :: PI = 3.14159265358979323846d0
  real(kind=8) :: mean, sigma
  real(kind=8) :: mean_out, sigma_out
  real(kind=8), allocatable :: normal_noise(:), noise_one(:)
  integer,      allocatable :: histo_count(:)
  real(kind=8), allocatable :: histo(:), dist(:)
  integer :: i, k, totalMM, histMM
  real(kind=8) :: x, bin_width, error

!--------------- Test 1 : single grid-----------------

! initialize
  mean = 5.0d0
  sigma = 13.d0
  totalMM = 20460

  bin_width = 0.2d0
  histMM = 200

  allocate(normal_noise(totalMM))
  allocate(noise_one(1))
  allocate(histo_count(-histMM:histMM))
  allocate(histo(-histMM:histMM))
  allocate(dist(-histMM:histMM))

  call rng_initialize()

! run
  do i = 1, totalMM
    do
      call rng_gaussian(1, mean, sigma, noise_one)
      if (ieee_is_normal(sum(normal_noise))) then
        exit
      else
        write(*,'(A)') "Infinity in normal_noise, retry..."
      end if
    end do
    normal_noise(i) = noise_one(1)
  end do

! check
  mean_out = sum(normal_noise)/real(totalMM,8)
  sigma_out = sqrt(sum((normal_noise-mean_out)**2)/real(totalMM,8))

  histo_count = 0
  do i = 1, totalMM
    k = nint(normal_noise(i)/bin_width)
    if ( -histMM <= k .and. k <= histMM ) then
      histo_count(k) = histo_count(k) + 1
    end if
  end do

  histo(:) = real(histo_count(:),8)/real(totalMM,8)/bin_width
  do i = -histMM, histMM
    x= i*bin_width
    dist(i) = exp(-(x-mean)**2/(2*sigma**2))/sigma/sqrt(2*PI)
  end do
  error = sqrt(sum((histo-dist)**2) * bin_width)

! write output
  write(*,'(A,I8)') "Test 1, single grid : ", totalMM
  write(*,'(A,2F15.7)') "mean :", mean, mean_out
  write(*,'(A,2F15.7)') "sigma :", sigma, sigma_out
  write(*,'(A,F15.7)') "error : ", error
  open(unit=1, file="random_1.txt")
  do i = -histMM, histMM
    x= i*bin_width
    write(1,'(I8,3F15.7)') i, x, histo(i), dist(i)
  end do

! finalize
  call rng_finalize()
  deallocate(normal_noise, noise_one)
  deallocate(histo_count)
  deallocate(histo)
  deallocate(dist)

!--------------- Test 2 : Array of even grid-----------------

! initialize
  mean = 5.0d0
  sigma = 13.d0
  totalMM = 20460000

  bin_width = 0.2d0
  histMM = 200

  allocate(normal_noise(totalMM))
  allocate(histo_count(-histMM:histMM))
  allocate(histo(-histMM:histMM))
  allocate(dist(-histMM:histMM))
  call rng_initialize()

! run
  do
    call rng_gaussian(totalMM, mean, sigma, normal_noise)
    if (ieee_is_normal(sum(normal_noise))) then
      exit
    else
      write(*,'(A)') "Infinity in normal_noise, retry..."
    end if
  end do

! check
  mean_out = sum(normal_noise)/real(totalMM,8)
  sigma_out = sqrt(sum((normal_noise-mean_out)**2)/real(totalMM,8))

  histo_count = 0
  do i = 1, totalMM
    k = nint(normal_noise(i)/bin_width)
    if ( -histMM <= k .and. k <= histMM ) then
      histo_count(k) = histo_count(k) + 1
    end if
  end do

  histo(:) = real(histo_count(:),8)/real(totalMM,8)/bin_width
  do i = -histMM, histMM
    x= i*bin_width
    dist(i) = exp(-(x-mean)**2/(2*sigma**2))/sigma/sqrt(2*PI)
  end do
  error = sqrt(sum((histo-dist)**2) * bin_width)

! write output
  write(*,'(A,I8)') "Test 2, Array of even grid : ", totalMM
  write(*,'(A,2F15.7)') "mean :", mean, mean_out
  write(*,'(A,2F15.7)') "sigma :", sigma, sigma_out
  write(*,'(A,F15.7)') "error : ", error
  open(unit=1, file="random_2.txt")
  do i = -histMM, histMM
    x= i*bin_width
    write(1,'(I8,3F15.7)') i, x, histo(i), dist(i)
  end do

! finalize
  call rng_finalize()
  deallocate(normal_noise)
  deallocate(histo_count)
  deallocate(histo)
  deallocate(dist)

!--------------- Test 3 : Array of odd grid-----------------

! initialize
  mean = 5.0d0
  sigma = 13.d0
  totalMM = 20460001

  bin_width = 0.2d0
  histMM = 200

  allocate(normal_noise(totalMM))
  allocate(histo_count(-histMM:histMM))
  allocate(histo(-histMM:histMM))
  allocate(dist(-histMM:histMM))
  call rng_initialize()

! run
  do
    call rng_gaussian(totalMM, mean, sigma, normal_noise)
    if (ieee_is_normal(sum(normal_noise))) then
      exit
    else
      write(*,'(A)') "Infinity in normal_noise, retry..."
    end if
  end do

! check
  mean_out = sum(normal_noise)/real(totalMM,8)
  sigma_out = sqrt(sum((normal_noise-mean_out)**2)/real(totalMM,8))

  histo_count = 0
  do i = 1, totalMM
    k = nint(normal_noise(i)/bin_width)
    if ( -histMM <= k .and. k <= histMM ) then
      histo_count(k) = histo_count(k) + 1
    end if
  end do

  histo(:) = real(histo_count(:),8)/real(totalMM,8)/bin_width
  do i = -histMM, histMM
    x= i*bin_width
    dist(i) = exp(-(x-mean)**2/(2*sigma**2))/sigma/sqrt(2*PI)

  end do
  error = sqrt(sum((histo-dist)**2) * bin_width)

! write output
  write(*,'(A,I8)') "Test 3, Array of odd grid : ", totalMM
  write(*,'(A,2F15.7)') "mean :", mean, mean_out
  write(*,'(A,2F15.7)') "sigma :", sigma, sigma_out
  write(*,'(A,F15.7)') "error : ", error
  open(unit=1, file="random_3.txt")
  do i = -histMM, histMM
    x= i*bin_width
    write(1,'(I8,3F15.7)') i, x, histo(i), dist(i)
  end do

! finalize
  call rng_finalize()
  deallocate(normal_noise)
  deallocate(histo_count)
  deallocate(histo)
  deallocate(dist)

end program

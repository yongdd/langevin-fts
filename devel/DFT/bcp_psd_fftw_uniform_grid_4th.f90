! Compile Command
! ifx bcp_psd_fftw_uniform_grid_4th.f90 -o bcp_psd_fftw_uniform_grid_4th -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -lfftw3
!
!--------------------------------------------------------------------
!this module defines simulation parameters defined once and used repeatedly.
!only static parameters are included in this module
module parameters
  implicit none
! chiN = interaction parameter between A and B Monomers,  pi = 3.141592..
  real(kind=8), parameter :: PI = 4.0d0 * atan(1.0d0)
! BOX_DIMENSION = dimension of simulation box
! if BOX_DIMENSION == 1, input parameters related to x and y directions will be ignored.
! if BOX_DIMENSION == 2, input parameters related to       y direction  will be ignored.
  integer, parameter :: BOX_DIMENSION = 1
! grid parameters
  type geomtry
!   lo = lower bound of each grid
!   hi = upper bound of each grid
    integer :: lo, hi
!   boundary conditions in each direction
!   if is_periodic == True -> periodic boundary
!   else is_periodc == False -> reflecting boundary
    logical :: is_periodic
  end type
  type(geomtry) :: x, y, z
! Lx, Ly, Lz = length of the block copolymer in each direction (in units of aN^1/2)
! dx, dy, dz, ds = discrete step sizes
! f = A fraction (1-f is the B fraction)
  real(kind=8) :: Lx, Ly, Lz, f, chiN
  real(kind=8), protected :: dx, dy, dz, ds
! NN = number of time steps
! NNf = number of time steps for A fraction
  integer :: NN
  integer, protected :: NNf
! seg = simple integral weight
  real(kind=8), protected, allocatable :: segx(:),segy(:),segz(:), seg(:,:,:)
! volume = volume of the system.
  real(kind=8), protected  :: volume

contains
!----------------- initialize -----------------------------
! this subroutine initializes repeatedly used parameters
  subroutine initialize()
    implicit none
    integer :: i, j, k
!   grid number for A fraction
    NNf = NN*f
    if( abs(NNf-NN*f) > 1.d-6) then
      write(*,*) "NN*f is not an integer"
    end if

    x%lo = 1
    y%lo = 1
    z%lo = 1

    if(BOX_DIMENSION == 1) then
      Ly = 1.0d0
      Lz = 1.0d0
      y%hi = 1
      y%lo = 1
      z%hi = 1
      z%lo = 1
    elseif(BOX_DIMENSION == 2) then
      Lz = 1.0d0
      z%hi = 1
      z%lo = 1
    end if

!   grid sizes in x, z and time direction
    dx = Lx/x%hi
    dy = Ly/y%hi
    dz = Lz/z%hi
    ds = 1.0d0/NN
    volume = Lx*Ly*Lz

!   define matrix size
    allocate(seg(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi),segx(x%lo:x%hi),segy(y%lo:y%hi),segz(z%lo:z%hi))

!   weight factor for integral, only half contribution at the boundary
    segx(:) = 1.0d0 * dx
    segy(:) = 1.0d0 * dy
    segz(:) = 1.0d0 * dz

    ! if ( .not. x%is_periodic) then
    !   segx(x%lo) = 0.5d0 * dx
    !   segx(x%hi) = 0.5d0 * dx
    ! end if
    ! if ( (.not. y%is_periodic) .and. BOX_DIMENSION >= 2) then
    !   segy(y%lo) = 0.5d0 * dy
    !   segy(y%hi) = 0.5d0 * dy
    ! end if
    ! if ( (.not. z%is_periodic) .and. BOX_DIMENSION >= 3) then
    !   segz(z%lo) = 0.5d0 * dz
    !   segz(z%hi) = 0.5d0 * dz
    ! end if

    do i=x%lo,x%hi
      do j=y%lo,y%hi
        do k=z%lo,z%hi
          seg(i,j,k) = segx(i)*segy(j)*segz(k)
        end do
      end do
    end do
    write(*,*) "volume, sum(seg)", volume, sum(seg)

  end subroutine
end module parameters
!--------------------------------------------------------------------
!this module defines parameter related to Fast Fourier Transform (FFT)
!only static parameters are included in this module
module fft
  use, intrinsic :: iso_c_binding
  use parameters
  implicit none
  include 'fftw3.f03'
! pointers for forward and backward transform
  type(C_PTR), protected :: plan_forward, plan_backward
! dividedby = factor for the nomalization
  real(kind=8), protected :: dividedby
! expfactor = exponential factor in the Fourier space
! expfactor_half = half exponential factor for the 4th order (in s direction) pseudospectral.
  real(kind=8), protected, allocatable :: expfactor(:,:,:), expfactor_half(:,:,:)

contains
!-------------- fft_initialize ---------------
  subroutine fft_initialize
    integer :: i, j, k, itemp, jtemp, ktemp
    real(kind=8) :: xfactor, yfactor, zfactor
!   types of FFT in  forward direction
    integer :: ttf_x, ttf_y, ttf_z
!   types of FFT in backward direction
    integer :: ttb_x, ttb_y, ttb_z
!   define matrix size
    allocate(expfactor     (0:x%hi-x%lo,0:y%hi-y%lo,0:z%hi-z%lo))
    allocate(expfactor_half(0:x%hi-x%lo,0:y%hi-y%lo,0:z%hi-z%lo))

!   calculate the exponential factor
    xfactor = -(pi/Lx)**2*ds/6.0d0
    yfactor = -(pi/Ly)**2*ds/6.0d0
    zfactor = -(pi/Lz)**2*ds/6.0d0
    if ( x%is_periodic) xfactor = -(2*pi/Lx)**2*ds/6.0d0
    if ( y%is_periodic) yfactor = -(2*pi/Ly)**2*ds/6.0d0
    if ( z%is_periodic) zfactor = -(2*pi/Lz)**2*ds/6.0d0

    do i=0,x%hi-x%lo
      if(x%is_periodic .and. i > (x%hi-x%lo+1)/2) then
        itemp = (x%hi-x%lo+1)-i
      else
        itemp = i
      end if
      do j=0,y%hi-y%lo
        if(y%is_periodic .and. j > (y%hi-y%lo+1)/2) then
          jtemp = (y%hi-y%lo+1)-j
        else
          jtemp = j
        end if
        do k=0,z%hi-z%lo
          if(z%is_periodic .and. k > (z%hi-z%lo+1)/2) then
            ktemp = (z%hi-z%lo+1)-k
          else
            ktemp = k
          end if
          expfactor(i,j,k) = exp(itemp**2*xfactor+jtemp**2*yfactor+ktemp**2*zfactor)
          expfactor_half(i,j,k) = exp((itemp**2*xfactor+jtemp**2*yfactor+ktemp**2*zfactor)/2)
        end do
      end do
    end do

    ! write(*,*) "ds/6.0d0", ds/6.0d0
    ! write(*,*) "expfactor"
    ! write(*,*) expfactor

!   define types of FFT
    ttf_x = FFTW_REDFT10
    ttf_y = FFTW_REDFT10
    ttf_z = FFTW_REDFT10
    if ( x%is_periodic) ttf_x = FFTW_R2HC
    if ( y%is_periodic) ttf_y = FFTW_R2HC
    if ( z%is_periodic) ttf_z = FFTW_R2HC

    ttb_x = FFTW_REDFT01
    ttb_y = FFTW_REDFT01
    ttb_z = FFTW_REDFT01
    if ( x%is_periodic) ttb_x = FFTW_HC2R
    if ( y%is_periodic) ttb_y = FFTW_HC2R
    if ( z%is_periodic) ttb_z = FFTW_HC2R

!   create plans for FFT
    if( BOX_DIMENSION == 3) then
      plan_forward =  fftw_plan_r2r_3d(z%hi-z%lo+1,y%hi-y%lo+1,x%hi-x%lo+1,expfactor,expfactor,ttf_z,ttf_y,ttf_x,FFTW_ESTIMATE)
    elseif( BOX_DIMENSION == 2) then
      plan_forward =  fftw_plan_r2r_2d(            y%hi-y%lo+1,x%hi-x%lo+1,expfactor,expfactor,      ttf_y,ttf_x,FFTW_ESTIMATE)
    elseif( BOX_DIMENSION == 1) then
      plan_forward =  fftw_plan_r2r_1d(                        x%hi-x%lo+1,expfactor,expfactor,            ttf_x,FFTW_ESTIMATE)
    end if

    if( BOX_DIMENSION == 3) then
      plan_backward = fftw_plan_r2r_3d(z%hi-z%lo+1,y%hi-y%lo+1,x%hi-x%lo+1,expfactor,expfactor,ttb_z,ttb_y,ttb_x,FFTW_ESTIMATE)
    elseif( BOX_DIMENSION == 2) then
      plan_backward = fftw_plan_r2r_2d(            y%hi-y%lo+1,x%hi-x%lo+1,expfactor,expfactor,      ttb_y,ttb_x,FFTW_ESTIMATE)
    elseif( BOX_DIMENSION == 1) then
      plan_backward = fftw_plan_r2r_1d(                        x%hi-x%lo+1,expfactor,expfactor,            ttb_x,FFTW_ESTIMATE)
    end if

!   compute a normalization factor
    if ( x%is_periodic) then
      dividedby = x%hi-x%lo+1
    else
      dividedby = (x%hi-x%lo+1)*2
    end if

    if ( y%is_periodic) then
      dividedby = dividedby*(y%hi-y%lo+1)
    elseif (BOX_DIMENSION >= 2) then
      dividedby = dividedby*(y%hi-y%lo+1)*2
    end if

    if ( z%is_periodic) then
      dividedby = dividedby*(z%hi-z%lo+1)
    elseif (BOX_DIMENSION >= 3) then
      dividedby = dividedby*(z%hi-z%lo+1)*2
    end if
  end subroutine
!-------------- fft_finalize ---------------
  subroutine fft_finalize
    call fftw_destroy_plan(plan_forward)
    call fftw_destroy_plan(plan_backward)
  end subroutine

end module fft
!--------------------------------------------------------------------
! The main program begins here
program block_copolymer

  use parameters
  use fft
  implicit none
! iter = number of iteration steps, maxiter = maximum number of iteration steps
  integer :: i, j, k, iter, maxiter
! anderson mixing related parameters
  integer :: nit, nit2, n_anderson, start_anderson, max_anderson, m
! QQ = total partition function
  real(kind=8) :: QQ, energy_chain, energy_field, energy_tot, energy_old
! error_level = variable to check convergence of the iteration
! dot = function performing dot product
! multidot = function performing multiple dot product
! mix = dynamically changing mixing parameter
! min_mix = its minimum value
  real(kind=8) :: error_level, old_error_level, dot, multidot
  real(kind=8) :: mix, min_mix
! input and output fields, xi is temporary storage for pressures
  real(kind=8), allocatable :: w(:,:,:,:),wout(:,:,:,:), xi(:,:,:)
! a few previous field values are stored for anderson mixing
  real(kind=8), allocatable :: wout_hist(:,:,:,:,:),wdiff_hist(:,:,:,:,:)
! arrays to calculate anderson mixing
  real(kind=8), allocatable :: u_nm(:,:), v_n(:), a_n(:), wdiffdots(:,:)
! segment concentration
  real(kind=8), allocatable :: phia(:,:,:), phib(:,:,:), phitot(:,:,:)
! strings to name output files
  character (len=1024) filename

! set up the parameters of the system we solve
! block copolymer sepcification
  chiN = 20.0d0
  f = 0.3d0

!  x%is_periodic = .True.
!  y%is_periodic = .True.
!  z%is_periodic = .True.

  x%is_periodic = .False.
  y%is_periodic = .False.
  z%is_periodic = .False.

  x%hi = 12
  y%hi = 24
  z%hi = 36

  Lx = 4.0d0
  Ly = 3.0d0
  Lz = 2.0d0

  NN = 50
!
! mixing parameter
  mix = 0.1d0
! minimum mixing parameter
  min_mix = 0.1d0
! number of iterations before Anderson mixing begins.
! currently, assign a large value. it will be determined later.
  start_anderson = 1000000
! max number of previous steps to calculate new field when using Anderson mixing
  max_anderson = 30
  m = max_anderson + 1
! number of anderson mixing steps, increases from 1 to max_anderson
  n_anderson = 0
! iteration must stop before iter = maxiter happens
  maxiter = 10

! initialize parameters
  call initialize()
! initialize fft parameters
  call fft_initialize()

  write(*,*) "BOX_DIMENSION", BOX_DIMENSION
  write(*,*) "x%lo,x%hi,y%lo,y%hi,z%lo,z%hi"
  write(*,*)  x%lo,x%hi,y%lo,y%hi,z%lo,z%hi
  write(*,*) "Lx,Ly,Lz"
  write(*,*)  Lx,Ly,Lz
  write(*,*) "dx,dy,dz"
  write(*,*)  dx,dy,dz

! define arrays for anderson mixing
  allocate( wout_hist(0:max_anderson,1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(wdiff_hist(0:max_anderson,1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(u_nm(1:max_anderson,1:max_anderson))
  allocate(v_n(1:max_anderson),a_n(1:max_anderson))
  allocate(wdiffdots(0:max_anderson,0:max_anderson))
!
! define arrays for field and density
  allocate(   w(1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(wout(1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(    phia(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(    phib(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
  allocate(  phitot(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))

!  do i=x%lo,x%hi
!    do j=y%lo,y%hi
!      do k=z%lo,z%hi
!        if(sqrt(min((j*dy)**2+(k*dz)**2,((y%hi/2-j)*dy)**2+((z%hi/2-k)*dz)**2,&
!          (j*dy)**2+((z%hi-k)*dz)**2,((y%hi-j)*dy)**2+(k*dz)**2,&
!          ((y%hi-j)*dy)**2+((z%hi-k)*dz)**2)) < Ly/8.0d0) then
!          phia(i,j,k) = 1.0d0
!        else
!          phia(i,j,k)= 0.0d0
!        end if
!      end do
!    end do
!  end do

  call random_number(phia)
!  phia = reshape( phia, (/ x%hi-x%lo+1,y%hi-y%lo+1,z%hi-z%lo+1 /), order = (/ 3, 2, 1 /))
!  call random_number(phia(:,:,z%lo))
!  do k=z%lo,z%hi
!    phia(:,:,k) = phia(:,:,z%lo)
!  end do

!  do i=x%lo,x%hi
!     phia(i,:,:) = cos(2.0*PI*(i-x%lo)/(x%hi-x%lo))
!  end do
!   write(*,*) "phia"
!   write(*,*) phia

  phib = 1.0d0 - phia
  w(1,:,:,:) = chiN*phib
  w(2,:,:,:) = chiN*phia

! keep the level of field value
  call zerofield(w)
! assign large initial value for the energy and error
  energy_tot =1.0d20
  error_level = 1.0d20
!
! iteration begins here
  do iter=1, maxiter
!   for the given fields find the polymer statistics
    call findphi(w(1,:,:,:),w(2,:,:,:),phia,phib,QQ)
    phitot = phia + phib

!   calculate the total energy
    energy_chain = -log(QQ/volume) - (dot(w(1,:,:,:),phia) + dot(w(2,:,:,:),phib))/volume
    energy_field = chiN*dot(phia,phib)/volume
!   calculate the total energy
    energy_old = energy_tot
    energy_tot = energy_chain + energy_field

!   calculate pressure field for the new field calculation, the method is modified from Fredrickson's
    xi = 0.5d0*(w(1,:,:,:)+w(2,:,:,:)-chiN)
!   calculate output fields
    wout(1,:,:,:) = chiN * phib + xi
    wout(2,:,:,:) = chiN * phia + xi
    call zerofield(wout)
!   error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    error_level = sqrt(multidot(wout-w,wout-w)/multidot(w,w))

!   print iteration # and error levels
    write(*,'(I6,F8.3,2F15.9)') iter, mix, energy_tot, error_level
!   condition to start anderson mixing
    if(error_level < 1.0d-2) start_anderson = min(start_anderson,iter)
!   conditions to end the iteration
    if(error_level < 5.0d-8) exit
!   conditions to apply the simple mixing method
    if(iter <= start_anderson) then
!     just before anderson mixing start accumulating past fields
      if(iter == start_anderson) then
        wout_hist(mod(iter,m),:,:,:,:) = wout
        wdiff_hist(mod(iter,m),:,:,:,:) = wout - w
        wdiffdots(mod(iter,m),mod(iter,m)) &
          = multidot(wdiff_hist(mod(iter,m),:,:,:,:),wdiff_hist(mod(iter,m),:,:,:,:))
      end if
!     dynamically change mixing parameter
      if(old_error_level < error_level) then
        mix = max(mix*0.7d0,min_mix)
      else
        mix = mix*1.01d0
      end if
!     make a simple mixing of input and output fields for the next iteration
      w = (1.0d0-mix)*w + mix*wout
!
    else
!     anderson mixing begins here
!     number of histories to use for anderson mixing
      n_anderson = min(max_anderson,iter-start_anderson)
!     store the input and output field (the memory is used in a periodic way)
      wout_hist(mod(iter,m),:,:,:,:) = wout
      wdiff_hist(mod(iter,m),:,:,:,:) = wout - w
!     evaluate wdiff dot products for calculating Unm and Vn in Thompson's paper
      wdiffdots(mod(iter,m),mod(iter,m)) &
        = multidot(wdiff_hist(mod(iter,m),:,:,:,:),wdiff_hist(mod(iter,m),:,:,:,:))
      do nit=1,n_anderson
        wdiffdots(mod(iter,m),mod(iter-nit,m)) &
          = multidot(wdiff_hist(mod(iter,m),:,:,:,:),wdiff_hist(mod(iter-nit,m),:,:,:,:))
        wdiffdots(mod(iter-nit,m),mod(iter,m)) = wdiffdots(mod(iter,m),mod(iter-nit,m))
      end do
!     calculate Unm and Vn
      do nit=1,n_anderson
        v_n(nit) =  wdiffdots(mod(iter,m),mod(iter,m)) &
          - wdiffdots(mod(iter,m),mod(iter-nit,m))
        do nit2=1,n_anderson
          u_nm(nit,nit2) =  wdiffdots(mod(iter,m),mod(iter,m)) &
            - wdiffdots(mod(iter,m),mod(iter-nit,m))&
            - wdiffdots(mod(iter-nit2,m),mod(iter,m))&
            + wdiffdots(mod(iter-nit,m),mod(iter-nit2,m))
        end do
      end do
!
      call find_an(u_nm(1:n_anderson,1:n_anderson),v_n(1:n_anderson),a_n(1:n_anderson),n_anderson)
!     calculate the new field
      w = wout_hist(mod(iter,m),:,:,:,:)
      do nit=1,n_anderson
        w = w + a_n(nit)*(wout_hist(mod(iter-nit,m),:,:,:,:)-wout_hist(mod(iter,m),:,:,:,:))
      end do
    end if
!
!   print intermediate outputs
    if(mod(iter,100) == 0) then
      write (filename, '( "fields.", I0.6)' ) iter
      open(20,file=filename,status='unknown')
      write(20,'(I8,3F8.3,2F13.9)') iter, mix, chiN, f, energy_tot, error_level
      write(20,'(8I5,3F8.3)') BOX_DIMENSION, NN, x%lo, y%lo, z%lo, x%hi, y%hi, z%hi, Lx, Ly, Lz
      write(20,*) " "
      do i=x%lo,x%hi
        do j=y%lo,y%hi
          do k=z%lo,z%hi
            write(20,'(4F12.6)') w(1,i,j,k), phia(i,j,k), w(2,i,j,k), phib(i,j,k)
          end do
        end do
      end do
      close(20)
    end if
! the main loop ends here
  end do
!
! write the final output
  open(30,file='fields.dat',status='unknown')
  write(30,'(I8,3F8.3,2F13.9)') iter, mix, chiN, f, energy_tot, error_level
  write(30,'(8I5,3F8.3)') BOX_DIMENSION, NN, x%lo, y%lo, z%lo, x%hi, y%hi, z%hi, Lx, Ly, Lz
  write(30,*) " "
  do i=x%lo,x%hi
    do j=y%lo,y%hi
      do k=z%lo,z%hi
        write(30,'(4F12.6)') w(1,i,j,k), phia(i,j,k), w(2,i,j,k), phib(i,j,k)
      end do
    end do
  end do
  close(30)

! finalize FFT
  call fft_finalize()
!
end program block_copolymer
!
!--------------------------------------------------------------------
!This subroutine keeps the level of field
subroutine zerofield(w)
!
  use parameters
  implicit none
  real(kind=8), intent(inout) :: w(1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) :: tot1, tot2
! each sum of Wa and Wb is set to zero
  tot1 = sum(seg(:,:,:)*w(1,:,:,:))
  w(1,:,:,:) = w(1,:,:,:) - tot1/volume
  tot2 = sum(seg(:,:,:)*w(2,:,:,:))
  w(2,:,:,:) = w(2,:,:,:) - tot2/volume
!
end subroutine zerofield
!
!--------------------------------------------------------------------
!this function calculates integral g*h
real(kind=8) function dot(g,h)
!
  use parameters
  implicit none
  real(kind=8), intent(in) :: g(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(in) :: h(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  dot = sum(seg(:,:,:)*g(:,:,:)*h(:,:,:))
end function dot
!
!--------------------------------------------------------------------
!this function calculates multiple integrals in z direction
real(kind=8) function multidot(g,h)
!
  use parameters
  implicit none
  real(kind=8), intent(in) :: g(1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(in) :: h(1:2,x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  multidot = sum(seg(:,:,:)*(g(1,:,:,:)*h(1,:,:,:)+g(2,:,:,:)*h(2,:,:,:)))
end function multidot
!
!--------------------------------------------------------------------
! finds the solution for the matrix equation (n*n size) for anderson mixing
! U * A = V, Gauss elimination method is used in its simplest level
subroutine find_an(u,v,a,n)
  implicit none
  integer, intent(in) :: n
  real(kind=8), intent(inout) :: u(1:n,1:n), v(1:n)
  real(kind=8), intent(out) :: a(1:n)
!
  integer :: i,j,k
  real(kind=8) :: factor, tempsum
! Elimination process
  do i=1,n
    do j=i+1,n
      factor = u(j,i)/u(i,i)
      v(j) = v(j) - v(i)*factor
      do k=i+1,n
        u(j,k) = u(j,k) - u(i,k)*factor
      end do
    end do
  end do
!
! find the solution
  a(n) = v(n)/u(n,n)
  do i=n-1,1,-1
    tempsum = 0.0d0
    do j=i+1,n
      tempsum = tempsum + u(i,j)*a(j)
    end do
    a(i) = (v(i) - tempsum)/u(i,i)
  end do
!
end subroutine find_an
!
!--------------------------------------------------------------------
! this subroutine calculates the segment concentration
subroutine findphi(wa,wb,phia,phib,QQ)
!
  use parameters
  implicit none
  real(kind=8), intent(in) ::    wa(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(in) ::    wb(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(out) :: phia(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(out) :: phib(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::                q1(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi,0:NN)
  real(kind=8) ::                q2(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::            expdwa(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::            expdwb(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::       expdwa_half(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::       expdwb_half(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(out) :: QQ
  real(kind=8) :: dot, temp
  integer :: n, i

  expdwa(:,:,:) = exp(-wa(:,:,:)*ds*0.5d0)
  expdwb(:,:,:) = exp(-wb(:,:,:)*ds*0.5d0)
  expdwa_half(:,:,:) = exp(-wa(:,:,:)*ds*0.25d0)
  expdwb_half(:,:,:) = exp(-wb(:,:,:)*ds*0.25d0)

! q1 is q and q2 is qdagger in the note
! free end initial condition (q1 starts from A end)
  q1(:,:,:,0) = 1.0d0

  ! do i=x%lo,x%hi
  !     q1(i,:,:,0) = exp(-((i-x%lo+0.5)*dx-Lx/2)**2/ (2*0.5**2))
  ! end do

! diffusion of A chain
  do n=1,NNf
    call pseudo(q1(:,:,:,n-1),q1(:,:,:,n),expdwa,expdwa_half)
    ! write(*,*) n, q1(:,:,:,n-1), sum(q1(:,:,:,n-1))
  end do
!
! diffusion of B chain
  do n=NNf+1,NN
    call pseudo(q1(:,:,:,n-1),q1(:,:,:,n),expdwb,expdwb_half)
  end do

! q2 starts from B end
  q2 = 1.0d0
! segment concentration. only half contribution from the end
  phib = 0.5d0*q1(:,:,:,NN)
!
! diffusion of B chain
  do n=NN-1,NNf+1,-1
    call pseudo(q2,q2,expdwb,expdwb_half)
    phib = phib + q1(:,:,:,n)*q2
  end do
! diffusion of the junction segment
  call pseudo(q2,q2,expdwb,expdwb_half)
! calculates the total partition function
  QQ = dot(q1(:,:,:,NNf),q2)
! the junction is half A and half B
  phia = 0.5d0*q1(:,:,:,NNF)*q2
  phib = phib + phia
!
! diffusion of A chain
  do n=NNf-1,1,-1
    call pseudo(q2,q2,expdwa,expdwa_half)
    phia = phia + q1(:,:,:,n)*q2
  end do
! last segment diffusion (A end)
  call pseudo(q2,q2,expdwa,expdwa_half)
! only half contribution from the end
  phia = phia + 0.5d0*q2
! normalize the concentration
  phia = phia*volume/QQ/NN
  phib = phib*volume/QQ/NN
! check the mass conservation (if the program is working correctly, temp = 1.0)
  temp = sum(seg*(phia+phib))/volume
  write(*,'(D13.3,D18.8)') temp-1.0d0, QQ
!
end subroutine findphi
!--------------------------------------------------------------------
! this subroutine integrates the block copolymer partition function, q,
! one time-step, ds, forward where expdw = exp(-w*ds/2), expdw_half = exp(-w*ds/4) and w is the field.
subroutine pseudo(qin,qout,expdw,expdw_half)

  use parameters
  use fft
  implicit none
  real(kind=8), intent(in) ::        qin(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(out) ::      qout(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(in) ::      expdw(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8), intent(in) :: expdw_half(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::                  qout1(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)
  real(kind=8) ::                  qout2(x%lo:x%hi,y%lo:y%hi,z%lo:z%hi)

! step 1
  ! write(*,*) "qin"
  ! write(*,*) qin, sum(qin)

  qout1 = expdw*qin                               ! evaluate e^(-w*ds/2) in real space
  call fftw_execute_r2r(plan_forward, qout1,qout1)! fourier discrete transform, forward and inplace

  ! write(*,*) "dct-2"
  ! write(*,*) qout1, sum(qout1)

  qout1 = expfactor*qout1                         ! multiply e^(-k^2 ds/6) in fourier space, in all 3 directions

  ! write(*,*) "dct-2*exp"
  ! write(*,*) qout1, sum(qout1)

  call fftw_execute_r2r(plan_backward,qout1,qout1)! fourier discrete transform, backword and inplace
  qout1 = expdw*qout1/dividedby                   ! normalization calculation and evaluate e^(-w*ds/2) in real space

  ! write(*,*) "qout1"
  ! write(*,*) qout1, sum(qout1)

  ! call exit(1)

! step 2
  qout2 = expdw_half*qin                          ! evaluate e^(-w*ds/4) in real space
  call fftw_execute_r2r(plan_forward, qout2,qout2)! fourier discrete transform, forward and inplace
  qout2 = expfactor_half*qout2                    ! multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
  call fftw_execute_r2r(plan_backward,qout2,qout2)! fourier discrete transform, backword and inplace
  qout2 = expdw*qout2/dividedby                   ! normalization calculation and evaluate e^(-w*ds/2) in real space
  call fftw_execute_r2r(plan_forward, qout2,qout2)! fourier discrete transform, forward and inplace
  qout2 = expfactor_half*qout2                    ! multiply e^(-k^2 ds/12) in fourier space, in all 3 directions
  call fftw_execute_r2r(plan_backward,qout2,qout2)! fourier discrete transform, backword and inplace
  qout2 = expdw_half*qout2/dividedby              ! normalization calculation and evaluate e^(-w*ds/4) in real space

  qout = (4.0d0*qout2 - qout1)/3.0d0

end subroutine pseudo
!--------------------------------------------------------------------
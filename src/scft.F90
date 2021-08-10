!--------------------------------------------------------------------
! The main program begins here
module scft

  use param_parser
  use simulation_box
  use polymer_chain
  use pseudo
  use anderson_mixing

  implicit none  
! iter = number of iteration steps, maxiter = maximum number of iteration steps
  integer :: iter, maxiter
! QQ = total partition function
  real(kind=8), private :: QQ, energy_chain, energy_field, energy_tot, energy_old
! error_level = variable to check convergence of the iteration
  real(kind=8), private :: error_level, old_error_level, tolerance
! chiW is the surface interaction strength
! the external field
! ext_w_plus = (x_wa + x_wb)/2
! ext_w_minus = (x_wa - x_wb)/2
  real(kind=8), private :: chiW
  real(kind=8), private, allocatable :: ext_w_plus(:,:,:)
  real(kind=8), private, allocatable :: ext_w_minus(:,:,:)
! input and output fields, xi is temporary storage for pressures
  real(kind=8), allocatable :: w(:,:,:,:), wout(:,:,:,:), xi(:,:,:)
  real(kind=8), allocatable :: w_plus(:,:,:), w_minus(:,:,:)
! initial value of q, q_dagger
  real(kind=8), allocatable :: q1_init(:,:,:), q2_init(:,:,:)
! segment concentration
  real(kind=8), allocatable :: phia(:,:,:), phib(:,:,:), phitot(:,:,:)
! fields mixing method
  integer :: mixing_method
! strings to name input and output files
  character(len=256), private :: filename, print_filename

! input parameters
  integer :: nx, ny, nz
  real(kind=8) :: new_Lx, new_Ly, new_Lz
  real(kind=8) :: new_f, new_chiN
  integer :: new_NN

contains
!-------------- initialize ------------
  subroutine initialize(input_param_file)
    character(len=*), intent(in) :: input_param_file
    integer :: i, j, k

!   initialize param parser parameters
    call pp_initialize(input_param_file)


    if(.not. pp_get("geometry.grids", nx, 1) ) then
      write(*,*) "geometry.grids[1] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", ny, 2) ) then
      write(*,*) "geometry.grids[2] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.grids", nz, 3) ) then
      write(*,*) "geometry.grids[3] is not specified."
      stop (1)
    end if

    if(.not. pp_get("geometry.box_size", new_Lx, 1) ) then
      write(*,*) "geometry.box_size[1] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.box_size", new_Ly, 2) ) then
      write(*,*) "geometry.box_size[2] is not specified."
      stop (1)
    end if
    if(.not. pp_get("geometry.box_size", new_Lz, 3) ) then
      write(*,*) "geometry.box_size[3] is not specified."
      stop (1)
    end if
    
!   initialize simulation box parameters
    call box_initialize(nx, ny, nz, new_Lx, new_Ly, new_Lz)

    if(.not. pp_get("chain.a_fraction", new_f)) then
      write(*,*) "chain.a_fraction is not specified."
      stop (1)
    end if
    if(.not. pp_get("chain.contour_step", new_NN)) then
      write(*,*) "chain.contour_step is not specified."
      stop (1)
    end if
    if(.not. pp_get("chain.chiN", new_chiN)) then
      write(*,*) "chain.chiN is not specified."
      stop (1)
    end if

!   initialize chain parameters
    call chain_initialize(new_f, new_NN, new_chiN)
!   initialize pseudo spectral parameters
    call pseudo_initialize()
!   initialize Anderson mixing
    call am_initialize(2) ! number of components of w, wout array. e.g. w(:,:,:,1:2)

    if(.not. pp_get("iter.tolerance", tolerance)) tolerance = 5.0d-9
    if(.not. pp_get("iter.step_saddle", maxiter)) maxiter   = 10
    if(.not. pp_get("iter.mixing_method", mixing_method)) mixing_method   = 0
    if(.not. pp_get("chain.surface_interaction", chiW)) chiW = 0.0d0

    ! assign large initial value for the energy and error
    energy_tot = 1.0d20
    error_level = 1.0d20

    write (print_filename, '( "print_", I0.6, "_", I0.4,  ".txt")' ) &
      nint(chiW), nint(chiN*100)

!-------------- print simulation parameters ------------
    open(10, file=print_filename, status = 'unknown', position='append')
    write(10,*) "Box Dimension: 3"
    write(10,*) "Precision: 8"
    write(10,*) "chiN, f, NN"
    write(10,*)  chiN, f, NN
    write(10,*) "x_lo, x_hi, y_lo, y_hi, z_lo, z_hi"
    write(10,*)  x_lo, x_hi, y_lo, y_hi, z_lo, z_hi
    write(10,*) "Lx, Ly, Lz, dx, dy, dz"
    write(10,*)  Lx, Ly, Lz, dx, dy, dz
    write(10,*) "chiW: ", chiW
    write(10,*) "volume, sum(seg)", volume, sum(seg)
    close(10)

!-------------- allocate array ------------

!   define arrays for field and density
    allocate(w          (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi,1:2))
    allocate(wout       (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi,1:2))

    allocate(ext_w_plus (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(ext_w_minus(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

    allocate(xi         (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(phia       (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(phib       (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(phitot     (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(w_plus     (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(w_minus    (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

    allocate(q1_init    (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))
    allocate(q2_init    (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

!-------------- setup fields ------------
    ! assign surface interaction

    ext_w_plus(:,:,:) = 0.0d0
    ext_w_minus(:,:,:) = 0.0d0

    ext_w_plus (x_lo,:,:) = chiW/2
    ext_w_plus (x_hi,:,:) = chiW/2

    ext_w_minus(x_lo,:,:) =-chiW/2
    ext_w_minus(x_hi,:,:) = chiW/2

    call random_number(phia)
!   phia = reshape( phia, (/ x_hi-x_lo+1,y_hi-y_lo+1,z_hi-z_lo+1 /), order = (/ 3, 2, 1 /))
!   call random_number(phia(:,:,z_lo))
!   do k=z_lo,z_hi
!     phia(:,:,k) = phia(:,:,z_lo)
!   end do

    phib = 1.0d0 - phia
    w(:,:,:,1) = chiN*phib
    w(:,:,:,2) = chiN*phia

    ! keep the level of field value
    call zeromean(w(:,:,:,1))
    call zeromean(w(:,:,:,2))

!   q1 is q and q2 is qdagger in the note
!   free end initial condition (q1 starts from A end)
    q1_init(:,:,:) = 1.0d0
!   q2 starts from B end
    q2_init(:,:,:) = 1.0d0

  end subroutine
!------------------ run ----------------------
  subroutine run()
    write(*,'(2A)') 'iteration, mass error, total_partition, ', &
      'energy_tot, error_level'
!   iteration begins here
    do iter=1, maxiter
!     for the given fields find the polymer statistics
      call pseudo_run(phia, phib, QQ, q1_init,q2_init, w(:,:,:,1), w(:,:,:,2))
      
!     calculate the total energy
      energy_old = energy_tot
      w_minus = (w(:,:,:,1)-w(:,:,:,2))/2
      w_plus = (w(:,:,:,1)+w(:,:,:,2))/2
      energy_tot = -log(QQ/volume) &
        + sum(seg(:,:,:)*w_minus(:,:,:)**2)/chiN/volume &
        - sum(seg(:,:,:)*w_plus(:,:,:))/volume &
        + sum(seg(:,:,:)*2*ext_w_minus(:,:,:)*w_minus(:,:,:))/chiN/volume

!     error_level measures the "relative distance" between the input and output fields
      old_error_level = error_level
!     calculate pressure field for the new field calculation, the method is modified from Fredrickson's
      xi = 0.5d0*(w(:,:,:,1)+w(:,:,:,2)-chiN)
!     calculate output fields
      wout(:,:,:,1) = chiN*phib + xi
      wout(:,:,:,2) = chiN*phia + xi
      call zeromean(wout(:,:,:,1))
      call zeromean(wout(:,:,:,2))

      error_level = sqrt(multidot(2,wout-w,wout-w)/(multidot(2,w,w)+1.0d0))
!     print iteration # and error levels and check the mass conservation
      write(*,'(I6,D13.3,D18.8)', advance='no') iter, sum(seg*(phia+phib))/volume-1.0d0, QQ
      write(*,'(2F18.9)') energy_tot, error_level

!     conditions to end the iteration
      if(error_level < tolerance) exit

!     calculte new fields using simple and Anderson mixing
      call am_caculate_new_fields(w, wout, old_error_level, error_level)
!   the main loop ends here
    end do

!------------- write the final output -------------
    write (filename, '( "fields_", I0.6, "_", I0.4, "_", I0.7, ".dat")' ) &
      nint(chiW), nint(chiN*100), nint(Lx*1000)
    call write_data(filename)
  end subroutine

!------------- finalize -------------
  subroutine finalize
!   finalize pseudo_spectral module
    call pseudo_finalize()
!   finalize Anderson mixing
    call am_finalize()
!   finalize simulation box
    call box_finalize()
!   finalize polymer chain
    call chain_finalize
!   finalize param parser
    call pp_finalize()
  end subroutine
!------------- write the final output -------------
! this subroutine write the final output
  subroutine write_data(filename)
    character (len=*), intent(in) :: filename
    integer :: i, j, k
    open(30,file=filename,status='unknown')
    write(30,'(A,I8)') "iter : ", iter
    write(30,'(A,F8.3)') "chain.chiN : ", chiN
    write(30,'(A,F8.3)') "chain.a_fraction : ", f
    write(30,'(A,F8.3)') "chain.surface_interaction : ", chiW
    write(30,'(A,I8)')   "chain.contour_step : ",  NN
    write(30,'(A,I8)')   "chain.contour_step_A : ", NNf
    write(30,'(A,3I8)')   "geometry.grids_lower_bound : ", x_lo, y_lo, z_lo
    write(30,'(A,3I8)')   "geometry.grids_upper_bound : ", x_hi, y_hi, z_hi
    write(30,'(A,3F8.3)') "geometry.box_size : ", Lx, Ly, Lz
    write(30,'(A,F13.9)') "energy_tot : ", energy_tot
    write(30,'(A,F13.9)') "error_level : ", error_level

    write(30,*) " "
    write(30,*) "DATA      # w_a, phi_a, w_b, phi_b"
    do i=x_lo,x_hi
      do j=y_lo,y_hi
        do k=z_lo,z_hi
          write(30,'(5F12.6)') w(i,j,k,1), phia(i,j,k), w(i,j,k,2), phib(i,j,k), phia(i,j,k)+phib(i,j,k)
        end do
      end do
    end do
    close(30)
  end subroutine
end module scft

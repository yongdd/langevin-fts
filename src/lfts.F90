!--------------------------------------------------------------------
! The main program begins here
module lfts

  use constants, only : rp
  use param_parser
  use simulation_box
  use polymer_chain
  use parallel_tempering
  use anderson_mixing
  use pseudo
  use langevin_equation
  use random_gaussian
  implicit none

! iter = number of iteration steps, maxiter = maximum number of iteration steps
  integer :: i, j, k, iter, maxiter
! variables for langevin iteration
  integer :: langevin_iter, langevin_start_iter, langevin_end_iter
! intermediate outputs steps
  integer :: print_period_before_ensemble, print_period_after_ensemble
! when to start to record ensemble average
  integer :: start_record_ensemble
! QQ = total partition function
  real(kind=rp) :: QQ
  real(kind=dp) :: energy_tot, energy_old
! error_level = variable to check convergence of the iteration
  real(kind=dp) :: error_level, old_error_level
  real(kind=dp) :: tolerance
! input and output fields
  real(kind=rp), allocatable :: wplus (:,:,:), wplus_out (:,:,:)
  real(kind=rp), allocatable :: wminus(:,:,:)
! etaas is the surface interaction strength
! ext_w is the external field
  real(kind=rp) :: etaas
  real(kind=rp), allocatable :: ext_w(:,:,:)
! segment concentration
  real(kind=rp), allocatable :: phia(:,:,:), phib(:,:,:), phiplus(:,:,:), phiminus(:,:,:)
! initial value of q, q_dagger
  real(kind=rp), allocatable :: q1_init(:,:,:), q2_init(:,:,:)
! Parallel tempering variables
  integer :: tempering_period
! strings to name input and output files
  character (len=256) :: print_filename, output_filename
  integer :: unit_print, unit_output
  integer :: verbose_level

contains
!-------------- initialize -----------------------------
  subroutine initialize(param_file)
    character(len=*), intent(in) :: param_file

!   initialize param parser parameters
    call pp_initialize(param_file)
!   initialize simulation box
    call box_initialize()
!   initialize chain parameters
    call chain_initialize()
!   initialize parallel tempering
    call pt_initialize()
!   initialize pseudo spectral parameters
    call pseudo_initialize(pt_myid)
!   initialize Anderson mixing
    call am_initialize(1, pt_myid) ! the number of components
!   initialize Langevin equation module
    call le_initialize()
!   initialize lfts parameters
    call lfts_initialize()

!-------------- print simulation parameters ------------
    write(unit_print,*) "Precision: ", rp

    write(unit_print,*) "langevin_start_iter, langevin_end_iter"
    write(unit_print,*)  langevin_start_iter, langevin_end_iter
    write(unit_print,*) "delta_tau, normal_sigma"
    write(unit_print,*)  delta_tau, normal_sigma
    write(unit_print,*) "invariant_polymerization_index_Nbar"
    write(unit_print,*)  invarint_Nbar
    write(unit_print,*) "chiN, f, NN"
    write(unit_print,*)  chiN, f, NN
    write(unit_print,*) "x%lo, x%hi, y%lo, y%hi, z%lo, z%hi"
    write(unit_print,*)  x%lo, x%hi, y%lo, y%hi, z%lo, z%hi
    write(unit_print,*) "Lx,Ly,Lz,dx,dy,dz"
    write(unit_print,*)  Lx,Ly,Lz,dx,dy,dz
    write(unit_print,*) "seg(:,y%lo,z%lo)"
    write(unit_print,*)  seg(:,y%lo,z%lo)
    write(unit_print,*) "etaas: ", etaas
    write(unit_print,*) "volume, sum(seg)"
    write(unit_print,*)  volume, sum(seg)
    write(unit_print,*) "wminus(:,y%lo,z%lo)"
    write(unit_print,*)  wminus(:,y%lo,z%lo)
    write(unit_print,*) "wplus(:,y%lo,z%lo)"
    write(unit_print,*)  wplus(:,y%lo,z%lo)
    write(unit_print,*) "ext_w(:,y%lo,z%lo)"
    write(unit_print,*)  ext_w(:,y%lo,z%lo)

!   q1 is q and q2 is qdagger in the note
!   free end initial condition (q1 starts from A end)
    q1_init(:,:,:) = 1.0d0
!   q2 starts from B end
    q2_init(:,:,:) = 1.0d0

    langevin_iter=langevin_start_iter
    write(unit_print,'(A)') "------------------ run --------------------"
    write(unit_print,'(2A)') 'iteration, mass error, total_partition, ', &
      'energy_tot, error_level'
  end subroutine

!------------------ run ----------------------------------
  subroutine pre_langevin()
    if(verbose_level >= 1 ) then
      write(unit_print,'(A,I8)') "langevin_iter", langevin_iter
    end if
    !------------- Update exchange field using Langeving euqation -----------
    if (langevin_iter /= 1) then
      call le_update(phia, phib, q1_init,q2_init, wplus, wminus, ext_w)
    end if
    !------------- Parallel Tempering ------------------------
    if(mod(langevin_iter,tempering_period) == 0) then
      call pt_attempt(chiN, rho_a3, wminus, wplus)
    end if
  end subroutine

  subroutine find_saddle_point()
    !------------- Partial saddle point approximation --------
    ! assign large initial value for the energy and error
    energy_tot = 1.0d20
    error_level = 1.0d20

    ! Reset Anderson mixing module
    call am_reset_count()

    ! Saddle point iteration begins here
    do iter=1, maxiter
      ! for the given fields find the polymer statistics
      call pseudo_run(phia, phib, QQ, q1_init,q2_init, wplus+wminus,wplus-wminus)
      phiplus  = phia + phib
      phiminus = phia - phib

      ! calculate the total energy
      energy_old = energy_tot
      energy_tot = -log(QQ/(Lx*Ly*Lz)) + (sum((wminus-ext_w)**2*seg)/chiN - sum(wplus*seg))/volume

      !  error_level measures the "relative distance" between the input and output fields
      old_error_level = error_level
      error_level = sqrt(dot(phiplus-1.0d0,phiplus-1.0d0)/volume)

      ! print iteration # and error levels
      if( verbose_level == 2 .or. &
        verbose_level == 1 .and. &
        (error_level < tolerance .or. iter == maxiter ))  then
        ! check the mass conservation (if the program is working correctly, temp = 1.0)
        write(unit_print,'(I6,D13.3,D18.8)', advance='no') &
          iter, sum(seg*(phiplus))/volume-1.0d0, QQ
        write(unit_print,'(2F15.9)') energy_tot, error_level
      end if
      !  conditions to end the iteration
      if(error_level < tolerance) exit

      !  calculte new fields using simple and Anderson mixing
      wplus_out(:,:,:) = wplus(:,:,:) + 1.0d0*(phiplus - 1.0d0)
      call zeromean(wplus_out(:,:,:))
      call am_caculate_new_fields(wplus, wplus_out, old_error_level, error_level)

      ! scft loop ends here
    end do

  end subroutine

  subroutine post_langevin()
    !-------------  Print intermediate outputs ---------------
    if((mod(langevin_iter,print_period_before_ensemble) == 0 .or. &
      (mod(langevin_iter,print_period_after_ensemble) == 0 .and. &
      langevin_iter > start_record_ensemble))) then
      write (output_filename, '(I0.6, "_", I0.4, ".", I0.8)' ) &
        nint(etaas*1000), nint(chiN*100), langevin_iter
      call write_data(output_filename)
    end if
    !------------- Increase Langevin iteration count ---------
    langevin_iter = langevin_iter + 1
  end subroutine

!------------- write the final output ---------------------
  subroutine write_final_output()
    write (output_filename, '(I0.6, "_", I0.4, ".dat")' ) &
      nint(etaas*1000), nint(chiN*100)
    call write_data(output_filename)
  end subroutine

!-------------- finalize --------------------------------------
  subroutine finalize()

    close(unit_print)
! finalize Langevin equation module
    call le_finalize()
! finalize pseudo_spectral module
    call pseudo_finalize()
! finalize Anderson mixing
    call am_finalize()
! finalize parallel tempering
    call pt_finalize()
! finalize simulation box
    call box_finalize()
! finalize polymer chain
    call chain_finalize
! finalize parameter parser
    call pp_finalize()

  end subroutine
!-------------- initialize L-FTS parameters-----------------
  subroutine lfts_initialize()

    ! strings to name input and output files
    character(len=256) :: input_filename, root_name
    character(len=1024) :: buf
    ! seed_array
    integer, allocatable :: seed_array(:)
!   fields type
    integer :: type_continue, type_fields
!   file_exists = logical for file search
    logical :: file_exists
!   strings for  input file name
    integer :: unit_input

!----------------- output streams --------------------------
    if(.not. pp_get("chain.surface_interaction", etaas)) etaas = 0.0d0

    unit_print = 10
    unit_output = 30

    if(.not. pp_get("output.verbose_level", verbose_level)) verbose_level = 1
    if(.not. pp_get("output.verbose_name", root_name)) root_name = "print"
    write (print_filename, '(I0.6, "_", I0.4, ".txt")' ) &
      nint(etaas*1000), nint(chiN*100)
    print_filename = trim(root_name)//"_"//trim(print_filename)
    open(unit_print, file=print_filename, status = 'unknown', position='append')

!------------------ Langevin paramters  -------------------------------
!   error tolerance
    if(.not. pp_get("iter.tolerance", tolerance)) tolerance = 1.0d-4
!   iteration must stop before iter = scft_maxiter happens
    if(.not. pp_get("iter.saddle_step_max", maxiter)) maxiter   = 100
!   minimum and maximum iteration for Langevin simulation
    langevin_start_iter = 1
    if(.not. pp_get("langevin_end_iter", langevin_end_iter)) langevin_end_iter = 10
!   when to start to record ensemble average
    if(.not. pp_get("start_record_ensemble", start_record_ensemble)) start_record_ensemble = 100000

!----------------- Intermediate operations ----------------
!   record intermediate outputs in every printstep1
    if(.not. pp_get("output.period_before_ensemble", print_period_before_ensemble)) print_period_before_ensemble = 1000
!   record intermediate outputs in every printstep2 after start_record_ensemble
    if(.not. pp_get("output.period_after_ensemble", print_period_after_ensemble)) print_period_after_ensemble = 100
!   parallel temperaing is attempted in every tempering_period
    if(.not. pp_get("pt.period", tempering_period)) tempering_period = 1000

!---------------- allocate array ---------------------------
!   define arrays for field and density
    allocate(wplus       (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(wplus_out   (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(wminus      (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(ext_w       (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(phia        (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(phib        (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(phiplus     (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(phiminus    (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(q1_init    (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))
    allocate(q2_init    (x%lo:x%hi,y%lo:y%hi,z%lo:z%hi))

!-------------- Setup initial fields ----------------------
    if(.not. pp_get("input.continue", type_continue)) type_continue = 0

    input_filename = ""
    ! continue from fields of last data
    if(type_continue == 1) then
      if(.not. pp_get("input.cont_iter", i)) then
        write(*,*) "'input.cont_iter' is not specified"
        stop (1)
      end if
      langevin_start_iter = i + 1
      write (input_filename, '( "fields_", I0.6, "_", I0.4, ".", I0.8)' ) &
        nint(etaas*1000), nint(chiN*100), i
      inquire(file=input_filename, exist=file_exists)
      if(.not. file_exists) then
        write(*,*) "Could not find input file to continue."
        stop (1)
      end if
      ! continue from fields of inputs file
    else if(type_continue == 2) then
      if(.not. pp_get("input.file_name", input_filename)) then
        write(*,*) "Input file name is not specified."
        stop (1)
      end if
      inquire(file=input_filename, exist=file_exists)
      if(.not. file_exists) then
        write(*,*) "Could not find input file : ", trim(input_filename)
        stop (1)
      end if
    end if

    select case(type_continue)
      ! setup field manually
     case(0)
      if(.not. pp_get("input.select_init_fields", type_fields)) type_fields = 1
      select case(type_fields)
       case(1)
        write(*,*) "wminus and wplus are initialized to zero."
        wminus = 0.0d0
        wplus = 0.0d0
       case(2)
        write(*,*) "Randomly initialize wminus and wplus."
        call random_number(wminus(:,:,:))
        call random_number(wplus(:,:,:))
       case(3)
        write(*,*) "wminus and wplus are initialized to test_input"
        do i=x%lo,x%hi
          do j=y%lo,y%hi
            do k=z%lo,z%hi
              wminus(i,j,k)= cos( 2.0d0*PI*(i-x%lo)/4.68d0 ) &
                * cos( 2.0d0*PI*(j-y%lo)/3.48d0 ) &
                * cos( 2.0d0*PI*(k-z%lo)/2.74d0 ) * 0.1d0
            end do
          end do
        end do
        wplus = 0.0d0
      end select
      ! setup field from an input file
     case(1:2)
      write(*,*) "Reading an input file... ", trim(input_filename)
      open(unit_input, file=input_filename, action='read', status='old')
      i = 1
      do
        read(unit_input, '(A)') buf
        if (index(buf, "DATA") /= 0) exit
        !write(*,*) trim(buf)
        if (index(buf, "random_seed") /= 0) then
          if(pp_get_from_string_int_alloc(buf, seed_array, i)) then
            write(*, '(A)', advance='no') "Resuming with random seed : "
            write(*, *) seed_array
            call rng_set_seed(seed_array, size(seed_array))
          end if
        end if
        i = i +1
      end do
!   read fields and concentrations from the file
      do i=x%lo,x%hi
        do j=y%lo,y%hi
          do k=z%lo,z%hi
            read(unit_input,*) wminus(i,j,k), phiminus(i,j,k), wplus(i,j,k), phiplus(i,j,k)
          end do
        end do
      end do
      close(unit_input)

    end select

!   keep the level of field value
    call zeromean(wminus(:,:,:))
    call zeromean(wplus(:,:,:))
    wminus = wminus + ext_w

  end subroutine
!------------- write the final output ----------------------------
  subroutine write_data(filename)
    character(len=*), intent(in) :: filename
    character(len=256) :: root_name

    if(.not. pp_get("output.data_name", root_name)) root_name = "fields"
    open(unit_output,file=trim(root_name)//"_"//trim(filename),status='unknown')

    write(unit_output,'(A,I8)') "langevin_iter : ", langevin_iter

    write(unit_output,'(A,F8.3)') "chain.chiN : ", chiN
    write(unit_output,'(A,F8.3)') "chain.a_fraction : ", f
    write(unit_output,'(A,F8.3)') "chain.surface_interaction : ", etaas
    write(unit_output,'(A,I8)')   "chain.contour_step : ",  NN
    write(unit_output,'(A,I8)')   "chain.contour_step_A : ", NNf

    write(unit_output,'(A,F8.3)') "langevin.delta_tau : ", delta_tau
    write(unit_output,'(A,F8.3)') "langevin.rho_a3 : ", rho_a3

    write(unit_output,'(A,3I8)')   "geometry.grids_lower_bound : ", x%lo, y%lo, z%lo
    write(unit_output,'(A,3I8)')   "geometry.grids_upper_bound : ", x%hi, y%hi, z%hi
    write(unit_output,'(A,3F8.3)') "geometry.box_size : ", Lx, Ly, Lz

    write(unit_output,'(A,F13.9)') "energy_tot : ", energy_tot
    write(unit_output,'(A,F13.9)') "error_level : ", error_level
    if ( rng_can_get_seed() ) then
      write(unit_output,'(A)', advance ='no') "random_seed : "
      write(unit_output,*)  rng_get_seed()
    end if

    write(unit_output,*) " "
    write(unit_output,*) "DATA      # w-, phi-, w+, phi+"
    do i=x%lo,x%hi
      do j=y%lo,y%hi
        do k=z%lo,z%hi
          write(unit_output,'(4F12.6)') wminus(i,j,k), phiminus(i,j,k), wplus(i,j,k), phiplus(i,j,k)
        end do
      end do
    end do
    close(unit_output)
  end subroutine

!--------- wrapper functions for python module ---------------------

  function get_chin()
    real(kind=8) :: get_chin
    get_chin = chiN
  end function

  function get_process_idx()
    integer :: get_process_idx
    get_process_idx = pt_myid
  end function

  subroutine parallel_barrier()
    call pt_barrier()
  end subroutine

  function get_param_str(param_name, param_value)
    character(len=256), intent(in) :: param_name
    character(len=256), intent(out) :: param_value
    logical :: get_param_str
    if(.not. pp_get(param_name, param_value)) then
      get_param_str = .false.
    else
      get_param_str = .true.
    end if
  end function

  function get_param_int_idx(param_name, idx, param_value)
    character(len=256), intent(in) :: param_name
    integer, intent(in) :: idx
    integer, intent(out) :: param_value
    logical :: get_param_int_idx
    if(.not. pp_get(param_name, param_value, idx)) then
      get_param_int_idx = .false.
    else
      get_param_int_idx = .true.
    end if
  end function

  function get_param_int(param_name, param_value)
    character(len=256), intent(in) :: param_name
    integer, intent(out) :: param_value
    logical :: get_param_int
    if(.not. pp_get(param_name, param_value)) then
      get_param_int = .false.
    else
      get_param_int = .true.
    end if
  end function

  function get_param_float(param_name, param_value)
    character(len=256), intent(in) :: param_name
    real(kind=8), intent(out) :: param_value
    logical :: get_param_float
    if(.not. pp_get(param_name, param_value)) then
      get_param_float = .false.
    else
      get_param_float = .true.
    end if
  end function

end module lfts

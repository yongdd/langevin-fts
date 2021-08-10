!--------------------------------------------------------------------
! The main program begins here
program block_copolymer_3d

  use param_parser
  use simulation_box
  use scft
  implicit none

! constants  
  real(kind=8), parameter :: PI = 3.14159265358979323846d0
! cpu timer
  real(kind=8) :: total_time
  integer :: count0, count1, count_max, count_rate
! input parameter file name
  character(len=256) :: param_file, input_filename
  character(len=1024) :: buf
! initial field type
  integer :: type_fields
  real(kind=8) :: distance
  integer :: i, j, k

!-------------- initialize ------------

  call get_command_argument(1, param_file)
  if ( LEN_TRIM(param_file) == 0) then
    write(*,*) "No parameter file..."
    stop (1)
  end if
  call initialize(param_file)

!-------------- change inital fields ------------
  if(.not. pp_get("input.select_init_fields", type_fields)) type_fields = 1

  select case(type_fields)
   case(0)
    input_filename = "fields.input"
    write(*,*) "Reading an input file... ", trim(input_filename)
    open(40, file=input_filename, action='read', status='old')
    i = 1
    do
      read(40, '(A)') buf
      !write(*,*) "i", i, buf
      if (index(buf, "DATA") /= 0) exit
      i = i + 1
    end do
!     read fields and concentrations from the file
    do i=x_lo,x_hi
      do j=y_lo,y_hi
        do k=z_lo,z_hi
          read(40,*) w(i,j,k,1), phia(i,j,k), w(i,j,k,2), phib(i,j,k), phitot(i,j,k)
        end do
      end do
    end do
    close(40)
   case(1)
    write(*,*) "wminus and wplus are initialized to random inputs"
    call random_number(phia)
   case(2)
    write(*,*) "wminus and wplus are initialized to a given test fields"
    do i=x_lo,x_hi
      do j=y_lo,y_hi
        do k=z_lo,z_hi
          phia(i,j,k)= cos( 2.0d0*PI*(i-x_lo)/4.68d0 ) &
            * cos( 2.0d0*PI*(j-y_lo)/3.48d0 ) &
            * cos( 2.0d0*PI*(k-z_lo)/2.74d0 ) * 0.1d0
        end do
      end do
    end do
  end select
  phib = 1.0d0 - phia
  w(:,:,:,1) = chiN*phib
  w(:,:,:,2) = chiN*phia

! keep the level of field value
  call zeromean(w(:,:,:,1))
  call zeromean(w(:,:,:,2))
!------------------ run ----------------------
! record start time
  call system_clock(count0, count_rate, count_max)

! run
  call run()

! print total simulation time
  call system_clock(count1, count_rate, count_max)
  total_time = real(count1 - count0,8) /count_rate
  write(*,*) "total time,", "time per step"
  write(*,*)  total_time, total_time/(iter-1)

!------------- finalize -------------
  call finalize()

end program block_copolymer_3d

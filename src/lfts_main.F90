!--------------------------------------------------------------------
! The main program begins here
program langevin_fts_block_copolymer_3d

  use constants, only : rp
  use param_parser
  use simulation_box
  use lfts
  implicit none

! cpu timer
  real(kind=rp) :: total_time
  integer(kind=8) :: count0, count1, count_max
  integer :: count_rate, nthreads

! strings to name input and output files
  character (len=256) :: param_file

!-------------- initialize -----------------------------
  call get_command_argument(1, param_file)
  if ( LEN_TRIM(param_file) == 0) then
    write(*,*) "No parameter file..."
    stop (1)
  end if
  call initialize(param_file)

!-------------- run ------------------------------------
! record start time
  call system_clock(count0, count_rate, count_max)

  do while (langevin_iter <= langevin_end_iter)
    call pre_langevin()
    call find_saddle_point()
    call post_langevin()
  end do

! print total simulation time
  call system_clock(count1, count_rate, count_max)
  total_time = real(count1 - count0,rp) /count_rate
  write(*,*) "total time,", "time per step"
  write(*,*)  total_time, total_time/(iter-1)

!-------------- finalize -------------------------------
  call write_final_output()
  call finalize()

end program langevin_fts_block_copolymer_3d

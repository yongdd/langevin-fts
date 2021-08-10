!--------------------------------------------------------------------
! The main program begins here
program param_parser_test

  use param_parser
  implicit none
  integer :: temp_int
  real(kind=8) :: temp_real 

  call pp_initialize("inputs")

  if( .not. pp_get("geometry.grids",temp_int, 1)) temp_int = -1
  if( .not. pp_get("chain.chiN", temp_real)) temp_real = -1.0d0
  write(*,*) "geometry.grids[1]", temp_int
  write(*,*) "chain.chiN", temp_real

  call pp_finalize()

end program param_parser_test

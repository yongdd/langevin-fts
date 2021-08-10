!--------------------------------------------------------------------
! The main program begins here
program circular_buffer_test

  use circular_buffer_module
  implicit none

  ! arrays to calculate anderson mixing
  type(circular_buffer) :: cb_buffer
  !!type(circular_buffer) :: cb_buffer_large
  integer :: total_grids = 5
  integer :: buffer_size = 3
  type(real_array_pt) :: arr_pt
  integer :: i, j

  call cbInit(cb_buffer,  buffer_size, total_grids)

  call cbInsert(cb_buffer, real( (/1,2,3,5,4/),8))
  write(*,'(A)') "cbGetArray"
  do i=0,buffer_size-1
    write(*,'(I5,5F8.3)') i, cbGetArray(cb_buffer, i)
  end do

  call cbInsert(cb_buffer, real( (/4,2,1,1,2/),8))
  write(*,'(A)') "cbGetArray"
  do i=0,buffer_size-1
    write(*,'(I5,5F8.3)') i, cbGetArray(cb_buffer, i)
  end do

  call cbInsert(cb_buffer, real( (/3,2,1,5,4/),8))
  write(*,'(A)') "cbGetArray"
  do i=0,buffer_size-1
    write(*,'(I5,5F8.3)') i, cbGetArray(cb_buffer, i)
  end do

  call cbInsert(cb_buffer, real( (/5,4,3,1,2/),8))
  write(*,'(A)') "cbGetArray"
  do i=0,buffer_size-1
    write(*,'(I5,5F8.3)') i, cbGetArray(cb_buffer, i)
  end do

  call cbInsert(cb_buffer, real( (/2,5,1,4,3/),8))
  write(*,'(A)') "cbGetArray"
  do i=0,buffer_size-1
    write(*,'(I5,5F8.3)') i, cbGetArray(cb_buffer, i)
  end do

  write(*,'(A)') "cbGetArrayPt"
  do i=0,buffer_size-1
    arr_pt = cbGetArrayPt(cb_buffer, i)
    write(*,'(I5,5F8.3)') i, arr_pt%p
  end do

  write(*,'(A)') "cbGet"
  do i=0,buffer_size-1
    write(*,'(I5)', advance='no') i
    do j=0,total_grids-1
      write(*,'(F8.3)', advance='no') cbGet(cb_buffer, i, j)
    end do
    write(*,'(A)') ""
  end do

  write(*,'(A)') "cbGetSym"
  do i=0,buffer_size-1
    write(*,'(I5)', advance='no') i
    do j=0,total_grids-1
      write(*,'(F8.3)', advance='no') cbGetSym(cb_buffer, i, j)
    end do
    write(*,'(A)') ""
  end do

  call cbFree(cb_buffer)

end program circular_buffer_test

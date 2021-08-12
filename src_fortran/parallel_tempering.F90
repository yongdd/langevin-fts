!--------------------------------------------------------------------
! this module defines parameters and subroutines related to parallel
! tempering. MPI(Message Passing Interface) is required for parallel
! tempering.
module parallel_tempering

  use simulation_box
  use polymer_chain
  use MPI
  implicit none

! MPI variables
  integer, protected :: pt_myid, pt_numprocs, PT_MPI_REAL

! Parallel tempering variables
  real(kind=8), private, allocatable :: wswap(:,:,:)
  logical, private :: even_turn

contains
!-------------- pt_initialize ---------------
  subroutine pt_initialize()
    real(kind=8) :: chiN_from, chiN_to
    integer :: ierr, istatus(MPI_STATUS_SIZE)

    pt_myid = 0
    pt_numprocs = 1

    call MPI_INIT( ierr )
    if( ierr /= MPI_SUCCESS) write(*,*) " MPI_INIT , error code=", ierr
    call MPI_COMM_RANK( MPI_COMM_WORLD, pt_myid, ierr )
    if( ierr /= MPI_SUCCESS) write(*,*) " MPI_COMM_RANK , error code=", ierr
    call MPI_COMM_SIZE( MPI_COMM_WORLD, pt_numprocs, ierr )
    if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": MPI_COMM_RANK , error code=", ierr

    allocate(wswap(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi))

    if (pt_numprocs > 1) then
      if(.not. pp_get("pt.chiN_range", chiN_from, 1) .or. &
        .not. pp_get("pt.chiN_range", chiN_to,   2)) then
        write(*,*) "pt.chiN_range is not specified."
      end if
      call chain_set_chiN(chiN_from + pt_myid*(chiN_to - chiN_from)/(pt_numprocs-1))
    end if

#if USE_SINGLE_PRECISION == 1
    PT_MPI_REAL = MPI_REAL4
#else
    PT_MPI_REAL = MPI_REAL8
#endif

    even_turn = .False.

  end subroutine

!-------------- pt_barrier ---------------
  subroutine pt_barrier()
    integer :: ierr
    call MPI_Barrier( MPI_COMM_WORLD, ierr)
  end subroutine 

!-------------- pt_attempt ---------------
  subroutine pt_attempt(chiN, rho_a3, wminus, wplus)

    real(kind=8), intent(in) :: chiN, rho_a3
    real(kind=8), intent(inout) :: wminus(x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8), intent(inout) :: wplus (x_lo:x_hi,y_lo:y_hi,z_lo:z_hi)
    real(kind=8) :: chiN_swap, p_swap, p_random
    logical :: is_receiver, is_sender, is_swap
    integer :: ierr, istatus(MPI_STATUS_SIZE)
    integer :: sender_rank, receiver_rank

    chiN_swap = 0.0d0
    p_swap = 0.0d0
    is_receiver = .FALSE.
    is_sender   = .FALSE.
    ! even case
    if ( even_turn ) then
      if (     mod(pt_myid,2) == 0 .and. 0 < pt_myid) then
        is_receiver = .TRUE.
      elseif ( mod(pt_myid,2) == 1 .and. pt_myid < pt_numprocs-1) then
        is_sender = .TRUE.
      end if
      even_turn = .False.
      ! odd case
    else
      if (     mod(pt_myid,2) == 0 .and. pt_myid < pt_numprocs-1) then
        is_sender = .TRUE.
      elseif ( mod(pt_myid,2) == 1 .and. 0 < pt_myid) then
        is_receiver = .TRUE.
      end if
      even_turn = .True.
    end if

    write(*,'(I5,A,2L3)') pt_myid, ": is_sender, is_receiver", is_sender, is_receiver
    if( is_sender) then
      receiver_rank = pt_myid+1
      write(*,'(I5,A,I5)') pt_myid, ": receiver id ", receiver_rank
      write(*,'(I5,A,F8.3)') pt_myid, ": sending chiN...", chiN
      call MPI_SEND(chiN,          1, PT_MPI_REAL, receiver_rank, 55, MPI_COMM_WORLD, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": SEND chiN , error code=", ierr

      write(*,'(I5,A)') pt_myid, ": sending wminus to other..."
      call MPI_SEND(wminus,  totalMM, PT_MPI_REAL, receiver_rank, 55, MPI_COMM_WORLD, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": SEND wminus , error code=", ierr

      write(*,'(I5,A)') pt_myid, ": receiving swap decision..."
      call MPI_RECV(is_swap,       1, MPI_LOGICAL, receiver_rank, 55, MPI_COMM_WORLD, istatus, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": RECV swap decision , error code=", ierr
      write(*,'(I5,A,I10,4I5)') pt_myid, ": RECV swap decision, [count,cancelled,source,tag,error]=", istatus 

      write(*,'(I5,A,L3)') pt_myid, ": is_swap", is_swap
      if ( is_swap) then
        ! swap wminus
        write(*,'(I5,A)') pt_myid, ": receiving wminus from other..."
        call MPI_RECV(wswap,  totalMM, PT_MPI_REAL, receiver_rank, 55, MPI_COMM_WORLD, istatus, ierr)
        if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": RECV wminus , error code=", ierr
        write(*,'(I5,A,I10,4I5)')  pt_myid, ": RECV wminus, [count,cancelled,source,tag,error]=", istatus 

        wminus = wswap

        ! swap wplus
        write(*,'(I5,A)') pt_myid, ": receiving wplus from other..."
        call MPI_RECV(wswap, totalMM, PT_MPI_REAL, receiver_rank, 55, MPI_COMM_WORLD, istatus, ierr)
        if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": RECV wplus , error code=", ierr
        write(*,'(I5,A,I10,4I5)')  pt_myid, ": RECV wplus, status=", istatus

        write(*,'(I5,A)') pt_myid, ": sending wplus to other..."
        call MPI_SEND(wplus, totalMM, PT_MPI_REAL, receiver_rank, 55, MPI_COMM_WORLD, ierr)
        if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": SEND wplus , error code=", ierr

        wplus = wswap
      end if
    end if

    if( is_receiver) then
      sender_rank = pt_myid-1
      write(*,'(I5,A,I5)') pt_myid, ": sender id ", sender_rank
      write(*,'(I5,A)') pt_myid, ": receiving chiN..."
      call MPI_RECV(chiN_swap,     1, PT_MPI_REAL, sender_rank, 55, MPI_COMM_WORLD, istatus, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": RECV chiN , error code=", ierr
      write(*,'(I5,A,I10,4I5)')  pt_myid, ": RECV chiN, [count,cancelled,source,tag,error]=", istatus 

      write(*,'(I5,A)') pt_myid, ": receiving wminus from other ..."
      call MPI_RECV(wswap,   totalMM, PT_MPI_REAL, sender_rank, 55, MPI_COMM_WORLD, istatus, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": RECV wminus , error code=", ierr
      write(*,'(I5,A,I10,4I5)')  pt_myid, ": RECV wminus, [count,cancelled,source,tag,error]=", istatus 

      p_swap = exp(rho_a3 * sqrt(real(NN,8)) * (1/chiN - 1/chiN_swap) * sum((wminus**2 - wswap**2)*seg))
      call random_number(p_random)
      write(*,'(I5,A,2F8.3)') pt_myid, ": p_swap, p_random,", p_swap, p_random

      if (p_random < p_swap) then
        is_swap = .TRUE.
      else
        is_swap = .FALSE.
      end if

      write(*,'(I5,A,L3)') pt_myid, ": sending swap decision...", is_swap
      call MPI_SEND(is_swap, 1, MPI_LOGICAL, sender_rank, 55, MPI_COMM_WORLD, ierr)
      if( ierr /= MPI_SUCCESS) write(*,*) pt_myid, ": SEND swap decision, error code=", ierr

      if ( is_swap) then
        ! swap wminus
        write(*,'(I5,A)') pt_myid, ": sending wminus..."
        call MPI_SEND(wminus, totalMM, PT_MPI_REAL, sender_rank, 55, MPI_COMM_WORLD, ierr)
        if( ierr /= MPI_SUCCESS) write(*,'(I5,A,I10)') pt_myid, ": SEND wminus , error code=", ierr
        wminus = wswap

        ! swap wplus
        write(*,'(I5,A)') pt_myid, ": sending wplus..."
        call MPI_SEND(wplus, totalMM, PT_MPI_REAL, sender_rank, 55, MPI_COMM_WORLD, ierr)
        if( ierr /= MPI_SUCCESS) write(*,'(I5,A,I10)') pt_myid, ": SEND wplus , error code=", ierr

        write(*,'(I5,A)') pt_myid, ": receiving wplus..."
        call MPI_RECV(wswap, totalMM, PT_MPI_REAL, sender_rank, 55, MPI_COMM_WORLD, istatus, ierr)
        if( ierr /= MPI_SUCCESS) write(*,'(I5,A,I10)') pt_myid, ": RECV wplus , error code=", ierr
        write(*,'(I5,A,I10,4I5)')  pt_myid, ": RECV wplus, [count,cancelled,source,tag,error]=", istatus 

        wplus = wswap
      end if
    end if

    write(*,'(I5,A)') pt_myid, ": waiting for synchronization..."
    call MPI_BARRIER(MPI_COMM_WORLD, ierr) 
    if( ierr /= MPI_SUCCESS) write(*,'(I5,A,I10)') pt_myid, ": MPI_BARRIER , error code=", ierr
    write(*,'(I5,A)') pt_myid, ": END."
    
  end subroutine
!-------------- pt_finalize ---------------
  subroutine pt_finalize
    integer :: ierr
    ! MPI Finalize
    call MPI_FINALIZE(ierr)
    if( ierr /= MPI_SUCCESS) write(*,*) " MPI_FINALIZE , error code=", ierr
    deallocate(wswap)
  end subroutine
end module parallel_tempering

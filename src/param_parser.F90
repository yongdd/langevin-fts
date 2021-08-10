!--------------------------------------------------------------------
! This is a parser implemented using regular expression (RE)
! and deterministic finite automata (DFA). This module reads input
! parameters from an input file as well as command line arguments.
! Each parameter pair are stored in static arrays, and retrieve them
! when 'pp_get' is invoked.
!-------------- Regular Expression ---------------------------------
! s = space
! t = tab
!
! word,   w = [a-zA-Z_]
! digit,  i = [0-9]
! assign, g = [=:]
! dot,        .
! quote,      "
! exponent, x = (d|D|e|E)
! sign, p = (+|-)
!
! blank,  b = (s|t)
! name,   c = (w|i)+(.(w|i)+)?
! number, n = p?i+(.i*)?(xp?i+)?
! string, s = "(w|i|.)*"
! value,  v = (n|s)
! all,    a = [all chatracter]
!
! Syntax :
! (b*cb*gb*v(b+v)*)?b*(#a*)?
!
!
!-------------- Transition DFA Table---------------------------------
!            |       | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
! state\input| Type  | b |w^x| g | p | i | . | x | " | # | Other|
!-------------------------------------------------------
!   1        | accept| 1 |12 |   |   |12 |   |12 |   |11 |    |
!   2(V)     |       |   | 2 |   |   | 2 | 2 | 2 | 3 |   |    |
!   3(V)     | accept| 7 |   |   |   |   |   |   |   |11 |    |
!   4(V)     | accept| 7 |   |   |   | 4 |   |10 |   |11 |    |
!   5(V)     | accept| 7 |   |   |   | 5 |   |   |   |11 |    |
!   6(V)     |       |   |   |   |   | 5 |   |   |   |   |    |
!   7        | accept| 7 |   |   | 9 | 8 |   |   | 2 |11 |    |
!   8(V)     | accept| 7 |   |   |   | 8 | 4 |10 |   |11 |    |
!   9(V)     |       |   |   |   |   | 8 |   |   |   |   |    |
!   10(V)    |       |   |   |   | 6 | 5 |   |   |   |   |    |
!   11       | accept|   |   |   |   |   |   |   |   |   |    |
!   12(S)    |       |15 |12 |14 |   |12 |13 |12 |   |   |    |
!   13(S)    |       |   |16 |   |   |16 |   |16 |   |   |    |
!   14       |       |14 |   |   | 9 | 8 |   |   | 2 |   |    |
!   15       |       |15 |   |14 |   |   |   |   |   |   |    |
!   16(S)    |       |15 |16 |14 |   |16 |   |16 |   |   |    |
!------------------------------------------------------------------------
! Blank cells in above table are all Syntax Error
!-----------------------------------------------------------------------
module param_parser
  use iso_fortran_env, only : iostat_end
  implicit none

  integer, private, parameter :: MAX_STRING_LENGTH  = 256
  integer, private, parameter :: MAX_TOKEN_LENGTH  = 64
  integer, private, parameter :: MAX_TOKEN_NUM = 256
  integer, private, parameter :: MAX_ITEM  = 256

  ! variable and value.
  type param_pair
    character(len=MAX_TOKEN_LENGTH) :: var_name
    character(len=MAX_TOKEN_LENGTH), allocatable :: values(:)
  end type param_pair

  ! parameters will be stored here.
  type(param_pair), private, dimension(MAX_ITEM) :: input_param_list
  ! for debuging, this records counts how many time each parameter are called.
  integer, private, dimension(MAX_ITEM) :: param_use_count

  integer, private :: n_items             ! the number of items
  ! logical, private :: finished = .False.  ! parsing is finished

  integer, private, parameter :: TYPE_BLANK    = 1
  integer, private, parameter :: TYPE_WORD     = 2 ! alplhabat except 'eEdD'
  integer, private, parameter :: TYPE_ASSIGN   = 3
  integer, private, parameter :: TYPE_SIGN     = 4
  integer, private, parameter :: TYPE_DIGIT    = 5
  integer, private, parameter :: TYPE_DOT      = 6
  integer, private, parameter :: TYPE_EXPONENT = 7 ! 'eEdD'
  integer, private, parameter :: TYPE_QUOTE    = 8
  integer, private, parameter :: TYPE_END      = 9
  integer, private, parameter :: TYPE_OTHER    = 10

  ! Transition Table of DFA
  integer, private, dimension(16,10) :: dfa_transit
  ! Ehether to save a character in each state
  integer, private, dimension(16) :: state_store_char

  ! an interface for function overloading.
  interface pp_get
    module procedure param_get_int
    module procedure param_get_real
    module procedure param_get_str
  end interface

  private :: line_has_parsed, insert_param, search_param_idx
  private :: param_get_int, param_get_real, param_get_str

contains
!----------------- pp_initialize -----------------------------
  subroutine pp_initialize(param_file)!, param_arg)

    character(len=*), intent(in) :: param_file!, param_arg
    character(len=MAX_STRING_LENGTH) :: buf
    integer :: i, j, file_stat
    integer :: n_line, arg_num
    type(param_pair) :: input_param

    dfa_transit( 1,:) = (/ 1,12, 0, 0,12, 0,12, 0,11, 0 /)
    dfa_transit( 2,:) = (/ 0, 2, 0, 0, 2, 2, 2, 3, 0, 0 /)
    dfa_transit( 3,:) = (/ 7, 0, 0, 0, 0, 0, 0, 0,11, 0 /)
    dfa_transit( 4,:) = (/ 7, 0, 0, 0, 4, 0,10, 0,11, 0 /)
    dfa_transit( 5,:) = (/ 7, 0, 0, 0, 5, 0, 0, 0,11, 0 /)
    dfa_transit( 6,:) = (/ 0, 0, 0, 0, 5, 0, 0, 0, 0, 0 /)
    dfa_transit( 7,:) = (/ 7, 0, 0, 9, 8, 0, 0, 2,11, 0 /)
    dfa_transit( 8,:) = (/ 7, 0, 0, 0, 8, 4,10, 0,11, 0 /)
    dfa_transit( 9,:) = (/ 0, 0, 0, 0, 8, 0, 0, 0, 0, 0 /)
    dfa_transit(10,:) = (/ 0, 0, 0, 6, 5, 0, 0, 0, 0, 0 /)
    dfa_transit(11,:) = (/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)
    dfa_transit(12,:) = (/15,12,14, 0,12,13,12, 0, 0, 0 /)
    dfa_transit(13,:) = (/ 0,16, 0, 0,16, 0,16, 0, 0, 0 /)
    dfa_transit(14,:) = (/14, 0, 0, 9, 8, 0, 0, 2, 0, 0 /)
    dfa_transit(15,:) = (/15, 0,14, 0, 0, 0, 0, 0, 0, 0 /)
    dfa_transit(16,:) = (/15,16,14, 0,16, 0,16, 0, 0, 0 /)
    state_store_char(:) = (/ 0,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1 /)

    param_use_count(:) = 0

    ! Read parameters from input files
    open(10, file=param_file, action='read')
    n_line = 1
    do
      read(10, '(A)', iostat=file_stat) buf
      if (file_stat==iostat_end) exit
      if( line_has_parsed(buf, input_param, n_line)) then
        call insert_param(input_param)
      end if
      n_line = n_line + 1 ! line number
    end do
    close(10)

!    ! Read parameters from command arguments
!    arg_num = 2
!    do
!      call get_command_argument(arg_num, buf)
!      if ( LEN_TRIM(buf) == 0) then
!        exit
!      end if
!      !write(*,*) arg_num, trim(buf)
!      if( line_has_parsed(buf, input_param, arg_num)) then
!        call insert_param(input_param)
!      end if
!      arg_num = arg_num +1
!    end do

    write(*,'(A)') "--- Input Parameters ---"
    do i = 1, n_items
      write(*,'(A,A)', advance='no') trim(input_param_list(i)%var_name), " :"
      do j = 1, size(input_param_list(i)%values)
        write(*,'(A,A)', advance='no') " ", trim(input_param_list(i)%values(j))
      end do
      write(*,*) ""
    end do
    write(*,'(A)') "-----------------------"

  end subroutine
!---------------- line_has_parsed ------------------------
  logical function line_has_parsed(buf, input_param, n_line)

    character(len=*), intent(in) :: buf
    integer, intent(in) :: n_line
    character(len=MAX_TOKEN_LENGTH), dimension(0:MAX_TOKEN_NUM-1) :: tokens
    type(param_pair), intent(inout) :: input_param
    integer :: i, tt, cur_state, old_state, str_idx
    integer :: n_values

    tokens(:) = ""
    input_param%var_name = ""
    line_has_parsed = .True.

    cur_state = 1
    str_idx = 1
    n_values = 0
    do i = 1, len(buf) + 1
      if ( i <= len(buf)) then
        tt = get_char_type(buf(i:i))
      else
        tt = TYPE_END
      end if
      old_state = cur_state
      cur_state = dfa_transit(old_state, tt)
      !write(*,*) "tt, old, cur", tt, old_state, cur_state

      ! Syntax Errror
      if( cur_state == 0) then
        write(*,*) "  Syntax Error at: ", n_line, i
        line_has_parsed = .False.
        return
        ! Making token
      else if( state_store_char(cur_state) == 1 ) then
        tokens(n_values)(str_idx:str_idx) = buf(i:i)
        if(str_idx >= MAX_TOKEN_LENGTH) then
          write(*,*) "  Token string is too long : ", n_line, i
          line_has_parsed = .False.
          return
        end if
        str_idx = str_idx + 1
        ! Token is made
      else if ( state_store_char(cur_state) == 0 &
        .and. state_store_char(old_state) == 1) then
        if(n_values >= MAX_TOKEN_NUM) then
          write(*,*) "  Too many token values! : ", n_line, i
          line_has_parsed = .False.
          return
        end if
        n_values = n_values + 1
        str_idx = 1
      end if
      if (tt == TYPE_END) then
        exit
      end if
    end do

    if (n_values > 0) then
      input_param%var_name = tokens(0)
      if(allocated(input_param%values)) deallocate(input_param%values)
      allocate(input_param%values(n_values-1))
      do i = 1, n_values-1
        input_param%values(i) = trim(tokens(i))
      end do
    end if

    if (input_param%var_name == "") then
      line_has_parsed = .False.
    end if
  end function
!---------------- line_has_parsed ------------------------
  ! integer allocatable
  logical function pp_get_from_string_int_alloc(buf, var, n_line)
    character(len=*), intent(in) :: buf
    integer, intent(inout), allocatable :: var(:)
    integer, intent(in) :: n_line
    type(param_pair) :: input_param

    pp_get_from_string_int_alloc = .False.
    if( line_has_parsed(buf, input_param, n_line)) then
      allocate(var(size(input_param%values)))
      read (input_param%values, *) var
      pp_get_from_string_int_alloc = .True.
    end if
    if(allocated(input_param%values)) deallocate(input_param%values)
  end function

  ! real
  logical function pp_get_from_string_real(buf, var, idx, n_line)
    character(len=*), intent(in) :: buf
    real(kind=8), intent(inout) :: var
    integer, intent(in) :: idx, n_line
    type(param_pair) :: input_param

    pp_get_from_string_real = .False.
    if( line_has_parsed(buf, input_param, n_line)) then
      if(size(input_param%values) >= idx) then
        read (input_param%values(idx), *) var
        pp_get_from_string_real = .True.
      end if
    end if
  end function

!---------------- get_char_type ------------------------
! character comparison is conducted based on ASCII code
  integer function get_char_type(ch)
    character, intent(in) :: ch

    if(ch == ' ' .or. ch == '	') then ! space and Tab
      get_char_type = TYPE_BLANK
    else if (ch == '-' .or. ch == '+') then ! plus minus sign
      get_char_type = TYPE_SIGN
    else if (ch == 'e' .or. ch == 'E' .or. & ! exponent
      ch == 'd' .or. ch == 'D') then
      get_char_type = TYPE_EXPONENT
    else if (ch == '_' .or. & ! alpha, underscore except 'e','E','c','D'
      ('A' <= ch .and. ch <= 'Z') .or. &
      ('a' <= ch .and. ch <= 'z')) then
      get_char_type = TYPE_WORD
    else if ('0' <= ch .and. ch <= '9' .or. ch == '-' ) then ! digit
      get_char_type = TYPE_DIGIT
    else if (ch == '.' ) then ! dot
      get_char_type = TYPE_DOT
    else if ( ch == '"') then ! double quote
      get_char_type = TYPE_QUOTE
    else if ( ch =='=' .or.  ch ==':') then ! assign
      get_char_type = TYPE_ASSIGN
    else if ( ch =='#') then ! comment #
      get_char_type = TYPE_END
    else ! invalid character
      get_char_type = TYPE_OTHER
    end if
  end function
!--------------------------------------------------------
  subroutine insert_param(new_param)
    type(param_pair), intent(in) :: new_param
    integer :: i

    do i = 1, n_items
      if( new_param%var_name == input_param_list(i)%var_name) then
        write(*,*) "Warning! '",trim(new_param%var_name), &
          "' is overwritten due to duplicated input parameter."
        if (allocated(input_param_list(i)%values)) then
          deallocate(input_param_list(i)%values)
        end if
        input_param_list(i) = new_param
        return
      end if
    end do

    if (n_items >= MAX_ITEM) then
      write(*,*) "Too many input lines"
      stop (1)
    end if
    n_items = n_items + 1
    input_param_list(n_items) = new_param
  end subroutine
!-------------------------------------------------------------
  integer function search_param_idx(str_name, idx)
    character(len=*), intent(in) :: str_name
    integer :: i, idx_count, idx
    idx_count = 0
    do i = 1, n_items
      if(str_name == input_param_list(i)%var_name) then
        if ( idx <= size(input_param_list(i)%values) ) then
          ! Update count. To prevent overflow min is used.
          param_use_count(i) = min(param_use_count(i) + 1, 100000)
          search_param_idx = i
        else
          search_param_idx = -1
        end if
        return
      end if
    end do
    search_param_idx = -1
  end function
!--------------------------------------------------------
  logical function param_get_int(str_name, var, idx_in)
    character(len=*), intent(in) :: str_name
    integer, intent(inout) :: var
    integer, optional :: idx_in
    integer :: idx, loc
    if(present(idx_in)) then
      idx = idx_in
    else
      idx = 1
    end if

    loc = search_param_idx(str_name, idx)
    if( loc > 0) then
      read (input_param_list(loc)%values(idx), *) var
      param_get_int = .True.
    else
      param_get_int = .False.
    end if
  end function
!---------------------------------------------------------------
  logical function param_get_real(str_name, var, idx_in)
    character(len=*), intent(in) :: str_name
    real(kind=8), intent(inout) :: var
    integer, optional :: idx_in
    integer :: idx, loc
    if(present(idx_in)) then
      idx = idx_in
    else
      idx = 1
    end if

    loc = search_param_idx(str_name, idx)
    if( loc > 0) then
      read (input_param_list(loc)%values(idx), *) var
      param_get_real = .True.
    else
      param_get_real = .False.
    end if
  end function
!--------------------------------------------------------------
  logical function param_get_str(str_name, var, idx_in)
    character(len=*), intent(in) :: str_name
    character(len=*), intent(inout) :: var
    integer, optional :: idx_in
    integer :: idx, loc
    if(present(idx_in)) then
      idx = idx_in
    else
      idx = 1
    end if

    loc = search_param_idx(str_name, idx)
    if( loc > 0) then
      read (input_param_list(loc)%values(idx), *) var
      param_get_str = .True.
    else
      param_get_str = .False.
    end if
  end function
!----------------- pp_finalize -----------------------------
  subroutine pp_finalize()
    integer :: i
    ! show parameters that are never used.
    write(*,'(A)') "Parameters that are never used: "
    do i = 1, n_items
      if(param_use_count(i) == 0) then
        write(*,*) trim(input_param_list(i)%var_name)
      end if
    end do

    do i = 1, n_items
      if (allocated(input_param_list(i)%values)) then
        deallocate(input_param_list(i)%values)
      end if
    end do

  end subroutine
end module param_parser
!----------------------------------------------------------------------

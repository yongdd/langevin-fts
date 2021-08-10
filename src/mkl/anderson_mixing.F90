 !--------------------------------------------------------------------
 ! Anderson mixing module
module anderson_mixing

  use param_parser
  use circular_buffer_module
  use simulation_box
  implicit none

  ! a few previous field values are stored for anderson mixing
  type(circular_buffer), private :: cb_wout_hist, cb_wdiff_hist
  ! arrays to calculate anderson mixing
  type(circular_buffer), private :: cb_wdiffdots
  real(kind=8), private, allocatable :: u_nm(:,:), v_n(:), a_n(:), wdiffdots(:)

  ! anderson mixing related parameters
  integer, private :: n_anderson, max_anderson
  integer, private :: num_components
  integer, private :: total_grids
  real(kind=8), private :: start_anderson_error
  real(kind=8), private :: mix, mix_min, mix_init

  private :: find_an

contains
  !-------------- am_initialize ---------------
  subroutine am_initialize(in_num_components, new_process_idx)
    integer, intent(in) :: in_num_components
!   visible device id
!   dummy variable for consistent with CUDA pseudo
    integer, optional, intent(in)  :: new_process_idx

    total_grids = in_num_components*totalMM
    num_components = in_num_components

    ! anderson mixing begin if error level becomes less then start_anderson_error
    if(.not. pp_get("am.start_error", start_anderson_error)) start_anderson_error = 0.01d0
    ! max number of previous steps to calculate new field when using Anderson mixing
    if(.not. pp_get("am.step_max", max_anderson)) max_anderson = 10
    ! minimum mixing parameter
    if(.not. pp_get("am.mix_min", mix_min)) mix_min = 0.1d0
    ! initialize mixing parameter
    if(.not. pp_get("am.mix_init", mix_init)) mix_init  = 0.01d0
    mix = mix_init
    ! number of anderson mixing steps, increases from 0 to max_anderson
    n_anderson = -1

    call cbInit(cb_wout_hist,  max_anderson+1, total_grids)
    call cbInit(cb_wdiff_hist, max_anderson+1, total_grids)
    call cbInit(cb_wdiffdots,  max_anderson+1, max_anderson+1)
    call am_reset_count()

    ! define arrays for anderson mixing
    allocate(u_nm(1:max_anderson,1:max_anderson))
    allocate(v_n(1:max_anderson),a_n(1:max_anderson))
    allocate(wdiffdots(0:max_anderson))

  end subroutine
  !---------------- am_reset_count -----------------
  subroutine am_reset_count()
    ! initialize mixing parameter
    mix = mix_init
    ! number of anderson mixing steps, increases from 0 to max_anderson
    n_anderson = -1

    call cbReset(cb_wout_hist)
    call cbReset(cb_wdiff_hist)
    call cbReset(cb_wdiffdots)

  end subroutine
  !---------------- am_caculate_new_fields ---------------
  subroutine am_caculate_new_fields(w, w_out, old_error_level, error_level)
    real(kind=8), intent(inout) :: w(1:total_grids)
    real(kind=8), target :: w_diff(1:total_grids)
    real(kind=8), intent(in) :: w_out(1:total_grids)
    real(kind=8), intent(in) :: old_error_level, error_level
    type(real_array_pt) :: arr_pt1, arr_pt2
    integer :: nit, nit2
    
    ! condition to start anderson mixing
    if(error_level < start_anderson_error .or. n_anderson >= 0) then
      n_anderson = n_anderson + 1
    end if

    !if( iter == start_anderson) write(*,*) iter
    if( n_anderson >= 0 ) then
      ! number of histories to use for anderson mixing
      n_anderson = min(max_anderson, n_anderson)
      ! store the input and output field (the memory is used in a periodic way)
      call cbInsert(cb_wout_hist,  w_out)
      call cbInsert(cb_wdiff_hist, w_out-w)
      ! evaluate wdiff dot products for calculating Unm and Vn in Thompson's paper
      wdiffdots = 0.0d0
      w_diff = w_out-w
      do nit=0,n_anderson
        arr_pt2 = cbGetArrayPt(cb_wdiff_hist, n_anderson-nit)
        wdiffdots(nit) = multidot(num_components, w_diff, arr_pt2%p)
      end do
      call cbInsert(cb_wdiffdots, wdiffdots)
    end if

    ! conditions to apply the simple mixing method
    if( n_anderson <= 0 ) then
      ! dynamically change mixing parameter
      if(old_error_level < error_level) then
        mix = max(mix*0.7d0,mix_min)
      else
        mix = mix*1.01d0
      end if
      ! make a simple mixing of input and output fields for the next iteration
      w = (1.0d0-mix)*w + mix*w_out
    else
      ! calculate Unm and Vn
      do nit=1, n_anderson
        v_n(nit) = cbGetSym(cb_wdiffdots, n_anderson, n_anderson) &
          - cbGetSym(cb_wdiffdots, n_anderson, n_anderson - nit)
        do nit2=1, n_anderson
          u_nm(nit,nit2) = cbGetSym(cb_wdiffdots, n_anderson, n_anderson)&
            - cbGetSym(cb_wdiffdots, n_anderson, n_anderson-nit) &
            - cbGetSym(cb_wdiffdots, n_anderson-nit2, n_anderson) &
            + cbGetSym(cb_wdiffdots, n_anderson-nit, n_anderson-nit2)
        end do
      end do
      !write(*,*) "u_nm", u_nm
      !write(*,*) "v_n", v_n
      !write(*,*) "a_n", a_n

      call find_an(u_nm(1:n_anderson,1:n_anderson),v_n(1:n_anderson),a_n(1:n_anderson),n_anderson)

      ! write anderson mixing coefficient
      !write(*,'(A, F12.6)',advance="no") "{", a_n(1)
      !do nit=2,n_anderson
      !  write(*,'(A, F12.6)',advance="no") ",", a_n(nit)
      !end do
      !write(*, "(A)") "},"

      !  calculate the new field
      arr_pt1 = cbGetArrayPt(cb_wout_hist, n_anderson)
      w = arr_pt1%p
      do nit=1, n_anderson
        arr_pt2 = cbGetArrayPt(cb_wout_hist, n_anderson-nit)
        w = w + a_n(nit)*(arr_pt2%p -arr_pt1%p)
      end do
    end if

  end subroutine
  !-------------- anderson_mixing_finalize ---------------
  subroutine am_finalize
    ! free memory
    call cbFree(cb_wout_hist)
    call cbFree(cb_wdiff_hist)
    call cbFree(cb_wdiffdots)

    deallocate(wdiffdots)
    deallocate(u_nm, v_n, a_n)

  end subroutine
  !--------------------------------------------------------------------
  ! finds the solution for the matrix equation (n*n size) for anderson mixing
  ! U * A = V, Gauss elimination method is used in its simplest level
  subroutine find_an(u,v,a,n)
    integer, intent(in) :: n
    real(kind=8), intent(inout) :: u(1:n,1:n), v(1:n)
    real(kind=8), intent(out) :: a(1:n)
    !
    integer :: i,j,k
    real(kind=8) :: factor, tempsum
    !   elimination process
    do i=1,n
      do j=i+1,n
        factor = u(j,i)/u(i,i)
        v(j) = v(j) - v(i)*factor
        do k=i+1,n
          u(j,k) = u(j,k) - u(i,k)*factor
        end do
      end do
    end do

    !   find the solution
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

end module anderson_mixing

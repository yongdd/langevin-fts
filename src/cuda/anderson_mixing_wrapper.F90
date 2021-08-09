!--------------------------------------------------------------------
! this is a wrapper module for cuda version of Anderson mixing module
module anderson_mixing
!
  use constants, only : rp
  use param_parser
  use simulation_box
  implicit none

  integer, private :: num_components
  integer, private :: total_grids

contains
!-------------- am_initialize ---------------
  subroutine am_initialize(in_num_components, new_process_idx)
    integer, intent(in) :: in_num_components
    integer, optional, intent(in)  :: new_process_idx
    integer :: n, process_idx
    integer :: max_anderson
    integer :: num_block, threads_per_block
    real(kind=rp) :: start_anderson_error, mix_min, mix_init

    if( PRESENT(new_process_idx)) then
        process_idx = new_process_idx
    else
        process_idx = 0
    end if

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

    if(.not. pp_get("gpu.num_block", num_block)) num_block = 512
    if(.not. pp_get("gpu.threads_per_block", threads_per_block)) threads_per_block = 256

    call am_cuda_initialize(num_components, totalMM, seg, &
      max_anderson, start_anderson_error, &
      mix_min, mix_init, &
      num_block, threads_per_block, process_idx)
    call am_cuda_reset_count()

  end subroutine

!---------------- am_reset_count -----------------
  subroutine am_reset_count()
    call am_cuda_reset_count()
  end subroutine
!---------------- am_caculate_new_fields ---------------
  subroutine am_caculate_new_fields(w, w_out, old_error_level, error_level)
    real(kind=rp), intent(inout) :: w(1:total_grids)
    real(kind=rp), intent(in)    :: w_out(1:total_grids)
    real(kind=dp), intent(in)    :: old_error_level, error_level

    call am_cuda_caculate_new_fields(w, w_out, w_out-w, old_error_level, error_level)

  end subroutine
!-------------- anderson_mixing_finalize ---------------
  subroutine am_finalize
    call am_cuda_finalize()
  end subroutine
end module anderson_mixing

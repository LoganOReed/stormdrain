subroutine lapack_sort(a) 
  implicit none
  ! integer :: n 
  real(8),dimension(6),intent(inout) :: a
  integer :: info       ! Output flag from LAPACK
  !f2py optional :: info
  call dlasrt('I', 6, a, info)
end subroutine lapack_sort

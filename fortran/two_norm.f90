subroutine two_norm(a,c,n) 
  implicit none
  real(8),dimension(n),intent(in) :: a
  real(8),intent(out) :: c 
  integer :: n
  !f2py optional , depend(a) :: n=len(a)
  real(8), external :: dnrm2
  c = dnrm2(n, a, 1)
end subroutine two_norm

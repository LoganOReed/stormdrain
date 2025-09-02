subroutine euclidian_norm(a,c,n) 
  integer :: n 
  real(8),dimension(n),intent(in) :: a
  !f2py optional , depend(a) :: n=len(a)
  real(8),intent(out) :: c 
  real(8) :: sommec 
  integer :: i
  sommec = 0 
  do i=1,n
    sommec=sommec+a( i )*a( i ) 
  end do
  c = sqrt (sommec) 
end subroutine euclidian_norm

PROGRAM Ising_model
USE random_generator
IMPLICIT NONE	

DOUBLE PRECISION :: DE, DE2, sigma, temperature, r, tt
INTEGER :: mcs, i, j, k, N, temp, seed, ii, jj, mm
integer :: ddt(8)
DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: Ising, Ising_temp
N = 20
seed = (ddt(8)-500)*54321 + 11223344
allocate(Ising(0:N+1,0:N+1), Ising_temp(0:N+1,0:N+1))
open(unit=10,file='PCA.txt')
do tt = 1.4, 2.9, 0.1
print*, tt
MCS = 10000000
DE = 0.0d0
sigma = 0
	do mm = 1, 10	
		print*, mm
		Ising = 1
		do k = 1, MCS
			Ising_temp = Ising
			i = floor(ran2(seed)*0.9999999d0*(N)) + 1
			j = floor(ran2(seed)*0.9999999d0*(N)) + 1
			Ising_temp(i,j) = -Ising(i,j)
!			print*, "111"
!boundary conditions
			do ii = 1,N
				Ising_temp(ii,0) = Ising_temp(ii,N) 
				Ising_temp(ii,N+1) = Ising_temp(ii,1)
				Ising_temp(0,ii) = Ising_temp(N,ii)
				Ising_temp(N+1,ii) = Ising_temp(1,ii)
			end do
			Ising_temp(0,0) = Ising_temp(N,N)
			Ising_temp(N+1,N+1) = Ising_temp(1,1)
			Ising_temp(0,N+1) = Ising_temp(N,1)
			Ising_temp(N+1,0) = Ising_temp(1,N)
!			print*, "222"
!get deltaE
			DE = - (Ising_temp(i,j)*Ising_temp(i+1,j) + Ising_temp(i,j)*Ising_temp(i-1,j)  &
			   	   +Ising_temp(i,j)*Ising_temp(i,j+1) + Ising_temp(i,j)*Ising_temp(i,j-1))  &
			 	 + (Ising(i,j)*Ising(i+1,j) + Ising(i,j)*Ising(i-1,j)  &
			   	   +Ising(i,j)*Ising(i,j+1) + Ising(i,j)*Ising(i,j-1)) 

!			print*, "333"
			r = ran2(seed)
			if (r < dexp(-(DE)/tt)) then
				Ising = Ising_temp
			end if
		end do
!		print*, "444"
		do i = 1, N
			do j = 1,N
				write(10,*) Ising(i,j)
				sigma = sigma + Ising(i,j)
			end do
		end do
	end do
	print*, "sigma = ", sigma/mm/N/N
end do
close(10)


END PROGRAM Ising_model

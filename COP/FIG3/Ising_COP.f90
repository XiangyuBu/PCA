PROGRAM Ising_model
USE random_generator
IMPLICIT NONE	

DOUBLE PRECISION :: DE, DE2, sigma, temperature, r, tt, sigma_temp
INTEGER :: mcs, i, j, k, N, temp, seed, ii, jj, mm, a, b
integer :: ddt(8)
DOUBLE PRECISION, DIMENSION(:,:), ALLOCATABLE :: Ising, Ising_temp
N = 20
seed = (ddt(8)-500)*54321 + 11223344
allocate(Ising(0:N+1,0:N+1), Ising_temp(0:N+1,0:N+1))
open(unit=10,file='PCA.txt')
open(unit=11,file='origin.txt')
!do tt = 1.4, 2.9, 0.1

do tt = 1.8, 1.8
print*, tt
MCS = 50000000
DE = 0.0d0
sigma = 0
	do mm = 1, 1	
		print*, mm
		do i = 1, N
			do j = 1, N
				if (mod(j,2)==0) then
					Ising(i,j) = -1
				else
					Ising(i,j) = 1
				end if
			end do
		end do
		
		sigma_temp = 0
		do i = 1, N
			do j = 1,N
				sigma_temp = sigma_temp + Ising(i,j)
				write(11,*) i, j, Ising(i,j)
			end do
		end do
		print*, "sigma_temp = ", sigma_temp/N/N
		close(11)


		do k = 1, MCS
			Ising_temp = Ising
			i = floor(ran2(seed)*0.9999999d0*(N)) + 1
			j = floor(ran2(seed)*0.9999999d0*(N)) + 1
			a = i
			b = j
			Ising_temp(i,j) = - Ising(i,j)
			do while (a==i .and. b==j)
				a = floor(ran2(seed)*0.9999999d0*(N)) + 1
				b = floor(ran2(seed)*0.9999999d0*(N)) + 1
				do while (Ising(a,b)==Ising(i,j))
					a = floor(ran2(seed)*0.9999999d0*(N)) + 1
					b = floor(ran2(seed)*0.9999999d0*(N)) + 1	
				end do
			end do
!			print*, i, j ,a ,b
			Ising_temp(a,b) = - Ising(a,b)
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
			   	   +Ising_temp(i,j)*Ising_temp(i,j+1) + Ising_temp(i,j)*Ising_temp(i,j-1)  &
			   	   +Ising_temp(a,b)*Ising_temp(a+1,b) + Ising_temp(a,b)*Ising_temp(a-1,b)  &
			   	   +Ising_temp(a,b)*Ising_temp(a,b+1) + Ising_temp(a,b)*Ising_temp(a,b-1))  &
			 	 + (Ising(i,j)*Ising(i+1,j) + Ising(i,j)*Ising(i-1,j)  &
			   	   +Ising(i,j)*Ising(i,j+1) + Ising(i,j)*Ising(i,j-1)  &
			   	   +Ising(a,b)*Ising(a+1,b) + Ising(a,b)*Ising(a-1,b)  &
			   	   +Ising(a,b)*Ising(a,b+1) + Ising(a,b)*Ising(a,b-1)) 

!			print*, "333"
			r = ran2(seed)
			if (r < dexp(-(DE)/tt)) then
				Ising = Ising_temp
			end if
			
		end do
!		print*, "444"
		do i = 1, N
			do j = 1,N
				write(10,*) i, j, Ising(i,j)
				sigma = sigma + Ising(i,j)
			end do
		end do
	end do
	print*, "sigma = ", sigma/mm/N/N
end do
close(10)


END PROGRAM Ising_model

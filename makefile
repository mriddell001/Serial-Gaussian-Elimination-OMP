sge:
	icc -O3 -qopenmp -fast -align *.c -Wall -W -Werror
clean:
	rm -f sge sge.exe

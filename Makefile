all: fftw.main dft.main pft.main fftw.bench dft.bench pft.bench pft.test

fftw.main: fftw.main.c funcs.c
	gcc -o $@ $^ -lfftw3 -lm

dft.main: dft.main.c dft.c funcs.c
	gcc -o $@ $^ -lm

pft.main: pft.main.c pft.c funcs.c
	mpicc -o $@ $^ -lm

fftw.bench: fftw.bench.c funcs.c
	mpicc -o $@ $^ -lfftw3 -lm

dft.bench: dft.bench.c dft.c funcs.c
	mpicc -o $@ $^ -lm

pft.bench: pft.bench.c pft.c funcs.c
	mpicc -o $@ $^ -lm

pft.test: pft.test.c pft.c funcs.c
	mpicc -o $@ $^ -lfftw3 -lm

.PHONY: clean
clean:
	@echo rm *.bench *.main *.test
	@rm *.bench *.main *.test 2>/dev/null; true

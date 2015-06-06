
all: clean
	/usr/bin/env python2 setup.py build_ext --inplace
	cd ..; ln -fs cython/RobustQuadraticSolverCython.so .; cd cython

clean:
	rm -f *.so

#! /usr/bin/env python2

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension("*",
        sources=["RobustQuadraticSolverCython.pyx",
                 "armadillosolver.cpp",],
        #include_dirs = [numpy.get_include(),],
        #extra_compile_args=["-O3", "-std=c++11", "-march=native"],
        #libraries = ['armadillo',], 
        #library_dirs = ['../../extern/armadillo/',],
        include_dirs = [numpy.get_include(), '../../extern/armadillo/include'],
        #extra_compile_args=["-O3", "-std=c++11", "-march=native", "-DARMA_DONT_USE_WRAPPER"],
        extra_compile_args=["-O3", "-march=native", "-DARMA_DONT_USE_WRAPPER"],
        libraries = ['openblas','gfortran'],
        language="c++",)))

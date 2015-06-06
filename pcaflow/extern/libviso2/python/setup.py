#! /usr/bin/env python2

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize(Extension("*",
        sources=["libvisomatcher.pyx",
                 "../src/matcher.cpp",
                 "../src/filter.cpp",
                 "../src/triangle.cpp",
                 "../src/matrix.cpp"],
        include_dirs = ['../src',numpy.get_include()],
        extra_compile_args=["-fPIC", "-march=native","-O3"],
        language="c++",)))

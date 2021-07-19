# To make TensorMol available.
# sudo pip install -e .
# pip install --user -e .
#
# to make and upload a source dist 
# python setup.py sdist
# twine upload dist/*
# And of course also be me. 
# 

from __future__ import absolute_import,print_function
from distutils.core import setup, Extension
import numpy
import os

print("Numpy Include Dir: ",numpy.get_include())

LLVM=os.popen('gcc --version | grep clang').read().count("LLVM")
#if (not LLVM):
#MolEmb = Extension(
#'MolEmb',
#sources=['./C_API/MolEmb.cpp'],
#       # kan extra_link_args=[],
#kan extra_compile_args=['-std=c++0x','-g','-fopenmp','-w'],
#extra_compile_args=['-std=c++0x,-L/p/home/apps/intel/compilers/15.0.3.187/composer_xe_2015.0.3.187/compiler/lib/intel64/lsvml'],
#extra_link_args=['-lpthread'],
##kan extra_link_args=['-lgomp'],
#       include_dirs=[numpy.get_include()]+['./C_API/'])
#lse:
MolEmb = Extension(
'MolEmb',
sources=['./C_API/MolEmb.cpp'],
extra_compile_args=['-std=c++0x'],
include_dirs=[numpy.get_include()]+['./C_API/'])
#-shared-libgcc
#-static-libgcc
#--static-libgcc
# warn unresolved references
#-symbolic
#-u symbol
#-u svml_exp2


# run the setup
setup(name='TensorMol',
      version='0.2',
      description='TensorFlow+Molecules = TensorMol',
      url='http://github.com/jparkhill/TensorMol',
      author='john parkhill',
      author_email='john.parkhill@gmail.com',
      license='GPL3',
      packages=['TensorMol'],
      zip_safe=False,
      include_package_data=True,
      ext_modules=[MolEmb])

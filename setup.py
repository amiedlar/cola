import os
import numpy
from setuptools import setup, Extension
import setuptools
from Cython.Build import cythonize
# from sklearn._build_utils import get_blas_info

# cblas_libs, blas_info = get_blas_info()

# if os.name == 'posix':
#     cblas_libs.append('m')

include_dirs = [numpy.get_include()]# + blas_info.pop('include_dirs', [])

ext_modules = [
    Extension("fast_cd.svm", sources=["fast_cd/svm.pyx"], include_dirs=include_dirs),
    Extension("fast_cd.elasticnet", sources=["fast_cd/elasticnet.pyx"], include_dirs=include_dirs)
]

setup(
    name="CoLA",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': 3}),
    # include_dirs=include_dirs,
    packages=[
        'cola',
        'fast_cd'],
    entry_points='''
        [console_scripts]
        preparedata=preparedata:cli
        run_cola=run_cola:main
    ''')

# -*- coding: utf-8 -*-
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext


ext_modules = [Extension('ridge.racer.gradient_steps', ['ridge/racer/gradient_steps.pyx'])]

setup(
    name='ridge',
    version='0.0.2',
    description='Python Machine Learning Library specialized in L2R and Recommendation.',
    author='Moriaki Saigusa',
    author_email='moriaki3193@gmail.com',
    install_requires=[],
    url='https://github.com/moriaki3193/ridge',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)

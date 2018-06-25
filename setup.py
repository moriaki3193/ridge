# -*- coding: utf-8 -*-
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext


path2src = 'ridge/racer/src/'
ext_modules = [
    Extension('ridge.racer.gradient_steps', [path2src + 'gradient_steps.pyx']),
    Extension('ridge.racer.link_functions', [path2src + 'link_functions.pyx']),
    Extension('ridge.racer.loss_calculators', [path2src + 'loss_calculators.pyx']),
    Extension('ridge.racer.predictors', [path2src + 'predictors.pyx']),
]

setup(
    name='ridge',
    version='0.0.4',
    description='Python Machine Learning Library specialized in L2R and Recommendation.',
    author='Moriaki Saigusa',
    author_email='moriaki3193@gmail.com',
    url='https://github.com/moriaki3193/ridge',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
    ],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    zip_safe=False,
)

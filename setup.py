#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import subprocess
import pip
import sys

REQUIREMENTS = [
    'pandas',
    'numpy',
    'pytest',
    'scipy',
    'numba',
    'tqdm',
    'matplotlib',
    'toolz',
]


setup(
    name='pymssa',
    version='0.1.0',
    description="Multivariate Singular Spectrum Analysis (MSSA)",
    author="Kiefer Katovich",
    author_email='kiefer.katovich@gmail.com',
    url='https://github.com/kieferk/pymssa.git',
    packages=find_packages(),
    package_dir={'pymssa':'pymssa'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords='Python Multivariate Singular Spectrum Analysis MSSA',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

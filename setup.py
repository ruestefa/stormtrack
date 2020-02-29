#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The setup script.
"""
# Third-party
import numpy
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext

# Import Cython AFTER setuptools
from Cython.Build import cythonize  # isort: skip
from Cython import Compiler  # isort:skip

def read_file(path):
    with open(path, "r") as f:
        return "\n".join([l.strip() for l in f.readlines()])

description_files = ["README.rst", "HISTORY.rst"]

metadata = {
    "name": "stormtrack",
    "version": "0.2.0",
    "description": "Track two-dimensional features over time in high-resolution weather/climate data.",
    "long_description": "\n\n".join([read_file(f) for f in description_files]),
    "author": "Stefan Ruedisuehli",
    "author_email": "stefan.ruedisuehli@env.ethz.ch",
    "url": "https://github.com/ruestefa/stormtrack",
    "keywords": "stormtrack",
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Cython",
    ],
}

python = ">=3.7"

dependencies = [
    "cython",
    "click >= 6.0",
    "h5py",
    "python-igraph",
    "netcdf4",
    "numpy",
    "scipy",
    "shapely",
    "pillow",
    "pytz",
    "pyproj",
]

scripts = [
    "identify-features=stormtrack.identify_features:pre_main",
    "track-features=stormtrack.track_features:pre_main",
]

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Compiler.Options.annotate = True

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
compiler_directives={"embedsignature": True}

cython_setup = {
    "ext_modules": cythonize("src/**/*.pyx"),
    "cmdclass": {"build_ext": build_ext},
    "include_dirs": [numpy.get_include()],
    "compiler_directives": compiler_directives,
}

setup(
    python_requires=python,
    install_requires=dependencies,
    entry_points={"console_scripts": scripts},
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    **cython_setup,
    **metadata,
)

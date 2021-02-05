#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The setup script.
"""
# Third-party
import numpy
from numpy import f2py
from setuptools.extension import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext

# Import Cython AFTER setuptools
from Cython.Build import cythonize  # isort: skip
from Cython import Compiler  # isort:skip

def read_file(path):
    with open(path, "r") as f:
        return "\n".join([l.strip() for l in f.readlines()])

description_files = ["README.md", "HISTORY.md"]

metadata = {
    "name": "stormtrack",
    "version": "0.4.3",
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

python = "==3.7"

dependencies = [
    "basemap @ git+https://github.com/matplotlib/basemap.git",
    "cython",
    "click >= 6.0",
    "descartes",
    "h5py",
    "python-igraph",
    "netcdf4",
    "numpy",
    "matplotlib",
    "scipy",
    "shapely",
    "pillow",
    "pytz",
    "pyproj",
]

scripts = [
    # Main
    "identify-features=stormtrack.identify_features:cli",
    "identify-front-fields=stormtrack.identify_front_fields:cli",
    "track-features=stormtrack.track_features:cli",
    # Additional
    "group-tracks=stormtrack.scripts.group_tracks:cli",
    "inspect-tracks=stormtrack.scripts.inspect_tracks:cli",
    "project-tracks=stormtrack.scripts.project_tracks:cli",
]

# Compile FORTRAN files
with open("src/stormtrack/extra/fronts/_libfronts.f77", "rb") as f:
    code = f.read()
stat = f2py.compile(code, modulename="src/stormtrack.extra.fronts._libfronts")
if stat != 0:
    raise Exception("f2py failed", stat)

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Compiler.Options.annotate = True

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
compiler_directives={"embedsignature": True}

extensions = [
    # Extension("*", ["src/**/*.pyx"], extra_compile_args=["-O0"]),
    Extension("*", ["src/**/*.pyx"], extra_compile_args=["-O3"]),
]

cython_setup = {
    "ext_modules": cythonize(extensions, compiler_directives={"language_level": 3}),
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The setup script.
"""

from setuptools import setup
from setuptools import find_packages

# Import Cython after setuptools
from Cython.Build import cythonize  # isort:skip

def read_file(path):
    with open(path, "r") as f:
        return "\n".join([l.strip() for l in f.readlines()])

description_files = ["README.rst", "HISTORY.rst"]

metadata = {
    "name": "stormtrack",
    "version": "0.1.0",
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
    "Click >= 6.0",
]

scripts = [
    "stormtrack=stormtrack.cli:main",
]

setup(
    python_requires=python,
    install_requires=dependencies,
    entry_points={"console_scripts": scripts},
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=cythonize("src/*/*.pyx", annotate=True),
    include_package_data=True,
    **metadata,
)

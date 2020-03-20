# -*- coding: utf-8 -*-
"""
Top-level package for StormTrack.
"""

__author__ = """Stefan Ruedisuehli"""
__email__ = "stefan.ruedisuehli@env.ethz.ch"
__version__ = "0.2.0"

import matplotlib as mpl  # isort:skip

mpl.use("Agg")

# Import netCDF4 BEFORE h5py to prevent error
import netCDF4  # isort:skip
import h5py  # isort:skip

# SR_DEV < Shorthand to embed IPython shell
try:
    import IPython  # isort:skip
except ImportError:
    ipy = None
else:
    ipy = IPython.terminal.embed.embed
# SR_DEV >

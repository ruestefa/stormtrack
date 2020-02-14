#!/usr/bin/env python3

from __future__ import print_function

cimport cython
cimport numpy as np
from libc.math cimport M_PI
from libc.math cimport atan2
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt

#------------------------------------------------------------------------------

import numpy as np

try:
    from .various import ipython
except ImportError:
    pass

#==============================================================================
# Trim mask to content
#==============================================================================

def trim_mask_grid(mask, grid, pad):

    # Copy in order not to change the input dict
    grid = {k: v for k, v in grid.items()}

    mask, (nx0, nx1, ny0, ny1) = trim_mask2d(mask, pad=pad)

    grid["nlat"]   -= nx0 + nx1
    grid["nlon"]   -= ny0 + ny1
    grid["latmin"] += nx0*grid["dlat"]
    grid["latmax"] -= nx1*grid["dlat"]
    grid["lonmin"] += ny0*grid["dlon"]
    grid["lonmax"] -= ny1*grid["dlon"]

    return mask, grid

def trim_mask2d(arr, pad=0):
    """Trim a 2D mask, optionally with padding around True area.

    Return the trimmed mask and the number of trimmed points in all directions.
    """

    proj_x = arr.max(axis=1)
    proj_y = arr.max(axis=0)

    nx0 = max(0, proj_x.tolist()[:   ].index(True) - pad)
    nx1 = max(0, proj_x.tolist()[::-1].index(True) - pad)
    ny0 = max(0, proj_y.tolist()[:   ].index(True) - pad)
    ny1 = max(0, proj_y.tolist()[::-1].index(True) - pad)

    nx, ny = arr.shape
    arr = arr[nx0:nx-nx1, ny0:ny-ny1]

    return arr, (nx0, nx1, ny0, ny1)

#==============================================================================


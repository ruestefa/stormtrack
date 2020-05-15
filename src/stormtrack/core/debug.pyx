# -*- coding: utf-8 -*-
"""
Debugging utilities.
"""

from __future__ import print_function

# C: Third-party
cimport numpy as np

# Third-party
import numpy as np


cdef void cregion_dump(cRegion* cregion, str path) except *:
    """Write the pixels of a cregion object to a file."""

    # Collect pixels
    cdef list pixels = []
    cdef int i
    cdef cPixel* cpixel
    for i in range(cregion.pixels_istart, cregion.pixels_iend):
        cpixel = cregion.pixels[i]
        pixels.append((cpixel.x, cpixel.y))

    # Dump pixels
    if path.endswith(".py"):
        with open(path, "w") as f:
            f.write("pixels = [\n")
            for x, y in pixels:
                f.write(f"    ({x}, {y}),\n")
            f.write("]\n")
    else:
        raise NotImplementedError("unknown file type", path)

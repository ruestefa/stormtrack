# !/usr/bin/env python3

# Third-party
import numpy as np


def trim_mask_grid(mask, grid, pad):
    """Trim mask to content."""

    # Copy in order not to change the input dict
    grid = {k: v for k, v in grid.items()}

    mask, (nx0, nx1, ny0, ny1) = trim_mask2d(mask, pad=pad)

    grid["nlat"] -= nx0 + nx1
    grid["nlon"] -= ny0 + ny1
    grid["latmin"] += nx0 * grid["dlat"]
    grid["latmax"] -= nx1 * grid["dlat"]
    grid["lonmin"] += ny0 * grid["dlon"]
    grid["lonmax"] -= ny1 * grid["dlon"]

    return mask, grid


def trim_mask2d(arr, pad=0):
    """Trim a 2D mask, optionally with padding around True area.

    Return the trimmed mask and the number of trimmed points in all directions.
    """

    proj_x = arr.max(axis=1)
    proj_y = arr.max(axis=0)

    nx0 = max(0, proj_x.tolist()[:].index(True) - pad)
    nx1 = max(0, proj_x.tolist()[::-1].index(True) - pad)
    ny0 = max(0, proj_y.tolist()[:].index(True) - pad)
    ny1 = max(0, proj_y.tolist()[::-1].index(True) - pad)

    nx, ny = arr.shape
    arr = arr[nx0 : nx - nx1, ny0 : ny - ny1]

    return arr, (nx0, nx1, ny0, ny1)


def reduce_grid_resolution(fld, stride, mode):
    """Reduce grid resolution by striding."""

    # Check stride
    if (stride - 1)%2 > 0:
        raise ValueError("stride must be an uneven number", stride)

    # Check and codify the reduction mode
    modes = ["pick", "mean"]
    try:
        imode = modes.index(mode)
    except ValueError:
        raise ValueError(f"mode must be one of {modes}", mode)

    if len(fld.shape) == 1:

        # Determine grid sizes (in and out)
        nxi, = fld.shape
        nxo = int(nxi / stride)

        # Initialize output field
        fld_out = np.empty(nxo, fld.dtype)

        # Reduce grid resolution
        _reduce_grid_resolution_1d__core(fld, fld_out, stride, imode)

    elif len(fld.shape) == 2:

        # Determine grid sizes (in and out)
        nxi, nyi = fld.shape
        nxo = int(nxi / stride)
        nyo = int(nyi / stride)

        # Initialize output field
        fld_out = np.empty([nxo, nyo], fld.dtype)

        # Reduce grid resolution
        _reduce_grid_resolution_2d__core(fld, fld_out, stride, imode)

    else:
        raise ValueError("invalid shape", fld.shape)

    return fld_out


def _reduce_grid_resolution_1d__core(
    np.float32_t [:] fld_in, np.float32_t [:] fld_out, int stride, int imode,
):
    cdef int half_stride = int((stride - 1)/2)
    cdef int nxi = fld_in.shape[0]
    cdef int nxo = fld_out.shape[0]
    cdef np.float32_t box_size = stride
    cdef int io
    cdef int ii
    cdef int iis
    cdef int iie
    for io in range(nxo):
        if imode == 0:
            # Pick the value at the center point
            ii = io*stride + half_stride
            fld_out[io] = fld_in[ii]
        elif imode == 1:
            # Compute mean over box
            iis = io*stride
            iie = (io + 1)*stride
            fld_out[io] = 0.0
            for ii in range(iis, iie):
                fld_out[io] += fld_in[ii]
            fld_out[io] /= box_size


def _reduce_grid_resolution_2d__core(
    np.float32_t [:, :] fld_in, np.float32_t [:, :] fld_out, int stride, int imode,
):
    cdef int half_stride = int((stride - 1)/2)
    cdef int nxi = fld_in.shape[0]
    cdef int nyi = fld_in.shape[1]
    cdef int nxo = fld_out.shape[0]
    cdef int nyo = fld_out.shape[1]
    cdef np.float32_t box_size = stride*stride
    cdef int io
    cdef int jo
    cdef int ii
    cdef int ji
    cdef int iis
    cdef int iie
    cdef int jis
    cdef int jie
    for io in range(nxo):
        for jo in range(nyo):
            if imode == 0:
                # Pick the value at the center point
                ii = io*stride + half_stride
                ji = jo*stride + half_stride
                fld_out[io, jo] = fld_in[ii, ji]
            elif imode == 1:
                # Compute mean over box
                iis = io*stride
                jis = jo*stride
                iie = (io + 1)*stride
                jie = (jo + 1)*stride
                fld_out[io, jo] = 0.0
                for ii in range(iis, iie):
                    for ji in range(jis, jie):
                        fld_out[io, jo] += fld_in[ii, ji]
                fld_out[io, jo] /= box_size


def shrink_mask(mask, n):
    """Shrink a 2D mask by N grid points in all four directions."""
    nx, ny = mask.shape
    shrunk = mask.copy().astype(np.uint8)
    _shrink_mask__core(shrunk, nx, ny, n)
    return shrunk.astype(np.bool)


cdef void _shrink_mask__core(np.uint8_t[:, :] mask, int nx, int ny, int n_shrink):
    cdef int i, j
    cdef int val
    cdef int prev
    cdef int changing = 0

    # Initialize array indicating which values have been changed
    cdef bint** changed = <bint**>malloc(nx*sizeof(bint*))
    for i in range(nx):
        changed[i] = <bint*>malloc(ny*sizeof(bint))
        for j in range(ny):
            changed[i][j] = 0

    # Vertical direction
    for i in range(nx):

        # Bottom to top
        prev = 0
        for j in range(ny):
            val = mask[i][j]
            if val == 1:
                # -- In positive area (not masked)
                if prev == 0:
                    # Just entering positive area!
                    changing = n_shrink
                if changing > 0:
                    # Still in the process of shrinking
                    changed[i][j] = 1
                    changing -= 1
            elif val == 0:
                # -- In negative area (masked)
                if changing > 0:
                    changing = 0 # reset
            prev = val

        # Top to bottom
        prev = 0
        for j in range(ny-1, -1, -1):
            val = mask[i][j]
            if val == 1:
                # -- In positive area (not masked)
                if prev == 0:
                    # Just entering positive area!
                    changing = n_shrink
                if changing > 0:
                    # Still in the process of shrinking
                    changed[i][j] = 1
                    changing -= 1
            elif val == 0:
                # -- In negative area (masked)
                if changing > 0:
                    changing = 0 # reset
            prev = val

    # Horizontal direction
    for j in range(ny):

        # Left to right
        prev = 0
        for i in range(nx):
            val = mask[i][j]
            if val == 1:
                # -- In positive area (not masked)
                if prev == 0:
                    # Just entering positive area!
                    changing = n_shrink
                if changing > 0:
                    # Still in the process of shrinking
                    changed[i][j] = 1
                    changing -= 1
            elif val == 0:
                # -- In negative area (masked)
                if changing > 0:
                    changing = 0 # reset
            prev = val

        # Top to bottom
        prev = 0
        for i in range(nx-1, -1, -1):
            val = mask[i][j]
            if val == 1:
                # -- In positive area (not masked)
                if prev == 0:
                    # Just entering positive area!
                    changing = n_shrink
                if changing > 0:
                    # Still in the process of shrinking
                    changed[i][j] = 1
                    changing -= 1
            elif val == 0:
                # -- In negative area (masked)
                if changing > 0:
                    changing = 0 # reset
            prev = val

    # Apply changes
    for i in range(nx):
        for j in range(ny):
            if changed[i][j]:
                mask[i][j] = 0

    # Clean up
    for i in range(nx):
        free(changed[i])
    free(changed)


def min_griddist_mask_boundary(mask, *, dirs=None, nan=None):
    """Compute the minimum along-grid distance to a mask boundary.

    Distances inside a mask are positive, those outside negative.

    """
    if dirs is None:
        dirs = list("NEWS")

    if len(dirs) == 0:
        return np.where(mask, mask.shape[0], nan).astype(np.int32)

    # Check directions
    dirs_choices = list("NESW")
    if not all(d in dirs_choices for d in dirs):
        raise ValueError("relaxation directions must all be among {dirs_choices}", dirs)
    N, E, S, W = [d in dirs for d in "NESW"]

    # Compute distances
    nx, ny = mask.shape
    if nan is None:
        nan = -(nx + ny)
    dists = np.full([nx, ny], nan, dtype=np.int32)
    _min_griddist_mask_boundary__core(
        mask.astype(np.uint8), dists, N, E, S, W, nan, nx, ny,
    )

    return dists


cdef void _min_griddist_mask_boundary__core(
    np.uint8_t[:, :] mask,
    np.int32_t[:, :] dists,
    bint do_N,
    bint do_E,
    bint do_S,
    bint do_W,
    int nan,
    int nx,
    int ny,
):
    cdef int i
    cdef int j
    cdef int val
    cdef int prev
    cdef int dist
    cdef int sign

    # Vertical direction
    for i in range(nx):

        # Bottom to top
        if do_S:
            prev = -1
            sign = 0
            for j in range(ny):
                val = mask[i][j]
                if val == 1 and prev <= 0:
                    # Entering positive area from a negative area or outside
                    sign = 1
                    dist = 0
                elif val == 0 and prev == 1:
                    # Entering negative area from a positive area
                    sign = -1
                    dist = -1
                if sign != 0:
                    if dists[i][j] == nan or abs(dist) < abs(dists[i][j]):
                        dists[i][j] = dist
                    dist += sign
                prev = val

        # Top to bottom
        if do_N:
            prev = -1
            sign = 0
            for j in range(ny-1, -1, -1):
                val = mask[i][j]
                if val == 1 and prev <= 0:
                    # Entering positive area from a negative area or outside
                    sign = 1
                    dist = 0
                elif val == 0 and prev == 1:
                    # Entering negative area from a positive area
                    sign = -1
                    dist = -1
                if sign != 0:
                    if dists[i][j] == nan or abs(dist) < abs(dists[i][j]):
                        dists[i][j] = dist
                    dist += sign
                prev = val

    # Horizontal direction
    for j in range(ny):

        # Left to right
        if do_W:
            prev = -1
            sign = 0
            for i in range(nx):
                val = mask[i][j]
                if val == 1 and prev <= 0:
                    # Entering positive area from a negative area or outside
                    sign = 1
                    dist = 0
                elif val == 0 and prev == 1:
                    # Entering negative area from a positive area
                    sign = -1
                    dist = -1
                if sign != 0:
                    if dists[i][j] == nan or abs(dist) < abs(dists[i][j]):
                        dists[i][j] = dist
                    dist += sign
                prev = val

        # Right to left
        if do_E:
            prev = -1
            sign = 0
            for i in range(nx-1, -1, -1):
                val = mask[i][j]
                if val == 1 and prev <= 0:
                    # Entering positive area from a negative area or outside
                    sign = 1
                    dist = 0
                elif val == 0 and prev == 1:
                    # Entering negative area from a positive area
                    sign = -1
                    dist = -1
                if sign != 0:
                    if dists[i][j] == nan or abs(dist) < abs(dists[i][j]):
                        dists[i][j] = dist
                    dist += sign
                prev = val


def decompose_rectangular_mask(mask):
    """Determine zones in a field containing a rectangular mask.

    In:
         0  0  0  0  0
         0  0  1  1  0
         0  0  1  1  0
         0  0  1  1  0
         0  0  0  0  0

    Out:
        15 15 35 35 55  y5
        13 13 33 33 53  y3
        13 13 33 33 53  y3
        13 13 33 33 53  y3
        11 11 31 31 51  y1

        x1 x1 x3 x3 x5

    """
    nx, ny = mask.shape
    out = np.zeros([nx, ny], np.int32)
    _decompose_rectangular_mask__core(mask.astype(np.uint8), out, nx, ny)
    return out


cdef void _decompose_rectangular_mask__core(
    np.uint8_t[:, :] mask, np.int32_t[:, :] out, int nx, int ny,
):
    cdef int i
    cdef int j

    # X -zones
    cdef int* zones_x = <int*>malloc(nx*sizeof(int))
    cdef int nseg1_y
    cdef int zone_x_curr
    cdef int zone_x_prev = 0
    cdef bint in_seg1_y
    for i in range(nx):

        # Count number of 1-segments in y-direction
        in_seg1_y = False
        nseg1_y = 0
        for j in range(ny):
            if mask[i, j] == 1 and not in_seg1_y:
                # Entering 1-segment
                in_seg1_y = True
                nseg1_y += 1
            elif mask[i, j] == 0 and in_seg1_y:
                # Exiting 1-segment
                in_seg1_y = False

        # Determine current x-zone
        zone_x_curr = -1
        if nseg1_y == 0:
            if zone_x_prev <= 1:
                zone_x_curr = 1
            else:
                zone_x_curr = 5
        elif nseg1_y == 1:
            zone_x_curr = 3
        zones_x[i] = zone_x_curr

        zone_x_prev = zone_x_curr

    # Y -zones
    cdef int* zones_y = <int*>malloc(nx*sizeof(int))
    cdef int nseg1_x
    cdef int zone_y_curr
    cdef int zone_y_prev = 0
    cdef bint in_seg1_x
    for j in range(ny):

        # Count number of 1-segments in x-direction
        in_seg1_x = False
        nseg1_x = 0
        for i in range(nx):
            if mask[i, j] == 1 and not in_seg1_x:
                # Entering 1-segment
                in_seg1_x = True
                nseg1_x += 1
            elif mask[i, j] == 0 and in_seg1_x:
                # Exiting 1-segment
                in_seg1_x = False

        # Determine current x-zone
        zone_y_curr = -1
        if nseg1_x == 0:
            if zone_y_prev <= 1:
                zone_y_curr = 1
            else:
                zone_y_curr = 5
        elif nseg1_x == 1:
            zone_y_curr = 3
        zones_y[j] = zone_y_curr

        zone_y_prev = zone_y_curr

    # XY -zones
    for i in range(nx):
        for j in range(ny):
            out[i, j] = 10*zones_x[i] + zones_y[j]

    free(zones_x)
    free(zones_y)


def decompose_holy_rectangular_mask(mask):
    """Determine zones in a field containing a rectangular mask with a hole.

    In:
         0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0  0
         0  0  0  1  1  1  1  1  0  0
         0  0  0  1  1  1  1  1  0  0
         0  0  0  1  0  0  1  1  0  0
         0  0  0  1  0  0  1  1  0  0
         0  0  0  1  0  0  1  1  0  0
         0  0  0  1  0  0  1  1  0  0
         0  0  0  1  1  1  1  1  0  0
         0  0  0  0  0  0  0  0  0  0

    Out:
        15 15 15 25 35 35 45 45 55 55  y5
        15 15 15 25 35 35 45 45 55 55  y5
        14 14 14 24 34 34 44 44 54 54  y4
        14 14 14 24 34 34 44 44 54 54  y4
        13 13 13 23 33 33 43 43 53 53  y3
        13 13 13 23 33 33 43 43 53 53  y3
        13 13 13 23 33 33 43 43 53 53  y3
        13 13 13 23 33 33 43 43 53 53  y3
        12 12 12 22 32 32 42 42 52 52  y2
        11 11 11 21 31 31 41 41 51 51  y1

        x1 x1 x1 x2 x3 x3 x4 x4 x5 x5

    """
    nx, ny = mask.shape
    out = np.zeros([nx, ny], np.int32)
    _decompose_holy_rectangular_mask__core(mask.astype(np.uint8), out, nx, ny)
    return out


# CALL <
# > CALLERS:
# > utilities::decompose_holy_rectangular_mask
# CALL >
cdef void _decompose_holy_rectangular_mask__core(
    np.uint8_t[:, :] mask, np.int32_t[:, :] out, int nx, int ny,
):
    cdef int i, j

    # X -zones
    cdef int* zones_x = <int*>malloc(nx*sizeof(int))
    cdef int nseg1_y
    cdef int zone_x_curr
    cdef int zone_x_prev=0
    cdef bint in_seg1_y
    for i in range(nx):

        # Count number of 1-segments in y-direction
        in_seg1_y = False
        nseg1_y = 0
        for j in range(ny):
            if mask[i, j] == 1 and not in_seg1_y:
                # Entering 1-segment
                in_seg1_y = True
                nseg1_y += 1
            elif mask[i, j] == 0 and in_seg1_y:
                # Exiting 1-segment
                in_seg1_y = False

        # Determine current x-zone
        zone_x_curr = -1
        if nseg1_y == 0:
            if zone_x_prev <= 1:
                zone_x_curr = 1
            else:
                zone_x_curr = 5
        elif nseg1_y == 1:
            if zone_x_prev <= 2:
                zone_x_curr = 2
            else:
                zone_x_curr = 4
        elif nseg1_y == 2:
                zone_x_curr = 3
        zones_x[i] = zone_x_curr

        zone_x_prev = zone_x_curr

    # Y -zones
    cdef int* zones_y = <int*>malloc(nx*sizeof(int))
    cdef int nseg1_x
    cdef int zone_y_curr
    cdef int zone_y_prev=0
    cdef bint in_seg1_x
    for j in range(ny):

        # Count number of 1-segments in x-direction
        in_seg1_x = False
        nseg1_x = 0
        for i in range(nx):
            if mask[i, j] == 1 and not in_seg1_x:
                # Entering 1-segment
                in_seg1_x = True
                nseg1_x += 1
            elif mask[i, j] == 0 and in_seg1_x:
                # Exiting 1-segment
                in_seg1_x = False

        # Determine current x-zone
        zone_y_curr = -1
        if nseg1_x == 0:
            if zone_y_prev <= 1:
                zone_y_curr = 1
            else:
                zone_y_curr = 5
        elif nseg1_x == 1:
            if zone_y_prev <= 2:
                zone_y_curr = 2
            else:
                zone_y_curr = 4
        elif nseg1_x == 2:
                zone_y_curr = 3
        zones_y[j] = zone_y_curr

        zone_y_prev = zone_y_curr

    # XY -zones
    for i in range(nx):
        for j in range(ny):
            out[i, j] = 10*zones_x[i] + zones_y[j]

    free(zones_x)
    free(zones_y)


def strip_accents(text):
    """Strip accents from input String.

    source: https://stackoverflow.com/a/31607735/4419816

    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    # SR_TMP <
    text = text.replace("Ø", "O").replace("ø", "o")
    # SR_TMP >
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

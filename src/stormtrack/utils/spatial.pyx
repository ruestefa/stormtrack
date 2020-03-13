# !/usr/bin/env python3

from __future__ import print_function

cimport cython
cimport numpy as np
from libc.math cimport M_PI
from libc.math cimport atan2
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt

# ------------------------------------------------------------------------------

import math

import PIL.Image
import PIL.ImageDraw
import numpy as np
import scipy as sp
import scipy.spatial

try:
    from .various import ipython
except ImportError:
    pass

# ==============================================================================
# Path to domain
# ==============================================================================

def path_lonlat_to_mask(path, *args, **kwas):
    """Create a 2D mask from a path of (lon, lat) tuples.

    See paths_lonlat_to_mask for arguments.
    """
    return paths_lonlat_to_mask([path], *args, **kwas)

def paths_lonlat_to_mask(paths, lon, lat, *, return_tree=False, silent=False,
        _tree=[]):
    """Create a 2D mask from multiple paths of (lon, lat) tuples."""

    # Ensure lon, lat are 2D
    if (len(lon.shape), len(lat.shape)) == (2, 2):
        pass
    elif (len(lon.shape), len(lat.shape)) == (1, 1):
        lat, lon = np.meshgrid(lat, lon)
    else:
        err = ("lon, lat must both be either 1D or 2D, not {}D {} and {}D {}"
                ).format(len(lon.shape), lon.shape, len(lat.shape), lat.shape)
        raise ValueError(err)

    # Initialize or fetch lon/lat lookup tree
    if len(_tree) == 0:
        if not silent:
            print("initialize reusable cKDTree for point lookup")
        lonlat = np.column_stack((lon.ravel(), lat.ravel()))
        tree = sp.spatial.cKDTree(lonlat)
        _tree.append(tree)
    else:
        tree = _tree[0]

    # Convert paths to mask
    nx, ny = lon.shape
    img = PIL.Image.new("L", [nx, ny], 0)
    for path in paths:
        d, inds = tree.query(path, k=1)
        pxs, pys = np.unravel_index(inds, lon.shape)
        path_ind = [(x, y) for x, y in zip(pxs, pys)]
        PIL.ImageDraw.Draw(img).polygon(path_ind, outline=1, fill=1)
    mask = np.array(img)

    # Return result(s)
    if return_tree:
        return mask, tree
    return mask

# ==============================================================================
# Area of points set on lon/lat grid
# ==============================================================================

def points_area_lonlat_reg(pxs, pys, lon1d, lat1d):
    """Compute total area of a set of points on a regular lon/lat grid."""

    # Number of points
    if pxs.size != pys.size:
        err = ("different numbers of points in x and y: {} != {}"
                ).format(pxs.size, pys.size)
        raise ValueError(err)
    npt = pxs.shape[0]

    # Longitude
    if len(lon1d.shape) != 1:
        err = "lon1d: not one-dimensional: {}".format(lon1d.shape)
        raise ValueError(err)
    nlon = lon1d.size

    # Latitude
    if len(lat1d.shape) != 1:
        err = "lat1d: not one-dimensional: {}".format(lat1d.shape)
        raise ValueError(err)
    nlat = lat1d.size

    # Longitude grid spacing
    dlons = lon1d[1:] - lon1d[:-1]
    if np.unique(dlons).size > 1:
        err = "longitude grid spacing varies: {}".format(np.unique(dlons))
    dlon = dlons[0]

    # Latitude grid spacing
    dlats = lat1d[1:] - lat1d[:-1]
    if np.unique(dlats).size > 1:
        err = "latitude grid spacing varies: {}".format(np.unique(dlats))
    dlat = dlats[0]

    # Ensure correct types
    pxs   = pxs.astype(np.int32)
    pys   = pys.astype(np.int32)
    lon1d = lon1d.astype(np.float64)
    lat1d = lat1d.astype(np.float64)

    # Compute area over all points
    area_km2 = _feature_area_lonlat__core(pxs, pys, lon1d, lat1d,
            npt, nlon, nlat, dlon, dlat)

    return area_km2

cdef np.float64_t _feature_area_lonlat__core(
        np.int32_t  [:] pxs,
        np.int32_t  [:] pys,
        np.float64_t[:] lon,
        np.float64_t[:] lat,
        np.int32_t      npt,
        np.int32_t      nlon,
        np.int32_t      nlat,
        np.float64_t    dlon,
        np.float64_t    dlat,
    ) except -1:

    cdef int p, j, k

    # -- Compute length of a degree longitude (m) at a given latitude
    #   src: https://en.wikipedia.org/wiki/Longitude (2019-04-02)

    cdef np.float64_t rad = 6378100.0       # Earth radius (m)
    cdef np.float64_t dlat_km = dlat*111.32 # latitude distance (km)
    cdef np.float64_t pi180 = M_PI/180

    cdef np.ndarray[np.float64_t, ndim=1] _areas_km2 = np.empty(nlat, np.float64)
    cdef np.float64_t[:] areas_km2 = _areas_km2

    # -- Compute the area at each latitude on the grid

    # Subdivide each grid box into nk subboxes in latitudinal direction to
    # try and account for the non-linear decrease in longitudinal distance.
    #
    # However, in tests (e.g., area of 800 km circle centered at 80 N)
    # this made no difference, even for nk=100... There might be cases
    # where this actually works, but non have been encountered so far.
    # (Of course, there might also be an algorithm/implementation issue!)
    cdef int nk = 1
    # cdef int nk = 5
    # cdef int nk = 10
    # cdef int nk = 100

    cdef int nkh = int((nk - 1)/2)
    cdef np.ndarray[np.float64_t, ndim=1] _areas_km2_j = np.empty(nk, np.float64)

    cdef np.float64_t[:] areas_km2_j = _areas_km2_j
    cdef np.float64_t dlon_km_k, area_km2_k
    cdef np.float64_t lat_k, f, f0, f1

    for j in range(nlat):

        areas_km2[j] = 0.0

        # Loop over subcells
        areas_km2[j] = 0.0
        for k in range(nk):

            # Determine latitude
            f = float(k - nkh)/nk
            lat_k = lat[j] + f*dlat
            if k < nkh:
                lat_k = max(lat_k, -90.0)
            elif k > nkh:
                lat_k = min(lat_k,  90.0)

            # Determine longitude distance at latitude
            dlon_km_k = dlon*pi180*rad*cos(lat_k*pi180)/1000.0

            # Compute area
            area_km2_k = dlon_km_k*dlat_km/float(nk)

            # Increment total area
            areas_km2[j] += area_km2_k

    # Compute total area over all points
    cdef np.float64_t area_km2=0.0
    for p in range(npt):
        j = pys[p]
        area_km2 += areas_km2[j]

    return area_km2

# ==============================================================================
# Various
# ==============================================================================

def path_along_domain_boundary(lon, lat, nbnd=0):
    """Construct a path parallel to the domain boundary.

    Slice the lon and lat arrays. The distance to the boundary is given in
    grid points as <nbnd>.

    Returns
    -------
     x coordinates of path: array
     y coordinates of path: array
    """
    # SR_TODO: Find out why e.g. the first is not the southern boundary!
    # SR_TODO: Somehow the values in lon,lat are stored as (y,x)...
    ns, ne = nbnd, -(nbnd + 1)
    pxs = (
            lon[ns:ne, ns].tolist()    +  # west
            lon[ne, ns:ne].tolist()    +  # north
            lon[ne:ns:-1, ne].tolist() +  # east
            lon[ns, ne:ns:-1].tolist()    # south
        )
    pys = (
            lat[ns:ne, ns].tolist()    +  # west
            lat[ne, ns:ne].tolist()    +  # north
            lat[ne:ns:-1, ne].tolist() +  # east
            lat[ns, ne:ns:-1].tolist()    # south
        )
    if (pxs[-1], pys[-1]) != (pxs[0], pys[0]):
        pxs.append(pxs[0])
        pys.append(pys[0])
    return pxs, pys

def great_circle_distance(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points

    based on : https://gist.github.com/gabesmed/1826175


    Parameters
    ----------

    lon1: float
        longitude of the starting point
    lat1: float
        latitude of the starting point
    lon2: float
        longitude of the ending point
    lat2: float
        latitude of the ending point

    Returns
    -------

    distance (km): float

    Examples
    --------

    >>> great_circle_distance(0, 55, 8, 45.5)
    1199.3240879770135
    """

    EARTH_CIRCUMFERENCE = 6378137  # earth circumference in meters

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_CIRCUMFERENCE * c

    return distance / 1000

def derive_lonlat_1d(lon2d, lat2d, *, nonreg_ok=False):
    """Derive 1D lon/lat from 2D regular lon/lat fields."""

    # Assume regular lon/lat grid
    lon1d = lon2d[:, 0]
    lat1d = lat2d[0, :]

    if (lon1d == lon1d[0]).all() and (lat1d == lat1d[0]).all():
        err = ("uniform 1D lon/lat fields ({}/{}); likely the input 2D fields "
                "need to be transposed").format(lon1d[0], lat1d[0])
        raise Exception(err)

    # -- Check if grid is indeed regular

    lat2d_check, lon2d_check = np.meshgrid(lat1d, lon1d)

    if lat2d_check.shape != lat2d.shape:
        err = ("inconsistent shapes: lon2d {} -> lon1d {} -> lon2d_check {}"
                ).format(lon2d.shape, lon1d.shape, lon2d_check.shape)
        raise Exception(err)

    if lat2d_check.shape != lat2d.shape:
        err = ("inconsistent shapes: lat2d {} -> lat1d {} -> lat2d_check {}"
                ).format(lat2d.shape, lat1d.shape, lat2d_check.shape)
        raise Exception(err)

    if (lon2d == lon2d_check).all() and (lat2d == lat2d_check).all():

        # Indeed, grid is regular!
        return lon1d, lat1d

    elif nonreg_ok:

        # Non-regular, but that's OK
        return None, None

    else:
        raise ValueError("non-regular lon/lat grid")

# ==============================================================================


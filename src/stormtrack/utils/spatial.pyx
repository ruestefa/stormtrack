# !/usr/bin/env python3

from __future__ import print_function

# C: C libraries
from libc.math cimport M_PI
from libc.math cimport atan2
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt

# C: Third-party
cimport cython
cimport numpy as np

# Standard library
import math

# Third-party
import PIL.Image
import PIL.ImageDraw
import numpy as np
import scipy as sp
import scipy.spatial


def path_lonlat_to_mask(path, *args, **kwas):
    """Create a 2D mask from a path of (lon, lat) tuples.

    See paths_lonlat_to_mask for arguments.

    """
    return paths_lonlat_to_mask([path], *args, **kwas)


def paths_lonlat_to_mask(paths, lon, lat, *, return_tree=False, silent=False, _tree=[]):
    """Create a 2D mask from multiple paths of (lon, lat) tuples."""

    # Ensure lon, lat are 2D
    if (len(lon.shape), len(lat.shape)) == (2, 2):
        pass
    elif (len(lon.shape), len(lat.shape)) == (1, 1):
        lat, lon = np.meshgrid(lat, lon)
    else:
        raise ValueError(
            f"lon, lat must both be either 1D or 2D, not {len(lon.shape)}D {lon.shape} "
            f"and {len(lat.shape)}D {lat.shape}"
        )

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
    # Note: Numpy reads in order (y, x[, z]), therefore transpose
    mask = np.array(img).T
    if mask.shape != lon.shape:
        raise Exception(
            f"constructed mask has wrong shape: {mask.shape} != {lon.shape}"
        )

    # Return result(s)
    if return_tree:
        return mask, tree
    return mask


def points_area_lonlat_reg(pxs, pys, lon1d, lat1d):
    """Compute total area of a set of points on a regular lon/lat grid."""

    # Number of points
    if pxs.size != pys.size:
        raise ValueError(
            f"different numbers of points in x and y: {pxs.size} != {pys.size}"
        )
    npt = pxs.shape[0]

    # Longitude
    if len(lon1d.shape) != 1:
        raise ValueError("lon1d not one-dimensional", lon1d.shape)
    nlon = lon1d.size

    # Latitude
    if len(lat1d.shape) != 1:
        raise ValueError("lat1d not one-dimensional", lat1d.shape)
    nlat = lat1d.size

    # Longitude grid spacing
    dlons = lon1d[1:] - lon1d[:-1]
    unique_dlons = np.unique(dlons.round(decimals=5))
    if unique_dlons.size > 1:
        raise ValueError("longitude grid spacing varies", unique_dlons, dlons)
    dlon = dlons[0]

    # Latitude grid spacing
    dlats = lat1d[1:] - lat1d[:-1]
    unique_dlats = np.unique(dlons.round(decimals=5))
    if unique_dlats.size > 1:
        raise ValueError("latitude grid spacing varies", unique_dlats, dlats)
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
    np.int32_t [:] pxs,
    np.int32_t [:] pys,
    np.float64_t[:] lon,
    np.float64_t[:] lat,
    np.int32_t npt,
    np.int32_t nlon,
    np.int32_t nlat,
    np.float64_t  dlon,
    np.float64_t  dlat,
) except -1:
    cdef int p
    cdef int j
    cdef int k

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

    cdef int nkh = int((nk - 1) / 2)
    cdef np.ndarray[np.float64_t, ndim=1] _areas_km2_j = np.empty(nk, np.float64)
    cdef np.float64_t[:] areas_km2_j = _areas_km2_j
    cdef np.float64_t dlon_km_k
    cdef np.float64_t area_km2_k
    cdef np.float64_t lat_k
    cdef np.float64_t f
    cdef np.float64_t f0
    cdef np.float64_t f1
    for j in range(nlat):
        areas_km2[j] = 0.0

        # Loop over subcells
        areas_km2[j] = 0.0
        for k in range(nk):

            # Determine latitude
            f = float(k - nkh) / nk
            lat_k = lat[j] + f * dlat
            if k < nkh:
                lat_k = max(lat_k, -90.0)
            elif k > nkh:
                lat_k = min(lat_k,  90.0)

            # Determine longitude distance at latitude
            dlon_km_k = dlon * pi180 * rad * cos(lat_k * pi180) / 1000.0

            # Compute area
            area_km2_k = dlon_km_k * dlat_km / float(nk)

            # Increment total area
            areas_km2[j] += area_km2_k

    # Compute total area over all points
    cdef np.float64_t area_km2 = 0.0
    for p in range(npt):
        j = pys[p]
        area_km2 += areas_km2[j]

    return area_km2


def path_along_domain_boundary(lon, lat, nbnd=0, bnd_nan=False, fld=None):
    """Construct a path parallel to the domain boundary.

    Slice the lon and lat arrays. The distance to the boundary is given in
    grid points as <nbnd>.

    Returns
    -------
     x coordinates of path: array
     y coordinates of path: array

    """
    if not bnd_nan:
        (nx, ny) = lon.shape
        (nx, ny) = lon.shape
        xs = 0
        xe = nx
        ys = 0
        ye = ny
    else:
        if fld is None:
            raise ValueError("must pass fld for bnd_nan=T")
        (xs, xe, ys, ye) = locate_domain_in_nans(fld)

    xs += nbnd
    xe -= nbnd
    ys += nbnd
    ye -= nbnd

    pxs = (
        lon[xs : xe, ys].tolist()  # west
        + lon[xe - 1, ys : ye].tolist()  # north
        + lon[xe : xs : -1, ye - 1].tolist()  # east
        + lon[xs, ye : ys : -1].tolist()  # south
    )
    pys = (
        lat[xs : xe, ys].tolist()  # west
        + lat[xe - 1, ys :  ye].tolist()  # north
        + lat[xe : xs : -1, ye - 1].tolist()  # east
        + lat[xs, ye : ys : -1].tolist()  # south
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
    a = (
        math.sin(dLat / 2)
        * math.sin(dLat / 2)
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dLon / 2)
        * math.sin(dLon / 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = EARTH_CIRCUMFERENCE * c
    return distance / 1000


def derive_lonlat_1d(lon2d, lat2d, *, nonreg_ok=False):
    """Derive 1D lon/lat from 2D regular lon/lat fields."""

    # Assume regular lon/lat grid
    lon1d = lon2d[:, 0]
    lat1d = lat2d[0, :]

    if (lon1d == lon1d[0]).all() and (lat1d == lat1d[0]).all():
        raise Exception(
            f"uniform 1D lon/lat fields ({lon1d[0]}/{lat1d[0]}); "
            f"likely the input 2D fields "
        )

    # -- Check if grid is indeed regular

    lat2d_check, lon2d_check = np.meshgrid(lat1d, lon1d)

    if lat2d_check.shape != lat2d.shape:
        raise Exception(
            f"inconsistent shapes: lon2d {lon2d.shape} -> lon1d {lon1d.shape} -> "
            f"lon2d_check {lon2d_check.shape}"
        )

    if lat2d_check.shape != lat2d.shape:
        raise Exception(
            f"inconsistent shapes: lat2d {lat2d.shape} -> lat1d {lat1d.shape} -> "
            f"lat2d_check {lat2d_check.shape}"
        )

    if (lon2d == lon2d_check).all() and (lat2d == lat2d_check).all():

        # Indeed, grid is regular!
        return lon1d, lat1d

    elif nonreg_ok:

        # Non-regular, but that's OK
        return None, None

    else:
        raise ValueError("non-regular lon/lat grid")


def locate_domain_in_nans(field):
    """"Determine the extent a rectangular domain nested in NaNs.

    Note: A slightly different version of this function is available in DyPy:
    https://git.iac.ethz.ch/atmosdyn/DyPy/-/blob/master/src/dypy/small_tools.py

    """
    fld = np.asarray(field)
    (xs, ys) = 0, 0
    (xe, ye) = fld.shape
    while np.isnan(fld[xs, :]).all():
        assert xs < xe
        xs += 1
    while np.isnan(fld[xe - 1, :]).all():
        assert xe > xs
        xe -= 1
    while np.isnan(fld[:, ys]).all():
        assert ys < ye
        ys += 1
    while np.isnan(fld[:, ye - 1]).all():
        assert ye > ys
        ye -= 1
    return (xs, xe, ys, ye)

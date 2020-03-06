#!/usr/bin/env python3

from __future__ import print_function

# C: C libraries
from cython.parallel cimport prange
from libc.math cimport M_PI
from libc.math cimport atan2
from libc.math cimport cos
from libc.math cimport sin
from libc.math cimport sqrt

# C: Third-party
cimport cython
cimport numpy as np

# Third-party
import cython
import numpy as np


def pairwise_distances_great_circle(lon, lat):
    """Compute the pairwise great circle distance (km) between all points."""
    return _pairwise_distances_great_circle__core(lon, lat)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray[np.float32_t, ndim=4] _pairwise_distances_great_circle__core(
        np.ndarray[np.float32_t, ndim=2] lon,
        np.ndarray[np.float32_t, ndim=2] lat,
    ):
    cdef int nx=lon.shape[0], ny=lon.shape[1]

    cdef np.ndarray[np.float32_t, ndim=4] \
            dists = np.empty([nx, ny, nx, ny], np.float32)

    cdef np.float32_t EARTH_CIRC = 6378137
    cdef np.float32_t PI180 = M_PI/180

    cdef np.float32_t lon0, lat0, lon1, lat1, dist
    cdef np.float32_t dlon, dlat, a, c

    cdef int i0, j0, i1, j1
    for i0 in range(nx):
        for j0 in range(ny):
            lon0 = lon[i0, j0]
            lat0 = lat[i0, j0]

            for i1 in range(nx):
                for j1 in range(ny):
                    lon1 = lon[i1, j1]
                    lat1 = lat[i1, j1]

                    if i1 == i0 and j1 == j0:
                        dist = 0
                    else:
                        dist = great_circle_distance(lon0, lat0, lon1, lat1)

                    dists[i0, j0, i1, j1] = dist

    return dists


def closest_feature(features, lon=None, lat=None, great_circle=True, *,
        nx=None, ny=None, out=None, mask=None, return_ids=False, approx=False):
    """Compute the distance to the closest feature at every point.

    For every non-feature point, the distance to all shell pixels is computed,
    and the shortest distance stored.

    Optionally, the distance can be approximated, whereby the nearest shell
    pixel is determined with sqrt(dx**2 + dy**2) and the great circle
    distance of the shortest respective distance is stored, which leads to
    a substantial speedup, albeit at the cost of precision, depending on the
    domain.
    """

    #SR_TMP<
    if return_ids:
        raise NotImplementedError("return feature ids")
    #SR_TMP>

    if len(features) == 0:
        dists = np.empty(lon.shape, np.float32)
        dists.fill(np.nan)
        return dists

    if great_circle:
        if lon is None or lat is None:
            raise ValueError("must pass lon, lat for great_circle=True")
    else:
        if nx is None or ny is None:
            raise ValueError("must pass nx, ny for great_circle=False")

    if nx is None or ny is None:
        if lon is not None:
            nx, ny = lon.shape
        elif lat is not None:
            nx, ny = lat.shape
        else:
            raise ValueError("must pass nx, ny or at least one of lon, lat")

    # Prepare mask (Cython-friendly dtype)
    if mask is not None:
        mask = mask.astype(np.uint8)
    else:
        mask = np.ones([nx, ny], np.uint8)

    # Create features mask
    mask_features = features_to_mask(features, nx, ny, np.uint8)

    # Extract shell coordinates
    inds_shell = features_extract_shell_inds(features, lon, lat)

    # Initialize distance array
    if out is not None:
        dists = out
    else:
        dists = np.empty([nx, ny], np.float32)

    # Compute distances
    if not great_circle:
        _compute_feature_distances_xy(dists, inds_shell,
                mask_features, mask, nx, ny)
    elif approx:
        _compute_feature_distances_lonlat_approx(dists, inds_shell,
                mask_features, mask, lon, lat)
    else:
        _compute_feature_distances_lonlat_exact(dists, inds_shell,
                mask_features, mask, lon, lat)

    if out is None:
        return dists


def features_to_mask(features, nx, ny, dtype=np.bool):
    mask_features = None
    for feature in features:
        mask_feature = feature.to_mask(nx, ny)
        if mask_features is None:
            mask_features = mask_feature
        else:
            mask_features |= mask_feature
    return mask_features.astype(dtype)


def features_extract_shell_inds(features, lon, lat, dtype=np.int32):
    inds = []
    for feature in features:
        for shell in feature.shells:
            inds.extend(shell.tolist())
    return np.array(inds, dtype)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _compute_feature_distances_lonlat_exact(
        np.ndarray[np.float32_t, ndim=2] dists,
        np.ndarray[np.int32_t,   ndim=2] inds_shell,
        np.ndarray[np.uint8_t,   ndim=2] mask_features,
        np.ndarray[np.uint8_t,   ndim=2] mask,
        np.ndarray[np.float32_t, ndim=2] lon,
        np.ndarray[np.float32_t, ndim=2] lat,
    ):
    cdef int nx=dists.shape[0], ny=dists.shape[1]
    cdef int ns=inds_shell.size/2
    cdef np.float32_t mindist, dist
    cdef int i0, j0, k, i1, j1
    cdef np.float32_t lon0, lat0, lon1, lat1
    for i0 in range(nx):
        for j0 in range(ny):

            if not mask[i0, j0]:
                continue

            if mask_features[i0, j0]:
                dists[i0, j0] = 0
                continue

            lon0 = lon[i0, j0]
            lat0 = lat[i0, j0]

            for k in range(ns):
                i1 = inds_shell[k, 0]
                j1 = inds_shell[k, 1]
                lon1 = lon[i1, j1]
                lat1 = lat[i1, j1]

                dist = great_circle_distance(lon0, lat0, lon1, lat1)

                if k == 0 or dist < mindist:
                    mindist = dist

            dists[i0, j0] = mindist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _compute_feature_distances_lonlat_approx(
        np.ndarray[np.float32_t, ndim=2] dists,
        np.ndarray[np.int32_t,   ndim=2] inds_shell,
        np.ndarray[np.uint8_t,   ndim=2] mask_features,
        np.ndarray[np.uint8_t,   ndim=2] mask,
        np.ndarray[np.float32_t, ndim=2] lon,
        np.ndarray[np.float32_t, ndim=2] lat,
    ):
    cdef int nx=dists.shape[0], ny=dists.shape[1]
    cdef int ns=inds_shell.size/2
    cdef np.float32_t mindist, dist
    cdef int i0, j0, k, i1, j1
    cdef np.float32_t lon0, lat0, lon1, lat1
    cdef np.float32_t minlon0, minlat0, minlon1, minlat1
    for i0 in range(nx):
        for j0 in range(ny):

            if not mask[i0, j0]:
                continue

            if mask_features[i0, j0]:
                dists[i0, j0] = 0
                continue

            lon0 = lon[i0, j0]
            lat0 = lat[i0, j0]

            for k in range(ns):
                i1 = inds_shell[k, 0]
                j1 = inds_shell[k, 1]

                dist = sqrt((i1 - i0)**2 + (j1 - j0)**2)

                if k == 0 or dist < mindist:
                    mindist = dist
                    lon1 = lon[i1, j1]
                    lat1 = lat[i1, j1]
                    minlon0 = lon0
                    minlat0 = lat0
                    minlon1 = lon1
                    minlat1 = lat1

            mindist = great_circle_distance(minlon0, minlat0, minlon1, minlat1)

            dists[i0, j0] = mindist


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _compute_feature_distances_xy(
        np.ndarray[np.float32_t, ndim=2] dists,
        np.ndarray[np.int32_t,   ndim=2] inds_shell,
        np.ndarray[np.uint8_t,   ndim=2] mask_features,
        np.ndarray[np.uint8_t,   ndim=2] mask,
        int nx, int ny,
    ):
    cdef int ns=inds_shell.size/2
    cdef np.float32_t mindist, dist
    cdef int i0, j0, k, i1, j1
    for i0 in range(nx):
        for j0 in range(ny):

            if not mask[i0, j0]:
                continue

            if mask_features[i0, j0]:
                dists[i0, j0] = 0
                continue

            for k in range(ns):
                i1 = inds_shell[k, 0]
                j1 = inds_shell[k, 1]

                dist = sqrt((i1 - i0)**2 + (j1 - j0)**2)

                if k == 0 or dist < mindist:
                    mindist = dist

            dists[i0, j0] = mindist


cdef np.float32_t great_circle_distance(
        np.float32_t lon0,
        np.float32_t lat0,
        np.float32_t lon1,
        np.float32_t lat1,
    ):
    """Compute great circle distance (km) between two lon, lat coordinates."""
    cdef np.float32_t EARTH_CIRC = 6378137
    cdef np.float32_t PI180 = M_PI/180
    cdef np.float32_t dlon, dlat, a, c
    cdef np.float32_t dist
    dlon = (lon1 - lon0)*PI180
    dlat = (lat1 - lat0)*PI180
    a = (sin(0.5*dlat)*sin(0.5*dlat) +
         cos(lat0*PI180)*cos(lat1*PI180)*
         sin(0.5*dlon)*sin(0.5*dlon))
    c = 2*atan2(sqrt(a), sqrt(1 - a))
    dist = EARTH_CIRC*c/1000
    return dist

# !/usr/bin/env python3

# Standard library
import os
import subprocess
import tempfile
from functools import partial

# Third-party
import netCDF4 as nc4
import numpy as np
import pyproj
import scipy as sp
import shapely.geometry as geo
import shapely.ops

# Local
from .spatial import path_lonlat_to_mask
from .array import trim_mask_grid


def circle_lonlat_mask(clon, clat, rad_km, dlon, dlat, method="dyntools", **kwas):
    """Project a circle on the globe onto a lon/lat grid."""

    # Note: 'dyntools' is more precise than 'dyntools', especially at high
    #       latitudes, and anyway much faster and memory-efficient for fine
    #       grids; 'pyproj' has no no practical advantages, really...
    method_choices = ["pyproj", "dyntools"]

    kwas.update(dict(clon=clon, clat=clat, rad_km=rad_km, dlon=dlon, dlat=dlat,))

    if method == "pyproj":
        if abs(clat) > 45:
            print(
                (
                    "warning: circle_lonlat_mask: abs(clat) {} > 45: "
                    "method 'pyproj' very imprecise at high latitudes; "
                    "better use 'dyntools' (more precise everywhere)"
                ).format(abs(clat))
            )
        return circle_lonlat_mask_pyproj(**kwas)

    elif method == "dyntools":
        return circle_lonlat_mask_dyntools(**kwas)

    else:
        err = ("unknown method '{}'; choices: {}").format(
            method, ", ".join(method_choices)
        )
        raise ValueError(err)


def circle_lonlat_mask_pyproj(
    clon,
    clat,
    rad_km,
    dlon,
    dlat,
    *,
    lon_range=None,
    lat_range=None,
    trim_pad=None,
    silent=True,
):
    """Project a circle on the globe onto a lon/lat grid using pyproj."""

    # Obtain path of circle outline
    path = circle_lonlat_path(clon, clat, rad_km)

    # Create 2D lon/lat arrays
    if lon_range is not None:
        lonmin, lonmax = lon_range
    else:
        lonmin = np.floor(np.array(path)[:, 0].min()) - dlon
        lonmax = np.ceil(np.array(path)[:, 0].max()) + dlon
    if lat_range is not None:
        latmin, latmax = lat_range
    else:
        latmin = np.floor(np.array(path)[:, 1].min()) - dlat
        latmax = np.ceil(np.array(path)[:, 1].max()) + dlat
    lon1d = np.arange(lonmin, lonmax + dlon / 2, dlon)
    lat1d = np.arange(latmin, latmax + dlat / 2, dlat)

    # Project path onto lon/lat grid
    mask = path_lonlat_to_mask(path, lon1d, lat1d, silent=silent)

    grid = {
        "nlat": len(lat1d),
        "nlon": len(lon1d),
        "dlat": dlat,
        "dlon": dlon,
        "latmin": latmin,
        "latmax": latmax,
        "lonmin": lonmin,
        "lonmax": lonmax,
    }

    if trim_pad is not None:
        mask, grid = trim_mask_grid(mask, grid, pad=trim_pad)

    return mask, grid


def circle_lonlat_path(clon, clat, rad_km):
    """Get the path of a circle on the globe as (lon, lat) tuples.

    Source: https://gis.stackexchange.com/a/268277
    """

    rad_m = 1000 * rad_km

    local_azimuthal_projection = (
        r"+proj=aeqd +R=6371000 +units=m +lat_0={clat} +lon_0={clon}"
    )

    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection),
    )

    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    )

    # Transform point to center equidistant azimuthal projection
    point = geo.Point(clon, clat)
    point_transformed = shapely.ops.transform(wgs84_to_aeqd, point)

    # Turn the point into a circle by buffering
    buffer = point_transformed.buffer(rad_m)

    # Transform the circle back to regular lon/lat grid
    buffer_wgs84 = shapely.ops.transform(aeqd_to_wgs84, buffer)

    path = [(x, y) for x, y in geo.mapping(buffer_wgs84)["coordinates"][0]]

    return path


def circle_lonlat_mask_dyntools(
    clon, clat, rad_km, dlon, dlat, lon_range, lat_range, *, trim_pad=None
):

    with tempfile.TemporaryDirectory() as temp_dir:

        center_file = temp_dir + "/center"
        fld_file = temp_dir + "/fld.nc"

        with open(center_file, "w") as fo:
            fo.write("{:f} {:f}".format(clon, clat))

        # Create NetCDF file with circle on globe using 'gridding' from dyn_tools
        kwas = dict(cf=center_file, of=fld_file, r=rad_km, dlo=dlon, dla=dlat)
        kwas["lo0"], kwas["lo1"] = lon_range
        kwas["la0"], kwas["la1"] = lat_range
        cmd = (
            "module load dyn_tools && gridding {cf} {of} {r} "
            "{lo0} {lo1} {la0} {la1} {dlo} {dla}"
        ).format(**kwas)
        out = subprocess.check_output(["/bin/bash", "-c", cmd])
        out = out.decode("utf-8").strip()
        try:
            os.rename(fld_file + ".cdf", fld_file)
        except FileNotFoundError:
            err = (
                "ERROR gridding circle on globe!\n\ncommand:\n{}: " "\n\noutput:\n{}"
            ).format(cmd, out)
            raise Exception(err) from None

        # Read input file
        mask, grid = read_circle_file(fld_file)

    if not mask.any():
        err = "circle mask creation failed: no circle!\n\noutput:{}".format(out)
        raise Exception(err)

    if trim_pad is not None:
        mask, grid = trim_mask_grid(mask, grid, pad=trim_pad)

    return mask, grid


def read_circle_file(infile, swap_latlon=False):

    with nc4.Dataset(infile, "r") as fi:

        latmin = fi.domymin
        latmax = fi.domymax
        lonmin = fi.domxmin
        lonmax = fi.domxmax

        nlat = fi.dimensions["dimy_N"].size
        nlon = fi.dimensions["dimx_N"].size

        mask = fi.variables["N"][0, 0, :, :]

    mask = np.where(mask == 0, 0, 1).astype(np.bool)

    shape = (nlat, nlon)
    assert mask.shape == shape

    dlat = (latmax - latmin) / (nlat - 1)
    dlon = (lonmax - lonmin) / (nlon - 1)

    if swap_latlon:
        mask = mask.T

    grid = {
        "nlat": nlat,
        "nlon": nlon,
        "dlat": dlat,
        "dlon": dlon,
        "latmin": latmin,
        "latmax": latmax,
        "lonmin": lonmin,
        "lonmax": lonmax,
    }

    return mask, grid

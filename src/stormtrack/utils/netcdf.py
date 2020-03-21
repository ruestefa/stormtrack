# !/usr/bin/env python3

import datetime as dt
import logging as log

import netCDF4 as nc4
import numpy as np
import pytz
import scipy as sp

try:
    from .various import ipython
except ImportError:
    pass

# ==============================================================================
# Input
# ==============================================================================

# SR_TODO need to add a check whether _tree is still usable when reused
# SR_TODO e.g. store lon/lat as well in argument lists and check whether
# SR_TODO the ones passed in are the same as the previous ones
def nc_read_var_list(infile, varname, lon, lat, dim_name="list", _tree=[]):
    """Read a field stored in point list form from a netcdf file."""

    # Get dimension length
    with nc4.Dataset(infile, "r") as fi:
        dim = fi.dimensions[dim_name]
        if isinstance(dim, int):
            npts = dim
        else:
            npts = len(dim)

    # Special case: empty list
    if npts == 0:
        return np.zeros(lon.shape)

    # General case: non-empty list
    # SR_TMP <
    err = (
        "read list with netCDF4: what about the option "
        "'aggdim=\"list\"' to dypy.netcdf.read_var? "
        "TODO: drop in with ipython() and check how to do this!"
    )
    raise NotImplementedError(err)
    # SR_TMP >
    try:
        # SR_TMP <
        # -pt_lon, pt_lat, pt_fld = nc.read_var(infile, ["lon", "lat", varname],
        # -        aggdim="list")
        # SR_TMP -
        with nc4.Dataset(infile, "r") as fi:
            pt_lon = fi["lon"][:]
            pt_lat = fi["lat"][:]
            pt_fld = fi[varname][:]  # ???
        # SR_TMP >
    except Exception as e:
        err = "error reading field '{}' from file '{}':\n{}({})".format(
            varname, infile, e.__class__.__name__, e
        )
        raise Exception(err)

    if any(len(f.shape) > 1 for f in (pt_lon, pt_lat, pt_fld)):
        err = "multi-dimensional fields: data not in list format"
        raise Exception(err)

    fld = point_list_to_field(pt_lon, pt_lat, pt_fld, lon, lat, _tree=_tree)
    return fld


def point_list_to_field(pt_lon, pt_lat, pt_fld, lon, lat, _tree=[]):
    inds = points_lonlat_to_inds(pt_lon, pt_lat, lon, lat, _tree)
    px, py = inds.T
    fld = np.zeros(lon.shape, dtype=np.float32)
    fld[px, py] = pt_fld
    return fld


def points_lonlat_to_inds(pt_lon, pt_lat, lon, lat, _tree=[]):
    if len(_tree) == 0:
        log.info("build kdtree (might take a while)")
        lonlat = np.column_stack((lon.ravel(), lat.ravel()))
        _tree.append(sp.spatial.cKDTree(lonlat))
    tree = _tree[0]
    (pts,) = np.dstack([pt_lon, pt_lat])
    dists, inds_raw = tree.query(pts, k=1)
    inds = np.column_stack(np.unravel_index(inds_raw, lon.shape))
    return np.array([(x, y) for x, y in inds])


# ==============================================================================
# Output
# ==============================================================================


def nc_prepare_file(
    fo,
    dims,
    *,
    lon=None,
    lat=None,
    rlon=None,
    rlat=None,
    timesteps=None,
    timestep_bnd0=None,
    rotpollat=43.0,
    rotpollon=-170.0
):
    """Prepare an empty opened netCDF file by adding grid variables etc."""

    # Global attributes
    # TODO

    # -- Dimensions

    # Determine whether to add timestep bounds
    add_bnds = timestep_bnd0 is not None or (
        timesteps is not None and len(timesteps) > 1
    )

    # Timestamps
    if "time" in dims:
        fo.createDimension("time", None)
        if add_bnds:
            fo.createDimension("nii", 2)

    # Grid latitude
    if "lat" in dims:
        if lat is None:
            raise ValueError("'lat' in 'dims' but not passed")
        if len(lat.shape) > 1:
            raise ValueError("'lat' in 'dims' but not 1D")
        fo.createDimension("lat", len(lat))

    # Grid longitude
    if "lon" in dims:
        if lon is None:
            raise ValueError("'lon' in 'dims' but not passed")
        if len(lon.shape) > 1:
            raise ValueError("'lon' in 'dims' but not 1D")
        fo.createDimension("lon", len(lon))

    # Rotated grid latitude
    if "rlat" in dims:
        if rlat is None:
            raise ValueError("'rlat' in 'dims' but not passed")
        fo.createDimension("rlat", len(rlat))
    elif rlat is not None:
        raise ValueError("'rlat' passed yet missing in 'dims'")

    # Rotated grid longitude
    if "rlon" in dims:
        if rlon is None:
            raise ValueError("'rlon' in 'dims' but not passed")
        fo.createDimension("rlon", len(rlon))
    elif rlon is not None:
        raise ValueError("'rlon' passed yet missing in 'dims'")

    rotated = rlat is not None or rlon is not None

    # -- Variables

    # Timestamps
    if "time" in dims and timesteps is not None:
        _v = fo.createVariable("time", "f8", ("time",))
        _v.standard_name = "time"
        _v.long_name = "time"
        _v.units = "seconds since 1970-01-01 00:00:00"
        _v.calendar = "proleptic_gregorian"
        if add_bnds:
            _v.bounds = "time_bnds"
        timestamps = [timestep2timestamp(ts) for ts in timesteps]
        _v[:] = timestamps

        # Time bounds
        if add_bnds:
            _v = fo.createVariable("time_bnds", "f8", ("time", "nii"))
            if timestep_bnd0 is not None:
                timestamp_bnd0 = timestep2timestamp(timestep_bnd0)
            else:
                timestamp_bnd0 = 2 * timestamps[0] - timestamps[1]
            bounds = list(zip([timestamp_bnd0] + timestamps[:-1], timestamps))
            _v[:] = bounds

    # Grid pole rotation
    if rotated:
        _v = fo.createVariable("rotated_pole", "S1", [])
        _v.long_name = "coordinates of the rotated North Pole"
        _v.grid_mapping_name = "rotated_latitude_longitude"
        _v.grid_north_pole_latitude = rotpollat
        _v.grid_north_pole_longitude = rotpollon

    # Rotated grid latitude
    if rlat is not None:
        _dims = ("rlat",)
        _v = fo.createVariable("rlat", "f4", _dims)
        _v.standard_name = "grid_latitude"
        _v.long_name = "rotated latitude"
        _v.units = "degrees"
        _v[:] = rlat

    # Rotated grid longitude
    if rlon is not None:
        _dims = ("rlon",)
        _v = fo.createVariable("rlon", "f4", _dims)
        _v.standard_name = "grid_longitude"
        _v.long_name = "rotated longitude"
        _v.units = "degrees"
        _v[:] = rlon

    # Geographical latitude
    if lat is not None:
        if rotated:
            _dims = ("rlat", "rlon")
        else:
            _dims = ("lat",)
        _v = fo.createVariable("lat", "f4", _dims)
        _v.standard_name = "latitude"
        _v.long_name = "latitude"
        _v.units = "degrees_north"
        if rotated:
            _v.coordinates = "lon lat"
        _v[:] = lat

    # Geographical longitude
    if lon is not None:
        if rotated:
            _dims = ("rlat", "rlon")
        else:
            _dims = ("lon",)
        _v = fo.createVariable("lon", "f4", _dims)
        _v.standard_name = "longitude"
        _v.long_name = "longitude"
        _v.units = "degrees_east"
        if rotated:
            _v.coordinates = "lon lat"
        _v[:] = lon


def timestep2timestamp(timestep, fmt=None):
    """Convert a timestep to a timestamp (seconds since 1970-01-01)."""

    fmts_default = {10: "%Y%m%d%H"}

    if isinstance(timestep, (int, np.integer)):
        # Convert int to str
        timestep = str(timestep)

    if isinstance(timestep, str):
        # Convert str to datetime
        if fmt is None:
            try:
                fmt = fmts_default[len(timestep)]
            except KeyError:
                err = ("no default format for {}-digit timestep among {}").format(
                    len(timestep), sorted(fmts_default)
                )
                raise Exception(err)
        timestep = dt.datetime.strptime(timestep, fmt)

    # Obtain timestamp from datetime
    try:
        timestamp = timestep.replace(tzinfo=pytz.UTC).timestamp()
    except AttributeError:
        err = (
            "cannot derive timestamp from timestep; "
            "'timestep' of type {} not datetime-compatible: {}"
        ).format(type(timestep), timestep)
        raise Exception(err)
    else:
        return timestamp


# ==============================================================================

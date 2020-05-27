#!/usr/bin/env python

# Standard library
import datetime as dt
import functools
import logging as log
import multiprocessing as mp
import os
import sys
from copy import copy
from functools import total_ordering
from pprint import pprint as pp

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import scipy as sp
import scipy.spatial
import shapely.geometry as geo
from scipy.ndimage import filters

# Local
from ..utils.spatial import great_circle_distance
from ..utils.spatial import locate_domain_in_nans
from ..utils.spatial import path_along_domain_boundary

sys.setrecursionlimit(10000)


__all__ = []


DOMAIN_BOUNDARY_CONTOUR = None

# DOMAINS


class Domain:
    def __init__(self, coords):
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        self._coords = coords
        self._poly = geo.Polygon(coords)

    def is_periodic(self):
        return False

    def is_rectangular(self):
        return False

    def boundary(self):
        return geo.LineString(self._coords)

    def boundary_intersection(self, feature, location=None):
        """Get the boundary intersection of a feature (GeometryCollection)."""
        if location:
            err = "Location of boundary intersection for general Domains"
            raise NotImplementedError(err)
        return intersection_objects(self.boundary(), feature.as_polygon())

    def contains(self, other):
        try:
            return self._poly.contains(other.as_polygon())
        except AttributeError:
            pass
        return self._poly.contains(other)


class RectangularDomain(Domain):
    def __init__(self, lon0, lat0, lon1, lat1):
        coords = [(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]
        super().__init__(coords)
        self._lon0, self._lat0, self._lon1, self._lat1 = lon0, lat0, lon1, lat1

    def is_rectangular(self):
        return True

    def boundary(self, location=None):
        if not location:
            coords = self._coords
        elif location == "west":
            coords = [(self._lon0, self._lat0), (self._lon0, self._lat1)]
        elif location == "east":
            coords = [(self._lon1, self._lat0), (self._lon1, self._lat1)]
        else:
            err = "Invalid boundary location {}".format(location)
            raise ValueError(err)
        return geo.LineString(coords)

    def boundary_intersection(self, feature, location=None):
        """Get the boundary intersection of a feature (GeometryCollection)."""
        return intersection_objects(self.boundary(location), feature.as_polygon())

    def lon0(self):
        return self._lon0

    def lon1(self):
        return self._lon1

    def lat0(self):
        return self._lat0

    def lat1(self):
        return self._lat1

    def dlon(self):
        return self._lon1 - self._lon0


class PeriodicDomain(RectangularDomain):
    def __init__(self, lon0, lat0, lon1, lat1):
        super().__init__(lon0, lat0, lon1, lat1)

    def is_periodic(self):
        return True


def intersection_objects(obj1, obj2):
    objs = obj1.intersection(obj2)
    if type(objs) is geo.GeometryCollection:
        return objs
    return geo.GeometryCollection([objs])


class FieldPoint(geo.Point):
    def __init__(self, lon, lat, level, i=None, j=None, id=None):
        geo.Point.__init__(self, lon, lat, level)
        # SR_TODO: Properly implement lon/lat vs. i/j
        # SR_TODO: Replace x/y by lon/lat throughout the code
        self.lon = self.x
        self.lat = self.y
        self.i = i
        self.j = j
        self.id = id
        self.lvl = self.z
        self._contours = None
        self.neighbours = []

    def __eq__(self, other):
        try:
            return (
                self.lon == other.lon
                and self.lat == other.lat
                and self.lvl == other.lvl
            )
        except AttributeError:
            try:
                return (
                    self.lon == other.x and self.lat == other.y and self.lvl == other.z
                )
            except AttributeError:
                return False

    def __lt__(self, other):
        return self.id < other.id

    def get_info(self):
        """Return the most important properties as a dict."""
        return {
            "id": self.id,
            "level": self.lvl,
            "lon": self.lon,
            "lat": self.lat,
            "i": self.i,
            "j": self.j,
        }

    @property
    def xy(self):
        return (self.x, self.y)

    def add_contour(self, contour):
        if self._contours is None:
            self._contours = []
        self._contours.append(contour)

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def add_neighbour_list(self, neighbour_list):
        for neighbour in neighbour_list:
            self.add_neighbour(neighbour)

    def contours_sorted(self):
        if not self._contours:
            return None
        return sorted(self._contours, key=lambda x: x.lvl)

    def confining_contour(self):
        cont = self._contours_sorted()
        return cont[-1] if cont else None

    def distance(self, other):
        return great_circle_distance(self.x, self.y, other.x, other.y)


# see http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class Field2D(np.ndarray):
    def __new__(cls, fld, lon, lat, name=None, **kwargs):

        # construct basic array object
        obj = np.asarray(fld).view(cls)

        # add lon and lat as new properties
        obj.lon = np.asarray(lon)
        obj.lat = np.asarray(lat)

        obj.name = name

        obj._minima = None
        obj._maxima = None

        obj._boundary_hack = False

        for attr, val in kwargs.items():
            cls.setattr(attr, val)

        return obj

    def contours(self):
        return self._contours

    def minima(self):
        return self._minima

    def maxima(self):
        return self._maxima

    def compute_contours(self, levels, conf=None):
        """Compute the contours of the field at certain levels.

        The list of contours is stored on the object and can be accessed
        with the property 'contours'.
        """
        if conf is None:
            conf = {}

        length_threshold, rtol = 10, 1e-4

        self.contour_levels = levels

        log.debug("compute contours")

        # Create contours of SLP (matplotlib.contour.QuadContourSet object)
        cs = plt.contour(self.lon, self.lat, self[:], levels=levels)

        # Initialize contour ID
        if conf.get("ids-datetime", False):
            cid = conf["datetime"] * 10 ** (conf["ids-datetime-digits"] + 1)
        else:
            cid = 0

        contours = []
        # cs.levels is a list of all contour levels (evenly spaced)
        # cs.collections is a mcoll.LineCollection object
        for level, collection in zip(cs.levels, cs.collections):

            # collection is in turn a LineCollection object
            for path_ in collection.get_paths():
                path = path_.vertices

                if len(path) < 4:
                    log.warning(
                        ("skipping short path:\n{}\n" "current ID: {}").format(
                            path, cid
                        )
                    )
                    continue

                if len(path) <= length_threshold:
                    if path_is_insignificant(path, rtol):
                        log.debug(
                            (
                                "skipping insignificant path:\n{}" "\ncurrent ID: {}"
                            ).format(path, cid)
                        )
                        continue

                # collection.get_paths() returns list of Path objects,
                # which contain the points that make up the contour,
                # which means there might be many contours for a given
                # level yield point arrays (themselves 2-element arrays)
                new_contour = Contour(geo.LineString(path), level, id=cid,)
                contours.append(new_contour)
                cid += 1

        self._contours = contours

        if self._boundary_hack:
            self._fix_contour_is_closed()

    def save_contours(self, outfile, cls_writer):
        """Save the contours to file."""
        log.info("save contours to '{}'".format(outfile))
        writer = cls_writer()
        contours = self._contours
        writer.write_contour_path_file(outfile, contours, write_levels=True)

    def read_contours(self, infile, levels, cls_reader):
        """Read the contours from file."""
        # SR_TODO: Move out of utilities; remove Cyclone-specific code
        self.contour_levels = levels
        log.info("read contours from '{}'".format(infile))
        reader = cls_reader()
        paths, contour_levels = reader.read_contour_paths(infile, read_levels=True)
        self._remove_insignificant_paths(paths, 10, 1e-4)
        self._contours = [
            Contour(path, contour_levels[int(id)], id=id) for id, path in paths.items()
        ]
        if self._boundary_hack:
            self._fix_contour_is_closed()

    def _remove_insignificant_paths(self, paths, threshold=10, rtol=None):
        """Remove all paths where all coordinates are too close to each other.

        The check is only performed for paths with a certain maximal length,
        as it is not plausible that long paths are insignificant.
        """
        if rtol is None:
            rtol = 1e-5
        for pid, path in paths.copy().items():
            if len(path) > threshold:
                continue
            if path_is_insignificant(path, rtol):
                del paths[pid]

    def _fix_contour_is_closed(self):
        """Fix the property "is_closed" of the contours.

        When the boundary hack is applied, all contours are closed by
        definition. Therefore, the property Contour.is_closed is wrong.
        Fix this by checking whether the contour path touches a boundary.

        This is hacky as it relies on the internals of the class Contour.
        """
        from shapely.geos import PredicateError, TopologicalError

        log.debug("fix property Contour.is_closed because of boundary hack")
        px, py = path_along_domain_boundary(
            self.lon, self.lat, nbnd=1, bnd_nan=True, fld=self[:]
        )
        domain = geo.Polygon(list(zip(px, py)))
        for cont in self._contours:
            try:
                if not cont.within(domain):
                    cont._mutable = True
                    cont._is_closed = False
                    cont._mutable = False
            except (PredicateError, TopologicalError) as e:
                print(
                    (
                        "warning: field {}: error ({}) determining if contour "
                        "is closed; considering it open"
                    ).format(self.name, e.__class__.__name__)
                )
                cont._mutable = True
                cont._is_closed = False
                cont._mutable = False

    def smooth(self, sig):
        """Apply Gaussian smoothing to the field."""
        self[:] = sp.ndimage.gaussian_filter(self[:], sigma=sig, order=0)

    def apply_boundary_hack(self, n, val):
        """Apply boundary hack to the field.

        Hard-code the outermost (non-NaN) grid points to the same value in order
        to close contours that leave the domain.
        """

        # Get start and end indices of domain (potentially) nested in NaNs.
        xs, xe, ys, ye = locate_domain_in_nans(self)

        # Get the outer indices of the boundary hack zone.
        # Eat into the NaN boundary as far as possible, if there is one.
        # If there is none, or if it's thin, (also) eat into the domain.
        nx, ny = self.shape
        xss = max([xs - n, 0])
        xee = min([xe + n, nx])
        yss = max([ys - n, 0])
        yee = min([ye + n, ny])

        # Get the inner indices of the boundary hack zone.
        xse = xss + n
        xes = xee - n
        yse = yss + n
        yes = yee - n

        self[xss:xse, yss:yee] = val
        self[xes:xee, yss:yee] = val
        self[xss:xee, yss:yse] = val
        self[xss:xee, yes:yee] = val

        self._boundary_hack = True

    def identify_minima(self, *args, **kwargs):
        """Locate all local minima. See core method for details."""
        log.debug("identify minima")
        self._minima = self._identify_extrema(filters.minimum_filter, *args, **kwargs)

    def identify_maxima(self, *args, **kwargs):
        """Locate all local maxima. See core method for details."""
        log.debug("identify maxima")
        self._maxima = self._identify_extrema(filters.maximum_filter, *args, **kwargs)

    def _identify_extrema(self, fct, conf):
        """Identify all local extrema in a field.

        The points are returned as a list of FieldPoint objects.

        Arguments:
         - fct: One of minimum- or maximum_filter from scipy.ndimage.filters.
         - conf: Configuration dict (section "IDENTIFY").
        """

        # `size` denotes the width of a box around a point in which all points
        # may have the same value for the center still to be considered extreme
        # (cf. documentation of e.g. scipy.ndimage.filters.minimum_filter).
        # In this rectangle, all points are set to the value of the exetremum.
        size = conf["extrema-identification-size"]

        # All values in a size**2 area around an extremum are set to it's value.
        # To extract the extreme points, check if they equal the resp. fld point.
        fld_areas = fct(self, size=size, mode="constant", cval=np.nan)
        fld_points = np.where(fld_areas == self, self, np.nan)

        # Set base ID (optionally based on datetime).
        id0 = 0
        if conf.get("ids-datetime", False):
            id0 = conf["datetime"] * 10 ** (conf["ids-datetime-digits"] + 1)

        # Create point objects
        mask = ~np.isnan(fld_points)
        lon, lat = self.lon, self.lat
        fct = lambda i, j, l, id: FieldPoint(lon[i, j], lat[i, j], l, i=i, j=j, id=id)
        pts = [
            fct(i, j, l, id0 + id)
            for id, (i, j, l) in enumerate(
                ((i, j, l) for (i, j), l in np.ndenumerate(self) if mask[i, j])
            )
        ]

        return pts

    def save_extrema(self, outfile, writer):
        """Save the minima and maxima to file."""
        from . import io

        log.info("save minima and maxima to '{}'".format(outfile))
        writer.write_points_file(outfile, minima=self._minima, maxima=self._maxima)

    def read_extrema(self, infile, cls_reader):
        """Read the minima and maxima from file."""
        from . import io

        log.info("read minima and maxima from '{}'".format(infile))
        reader = cls_reader()
        extrema = reader.read_points_file(infile)
        self._minima = extrema["minima"]
        self._maxima = extrema["maxima"]

    def points_filter_boundary(self, nbd):
        """Remove all identified points near the domain boundaries.

        Remove all minima and maxima located in a zone <nbd> grid points
        wide next to the domain boundaries.
        """
        if nbd <= 0:
            return
        px, py = path_along_domain_boundary(self.lon, self.lat, nbd)
        domain = geo.Polygon(list(zip(px, py)))
        self._minima = [p for p in self._minima if p.within(domain)]
        self._maxima = [p for p in self._maxima if p.within(domain)]

    def points_filter_topography(self, field, cutoff_level):
        """Remove all identified points over high topography.

        The criterion for removal of a point is whether the value of the
        surface field at this point is above (height field) or below
        (pressure field) a given threshold.
        """
        if field is None:
            raise ValueError("Invalid topography field!")
        if field.name is None:
            raise ValueError("Invalid topography field; name missing!")

        if field.name in ["ZB", "HSURF"]:
            fct = lambda p: field[p.i, p.j] < cutoff_level
        elif field.name in ["PS"]:
            fct = lambda p: field[p.i, p.j] > cutoff_level
        else:
            err = "Invalid topography field name '{}'".format(field.name)
            raise ValueError(err)

        self._minima = [p for p in self._minima if fct(p)]
        self._maxima = [p for p in self._maxima if fct(p)]


class ImmutableError(Exception):
    pass


class Contour(geo.Polygon):

    _immutables = ["lvl", "shell", "id", "_immutables", "_is_closed"]

    def __init__(self, path, level=None, id=-1, rtol=1e-5):
        self._mutable = True

        if not isinstance(path, geo.LineString):
            path = geo.LineString(path)

        self.set_is_closed(path)
        if self.is_closed():
            path = self._fix_closed_paths(path, rtol)

        super(Contour, self).__init__(shell=path, holes=None)

        self._path = path
        self.lvl = level
        self.z = self.lvl
        self.id = id

        self._mutable = False

    def _fix_closed_paths(self, path, rtol):
        """Remove points from the end that are equal to the first point."""
        while np.allclose(path.coords[0], path.coords[-1], rtol=rtol):
            path = geo.LineString(path.coords[:-1])
            if len(path.coords) <= 2:
                raise ValueError("Invalid path")
        return path

    def __repr__(self):
        cls = self.__class__.__name__
        return "<{cls}[{id}] z={z}>".format(cls=cls, id=id(self), z=self.lvl)

    # SR_TODO: Proper implementation!
    def __eq__(self, other):
        """Two contours are equal if their path, level, and id matches.

        - If both contours have an id, they must match for equality.
        In this case, an exception is raised if their path and/or level
        doesn't match, because contour ids must be unique.

        - If one or both contours don't have an id, both their path and
        (if given for both) level have to be identical.
        """
        try:
            if self.has_valid_id() and other.has_valid_id():
                if self.id != other.id:
                    return False
        except AttributeError:
            pass
        try:
            if self.path() != other.path():
                return False
        except AttributeError:
            pass
        try:
            if self.lvl and other.lvl and self.lvl != other.lvl:
                return False
        except AttributeError:
            pass
        return True

    def _assert_equal_level(self, other):
        if not (self.has_valid_level() and other.has_valid_level()):
            return
        if self.lvl != other.lvl:
            msg = (
                "Contour objects with the same ID must have"
                " identical levels!\n"
                " -> self  : id={si}, level={sl}\n"
                " -> other : id={oi}, level={ol}\n"
            ).format(si=self.id, sl=self.lvl, oi=other.id, ol=other.lvl)
            raise ValueError(msg)

    def _assert_equal_path(self, other):
        if self.path() != other.path():
            msg = (
                "Contour objects with the same ID must have"
                " different paths!\n"
                " -> self  : id={si}, level={sp}\n"
                " -> other : id={oi}, level={op}\n"
            ).format(si=self.id, sp=self.path(), oi=other.id, op=other.path())
            raise ValueError(msg)

    @total_ordering
    def __lt__(self, other):
        if not (self.has_valid_id() and other.has_valid_id()):
            msg = "To be orderable, contours must have a valid id"
            raise ValueError(msg)
        return self.lvl < other.lvl

    def has_valid_id(self):
        try:
            return self.id >= 0
        except TypeError:
            return False

    def has_valid_level(self):
        return self.lvl is not None

    def get_info(self, paths=True):
        cont_entry = {}
        cont_entry["id"] = self.id
        cont_entry["level"] = self.lvl
        if paths:
            cont_entry["path"] = self.path()
        return cont_entry

    # Make selected properties immutable

    def __setattr__(self, key, value):
        """Make the class immutable.

        Working under this assumption simplyfies the design of some methods.

        Note that this immutability can easily be circumvented by setting
        _mutable=True temporarily, but this requires some effort and thus
        cannot be done accidentally.
        """
        self.__check_mutability(key)
        # super.__setattr__(key, value) # python3
        geo.Polygon.__setattr__(self, key, value)

    def __delattr__(self, key, value):
        self.check__mutability(key)
        # super.__delattr__(key, value) # python3
        geo.Polygon.__delattr__(self, key, value)

    def __check_mutability(self, key):
        if hasattr(self, "_immutables") and hasattr(self, "_mutable"):
            if key in self._immutables and not self._mutable:
                err = "Objects of class {cls} are immutable!".format(
                    cls=self.__class__.__name__
                )
                raise ImmutableError(err)

    def set_is_closed(self, path, val=None):
        if val is not None:
            self._is_closed = val
            return
        self._is_closed = np.allclose(path.coords[0], path.coords[-1])

    def is_closed(self):
        return self._is_closed

    def is_boundary_crossing(self):
        return not self.is_closed()

    def path(self):
        """Return the path of the contour as a list of (x, y) tuples."""
        try:
            return list(zip(*self.boundary.xy))
        except ValueError:
            return list(zip(*self._path.xy))

    def length_great_circle(self):
        """Get the length of a contour in km along a great circle.

        It is computed as the sum of the lengths of all line segments.
        """
        path = np.asarray(self.path())
        segments = list(zip(path[:-1, 0], path[:-1, 1], path[1:, 0], path[1:, 1]))
        dists = [great_circle_distance(*pts) for pts in segments]
        dist = sum(dists)
        return dist

    def contains_minimum(self, points):
        """Check whether a contour contains a minimum."""
        points = self.contained_minima(points)
        return len(points) > 0

    def contains_maximum(self, points):
        """Check whether a contour contains a maximum."""
        points = self.contained_maxima(points)
        return len(points) > 0

    def contained_minima(self, points):
        """Find all minima inside the contour from a list of points."""
        fct = lambda p_lvl: self.lvl >= p_lvl
        return [p for p in points if self.contains(p) and fct(p.lvl)]

    def contained_maxima(self, points):
        """Find all maxima inside the contour from a list of points."""
        fct = lambda p_lvl: p_lvl >= self.lvl
        return [p for p in points if self.contains(p) and fct(p.lvl)]


def inds2lonlat(px, py, lon, lat):
    indx = np.array(np.round(px), dtype=int)
    indy = np.array(np.round(py), dtype=int)
    plon = lon[indx, indy]
    plat = lat[indx, indy]
    return plon, plat


def area_lonlat(lon, lat):
    lon0, lon1, lat0, lat1 = min(lon), max(lon), min(lat), max(lat)
    proj_str = (
        "+proj=aea" " +lat_1={lat1} +lat_2={lat2}" " +lat_0={lat0} +lon_0={lon0}"
    ).format(
        lat1=min(lat),
        lat2=max(lat),
        lat0=0.5 * (min(lat) + max(lat)),
        lon0=0.5 * (min(lon) + max(lon)),
    )
    proj = pyproj.Proj(proj_str)
    x, y = proj(lon, lat)
    poly = geo.Polygon(list(zip(x, y)))
    return poly.area


def path_is_insignificant(path, rtol):
    path_ = np.asarray(path)
    while np.allclose(path_[0], path_[-1], rtol=rtol):
        path_ = path_[:-1]
        if len(path_) <= 2:
            return True
    return False


def import_file_list(infile):
    """Read a list of files from a file and check their existence."""
    try:
        with open(infile, "r") as f:
            files = f.read().splitlines()
    except IOError as e:
        err = "Error reading file {}: {}".format(infile, e)
        raise IOError(err)
    for file in files:
        if not os.path.isfile(file):
            err = "File not found: {}".format(file)
            raise IOError(file)
    return files


def construct_file_list(infile_tmpl, dt_start, dt_end, timestep=None):
    if timestep is None:
        timestep = dt.timedelta(hours=1)
    infiles = []
    now = dt_start
    while now <= dt_end:
        yyyy, mm, dd, hh = now.strftime("%Y-%m-%d-%H").split("-")
        infiles.append(infile_tmpl.format(YYYY=yyyy, MM=mm, DD=dd, HH=hh))
        now += timestep
    return infiles


class IDManager:
    """Retrieve unique IDs for objects, either as integers or datetimes."""

    def __init__(self, start=0, base=None, digits=4):
        self.start = start
        self.base = base
        self.digits = digits
        self.reset(start=start, base=base, digits=digits)

    def __repr__(self):
        return "{}(start={}, base={}, digits={})".format(
            self.__class__.__name__, self.start, self.base, self.digits
        )

    def current(self):
        return self._next_id

    def next(self):
        while True:
            next_id = self.current()
            self.increment()
            if next_id not in self._blacklist:
                return next_id

    def set_next(self, i):
        self._next_id = i

    def increment(self):
        if self.base:
            if self._next_id == self.max():
                err = "Max. ID reached for {}".format(self)
                raise OverflowError(err)
        self._next_id += 1

    def max(self):
        if self.base:
            return (self.base + 1) * 10 ** self.digits - 1
        return None

    def reset(self, start=None, base=None, digits=None):
        if start:
            self.start = start
        if base:
            self.base = base
        if digits:
            self.digits = digits
        if base:
            self.first = self.base * 10 ** self.digits + self.start
        else:
            self.first = self.start
        self.set_next(self.first)
        self.reset_blacklist()

    def reset_blacklist(self):
        self._blacklist = set()

    def blacklist(self, i):
        self._blacklist.add(i)


def int_list(arg, sep=None):
    return _list_core(arg, "int", int, sep=sep)


def float_list(arg, sep=None):
    return _list_core(arg, "float", float, sep=sep)


def str_list(arg, sep=None):
    return _list_core(arg, "str", str, sep=sep)


def _list_core(arg, name, fct, sep=None, seps=None):
    if seps is None:
        seps = [",", ":", ";", "/"]
    if sep is not None:
        seps = [sep]
    for sep in seps:
        if sep in arg:
            return [fct(s) for s in arg.split(sep)]
    return [arg]


def order_dict(random_dict):
    """Sort a dict and return it as a dict."""
    return dict(sorted(random_dict.items()))


def threshold_at_timestep(thr, ts):
    """Derive timestep-specific threshold from, e.g., monthly thresholds.

    If thr is a 12-element list, those values refer to the thresholds in the
    middle of each month. If the mid-monthly threshold differs from one month
    to the next, the values in-between are obtained by linear interpolation.

    """
    if isinstance(thr, (float, int)):
        # Constant threshold (trivial case)
        return thr

    elif isinstance(thr, (list, tuple)) and len(thr) == 1:
        # Constant threshold (slightly less trivial case)
        return next(iter(thr))

    elif isinstance(thr, (list, tuple)) and len(thr) == 12:
        # -- Monthly thresholds
        if not isinstance(ts, dt.datetime):
            # Convert timestep to dt.datetime
            sts = str(ts)
            if len(sts) == 8:
                ts_fmt = "%Y%m%d"
            elif len(sts) == 10:
                ts_fmt = "%Y%m%d%H"
            elif len(sts) == 12:
                ts_fmt = "%Y%m%d%H%M"
            else:
                raise Exception("cannot deduce timestep format", sts)
            ts = dt.datetime.strptime(str(ts), ts_fmt)

        thrs_ref = thr
        year = ts.year

        # Determine reference timesteps
        thrs_ref = [thrs_ref[-1]] + thrs_ref + [thrs_ref[0]]
        tss_ref = [None] * 14
        tss_ref[0] = dt.datetime(year - 1, 12, 1) + 0.5 * (
            dt.datetime(year, 1, 1) - dt.datetime(year - 1, 12, 1)
        )
        for m in range(1, 12):
            tss_ref[m] = dt.datetime(year, m, 1) + 0.5 * (
                dt.datetime(year, m + 1, 1) - dt.datetime(year, m, 1)
            )
        tss_ref[12] = dt.datetime(year, 12, 1) + 0.5 * (
            dt.datetime(year + 1, 1, 1) - dt.datetime(year, 12, 1)
        )
        tss_ref[13] = dt.datetime(year + 1, 1, 1) + 0.5 * (
            dt.datetime(year + 1, 2, 1) - dt.datetime(year + 1, 1, 1)
        )

        # Determine thresholds sourrounding timestep
        for (thr0, thr1, ts0, ts1) in zip(
            thrs_ref[:-1], thrs_ref[1:], tss_ref[:-1], tss_ref[1:]
        ):
            if ts0 <= ts < ts1:
                break
        else:
            raise Exception("could not place ts among {tss_ref}", ts)

        # Derive threshold by linear interpolation
        if ts == ts0:
            f = 0.0
        else:
            f = (ts - ts0) / (ts1 - ts0)
        thr = f * thr1 + (1 - f) * thr0

        return thr

    else:
        raise ValueError("invalid threshold format", thr)

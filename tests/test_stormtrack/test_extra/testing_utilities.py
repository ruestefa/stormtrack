"""
Utilities for idealized unit tests.
"""

# Standard library
import random
from collections import OrderedDict
from pprint import pformat

# Third-party
import numpy as np
import shapely
from shapely.geometry import Point, Polygon

# Fist-party
from stormtrack.extra.utilities_misc import FieldPoint


# CLASSES


class ContourSimple(Polygon):
    def __init__(self, path, level=None, id=None):
        self._path = path
        Polygon.__init__(self, path, None)
        self.lvl = level
        self.z = self.lvl
        self.id = id

    def __eq__(self, other):
        return self.lvl == other.lvl and self.id == other.id

    def __repr__(self):
        return "<{cls}[id0]: id={id1}, lvl={lvl}>".format(
            cls=self.__class__.__name__, id0=id(self), id1=self.id
        )

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return id(self)

    def __lt__(self, other):
        return self.id < other.id

    def path(self):
        return [(x, y) for x, y in self._path]

    def __repr__(self):
        cls = self.__class__.__name__
        return "<{cls}: id={id}, level={lvl}, area={a:.2f}>".format(
            cls=cls, id=id(self), lvl=self.lvl, a=self.area
        )

    def get_info(self, paths=True):
        cont_entry = OrderedDict()
        cont_entry["id"] = self.id
        cont_entry["level"] = self.lvl
        if paths:
            cont_entry["path"] = self.path()
        return cont_entry

    def is_closed(self):
        return (
            self.path()[0][0] == self.path()[-1][0]
            and self.path()[0][1] == self.path()[-1][1]
        )

    def is_boundary_crossing(self):
        return not self.is_closed()

    # SR_TODO: Clean this up (remove length_great_circle all-together,
    # SR_TODO: or find some other solution)
    def length_great_circle(self):
        pass


class PointSimple(Point):
    def __init__(self, lon, lat, lvl, id=-1):
        Point.__init__(self, lon, lat, lvl)
        self.lon = self.x
        self.lat = self.y
        try:
            self.lvl = self.z
        except Exception:
            self.lvl = None
        self.id = id

    def __repr__(self):
        cls = self.__class__.__name__
        return "<{cls}: id={id}, level={lvl}, xy=({x},{y})>".format(
            cls=cls, id=id(self), lvl=self.lvl, x=self.x, y=self.y
        )

    @property
    def xy(self):
        return (self.x, self.y)

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
            except (AttributeError, shapely.geos.DimensionError):
                return False

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(repr(self))

    def get_info(self):
        pt = FieldPoint(self.lon, self.lat, self.lvl, id=self.id)
        return pt.get_info()


# SETUP


CLS_POINT = PointSimple
CLS_CONTOUR = ContourSimple

CONTOUR_RESOLUTION = 3


# HELPER FUNCTIONS


def contours_are_sorted(contours, reverse=False):
    """Check whether a list of contours is sorted."""
    for i, cont in enumerate(contours[:-1]):
        if not reverse and cont.lvl > contours[i + 1].lvl:
            return False
        if reverse and cont.lvl < contours[i + 1].lvl:
            return False
    return True


def shuffle_contours_assert(cont, n=10):
    """Shuffle a contour list.

    To avoid an AssertionError in case a sorted array is produced by chance
    (can very well happen given the small array size), the shuffling is
    repeated <n> times until no AssertionError is raised.

    """
    for i in range(n):
        random.shuffle(cont)
        try:
            assert not contours_are_sorted(cont)
            assert not contours_are_sorted(cont, reverse=True)
        except AssertionError:
            random.shuffle(cont)
        else:
            break


def create_nested_circular_contours(
    n,
    coord,
    dr,
    lvl,
    *,
    rmin=None,
    no_min=False,
    cls_contour=CLS_CONTOUR,
    cls_point=CLS_POINT,
    contour_resolution=CONTOUR_RESOLUTION,
):
    """Create N nested circular contours around a point.

    A list of lists is returned. The first containins all contours, the
    second the central point. The latter is returned as a one-element
    list instead of a plain object because a list of lists is simpler
    to process further than a list of a list and a single object.

    Arguments:
     - n: Number of contours.
     - coord: Coordinates of the center as an (X,Y) tuple.
     - dr: Radial distance between two contours.
     - lvl: A tuple specifying the level of the innermost contour
    and the contour interval. Note that the level of the central point
    is one interval below the level of the innermost contour.

    Optional arguments:
     - rmin: The radius of the innermost contour. The default is <delta>.
     - no_min: Don't return the minimum, but an empty list instead. The
     - cls_contour: Contour class.
     - cls_point: Point class.
     - contour_resolution: Resolution of contours.

    """
    res = contour_resolution
    lvl, dlvl = [float(l) for l in lvl]
    args = coord + (lvl - dlvl,)
    center = cls_point(*args)
    r0 = dr if rmin is None else rmin
    cont = []
    for i in range(n):
        path = center.buffer(r0 + dr * (i), resolution=res).exterior.coords
        cont.append(cls_contour(path, level=lvl))
        lvl += dlvl
    return [cont, []] if no_min else [cont, [center]]


# ASSERT FUNCTIONS


def assert_dict_contained(full, partial):
    """Check that one dict contains another (possibly smaller) dict."""
    err = "Dict not fully contained in other dict"

    def assert_objects(obj1, obj2, err):
        if obj1 != obj2:
            err += "\nObjects differ:\n{}\n{}".format(pformat(obj1), pformat(obj2))
            raise AssertionError(err)

    def _assert_dict_contained_rec(full, partial, err):
        if not isinstance(partial, dict):
            if isinstance(partial, str):
                assert_objects(full, partial, err)
            else:
                try:
                    it = zip(full, partial)
                except TypeError:
                    assert_objects(full, partial, err)
                else:
                    for f, p in it:
                        _assert_dict_contained_rec(f, p, err)
        else:
            for key, val in partial.items():
                if not key in full:
                    err += "\nKey {k} missing in dict:\n{d}".format(k=key, d=partial)
                    raise AssertionError(err)
                _assert_dict_contained_rec(full[key], val, err)

    _assert_dict_contained_rec(full, partial, err)


if __name__ == "__main__":
    pass

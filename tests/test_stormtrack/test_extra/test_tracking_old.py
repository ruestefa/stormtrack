#!/usr/bin/env python3

# Standard library
import inspect
import itertools
import json
import logging as log
import pickle
import pytest
import sys
import unittest
from unittest import TestCase
from pprint import pprint
from pprint import pformat

# Third-party
import numpy as np
import shapely.affinity
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

# First-party
from stormtrack.extra.io_misc import plot_contours
from stormtrack.extra.tracking_old.tracking import FeatureCombination
from stormtrack.extra.tracking_old.tracking import FeatureFactory
from stormtrack.extra.tracking_old.tracking import FeatureTrackBase
from stormtrack.extra.tracking_old.tracking import FeatureTrackEventBase
from stormtrack.extra.tracking_old.tracking import FeatureTrackEventFactory
from stormtrack.extra.tracking_old.tracking import FeatureTrackExtender
from stormtrack.extra.tracking_old.tracking import FeatureTrackFactory
from stormtrack.extra.tracking_old.tracking import FeatureTrackIOReaderJson
from stormtrack.extra.tracking_old.tracking import FeatureTrackIOWriterJson
from stormtrack.extra.tracking_old.tracking import FeatureTrackerBase
from stormtrack.extra.tracking_old.tracking import PeriodicTrackableFeatureBase
from stormtrack.extra.tracking_old.tracking import TrackableFeatureBase
from stormtrack.extra.utilities_misc import PeriodicDomain
from stormtrack.extra.utilities_misc import RectangularDomain

# Local
from .testing_utilities import assert_dict_contained


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


def assert_complex_structure_equal(obj1, obj2):
    """Compare two complex structures for equality (must be pickle-able)."""
    try:
        assert pickle.dumps(obj1) == pickle.dumps(obj2)
    except AssertionError:
        err = "Complex structures not equal:\n\n{}\n\n{}\n".format(
            pprint.pformat(obj1, width=1), pprint.pformat(obj2, width=1)
        )
        raise AssertionError(err)


def circle(x, y, radius):
    return list(Point(x, y).buffer(radius).exterior.coords)


def is_sequence(obj):
    # src: http://stackoverflow.com/a/31043360/4419816
    try:
        len(obj)
        obj[0:0]
        return True
    except TypeError:
        return False


def dict2list(dict_):
    """Convert a dictionary recursively to a nested list."""
    list_ = []
    if not isinstance(dict_, dict):
        return dict_
    for key, val in sorted(dict_.items()):
        if isinstance(val, dict):
            val = dict2list(val)
        elif is_sequence(val):
            val = [dict2list(v) for v in val]
        list_.append([key, val])
    return list_


def assert_nested_dict_almost_equal(obj1, obj2, digits=None):
    """Compare two nested dicts for equality. Numbers must be almost-equal."""
    obj1 = dict2list(obj1)
    obj2 = dict2list(obj2)

    def recursive_assert(obj1, obj2, digits):
        if is_sequence(obj1):
            for i, e1 in enumerate(obj1):
                try:
                    e2 = obj2[i]
                except TypeError:
                    e2 = obj2
                recursive_assert(e1, e2, digits)
        else:
            try:
                assert_almost_equal(obj1, obj2, digits)
            except TypeError:
                assert obj1 == obj2

    try:
        recursive_assert(obj1, obj2, digits)
    except AssertionError:
        err = "Complex structures not equal:\n\n{}\n\n{}\n".format(
            pprint.pformat(obj1, width=1), pprint.pformat(obj2, width=1)
        )
        raise AssertionError(err)


def assert_multipolygon_equal(test, mp1, mp2):
    feats1 = list(mp1.geoms)
    feats2 = list(mp2.geoms)
    for f1, f2 in zip(feats1, feats2):
        test.assertEqual(f1.area, f2.area)
        test.assertEqual(f1.bounds, f2.bounds)


def feature_circle(coords, id, ts):
    return FeatureSimple(circle(*coords), id=id, timestep=ts)


def feature_event_circle(coords, id, timestep, etype="continuation"):
    feature = FeatureSimple(circle(*coords), id=id)
    track, prev = None, None
    event = EventSimple(track, feature, timestep=timestep, id=id, etype=etype)
    return [feature, event]


def revert_features_events(ts_max, feature_groups, event_groups):
    reverted = []
    for feature_group, event_group in zip(feature_groups, event_groups):
        events_rev = [revert_event(event, ts_max) for event in event_group]
        reverted.insert(0, [feature_group, events_rev])
    return list(zip(*reverted))


def revert_event(event, ts_max):
    ts_rev = ts_max - event.timestep()
    etype = []
    if event.is_genesis():
        etype.append("end")
        if ts_rev == ts_max:
            etype.append("stop")
        else:
            etype.append("lysis")
    if event.is_lysis() or event.is_stop():
        etype.append("start")
        etype.append("genesis")
    if event.is_merging():
        etype.append("splitting")
    if event.is_splitting():
        etype.append("merging")
    if event.is_continuation():
        etype.append("continuation")
    return EventSimple(
        track=None, feature=event.feature(), timestep=ts_rev, id=event.id(), etype=etype
    )


def group_by_timestep(lst_):
    f = lambda x: x[1].timestep()
    return [list(g) for k, g in itertools.groupby(sorted(lst_, key=f), f)]


# CLASS DEFINITIONS


class FeatureSimple(TrackableFeatureBase):

    cls_periodic = None  # PeriodicFeatureSimple defined below

    def __init__(self, path, **kwargs):
        super_kwargs = {
            key: kwargs.pop(key, val)
            for key, val in (("id", None), ("event", None), ("domain", None))
        }
        timestep = kwargs.pop("timestep", None)
        self._timestep = timestep
        self._poly = Polygon(path)
        super().__init__(**super_kwargs)
        self._set_attrs(kwargs)

    def _set_attrs(self, kwargs):
        self._attrs = []
        for key, val in kwargs.items():
            if hasattr(self, key):
                continue
            setattr(self, key, val)
            self._attrs.append(key)
        return self._attrs

    def __repr__(self):
        attrs = {
            a: getattr(self, a)() if callable(a) else getattr(self, a)
            for a in self._attrs
        }
        attrs = ", ".join(["{}={}".format(k, v) for k, v in attrs.items()])
        if attrs != "":
            attrs = ", " + attrs
        attrs = ""
        try:
            self_id = self.id()
        except AttributeError:
            self_id = id(self)
        return "<{cls}[{id0}]: id={id1}, event={e}{a}>".format(
            cls=self.__class__.__name__,
            id0=id(self),
            id1=self_id,
            e=self.event(),
            a=attrs,
        )

    # Methods that need to be overridden by the subclass

    def copy(self, path=None, id_=None, event=None, domain=None):
        if not path:
            self.as_polygon().boundary.coords
        if not id_:
            id_ = -1
        return self.__class__(path, id=id_, event=event, domain=domain)

    def as_polygon(self):
        return self._poly

    def intersection(self, other):
        return self.as_polygon().intersection(other.as_polygon())

    def area(self):
        return self.as_polygon().area

    def overlap_area(self, other):
        overlap = self.intersection(other)
        try:
            return overlap.area()
        except TypeError:
            return overlap.area

    def center(self):
        x0, y0, x1, y1 = self.as_polygon().bounds
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def radius(self):
        return np.sqrt(self.area() / np.pi)

    def __getattr__(self, attr):
        """Delegate undefined properties to self._poly."""
        try:
            return getattr(self._poly, attr)
        except (AttributeError, RuntimeError):
            err = "Error getting attribute {} from {} object".format(
                attr, self.__class__.__name__
            )
            raise AttributeError(err)


class PeriodicFeatureSimple(PeriodicTrackableFeatureBase, FeatureSimple):
    def __init__(self, features, domain, *args, **kwargs):
        FeatureSimple.__init__(self, None, **kwargs)
        PeriodicTrackableFeatureBase.__init__(self, features, domain)


FeatureSimple.cls_periodic = PeriodicFeatureSimple


class EventSimple:
    def __init__(self, track, feature, *, timestep=None, id=None, etype=None):

        self.feature = lambda: feature
        self.timestep = lambda: timestep
        self.id = lambda: id
        self.next = lambda: []

        self.is_continuation = lambda: False
        self.is_merging = lambda: False
        self.is_splitting = lambda: False
        self.is_start = lambda: False
        self.is_genesis = lambda: False
        self.is_end = lambda: False
        self.is_lysis = lambda: False
        self.is_stop = lambda: False
        self.is_isolated = lambda: False

        if isinstance(etype, str):
            setattr(self, "is_" + etype, lambda: True)
        else:
            for et in etype:
                setattr(self, "is_" + et, lambda: True)

    def __lt__(self, other):
        return self.id() < other.id()

    def __repr__(self):
        attr = ", ".join(
            [str(k) for k, v in inspect.getmembers(self) if k.startswith("is_") and v()]
        )
        attr = "" if attr == "" else ", " + attr
        ts = self.timestep() if self.timestep else "N/A"
        return "<{cls}[{id0}]: id={id1}, ts={ts}{attr}>".format(
            cls=self.__class__.__name__, id0=id(self), id1=self.id(), ts=ts, attr=attr
        )


class TrackerSimple(FeatureTrackerBase):

    # Methods that need to be overridden by the subclass

    def new_track(self, feature, timestep, id=None):
        return TrackSimple(feature=feature, timestep=timestep, id=id)


class TrackSimple(FeatureTrackBase):
    def __hash__(self):
        return id(self)


class AttributeObject:
    def __init__(self, **kwargs):
        self._attrs = {}
        for key, val in kwargs.items():
            self._store_attr(key, val)
            setattr(self, key, val)

    def _store_attr(self, key, val):
        """Store attributes if it's non-callable or takes no arguments.

        Functions which take no arguments can be used to mimick methods
        that simply return a value (e.g. obj.timestep()) using a lambda.
        """
        if callable(val):
            try:
                self._attrs[key] = val()
            except TypeError:
                pass
        else:
            self._attrs[key] = val

    def __repr__(self):
        return "<{cls}: {attrs}>".format(
            cls=self.__class__.__name__,
            attrs=", ".join(["{}={}".format(k, v) for k, v in self._attrs.items()]),
        )


# SR_TODO: Consider a clean implementation of this!
# Hacky way to override class method without subclassing
# (The method is a 1:1 copy from GenericFeatureTrackIOReaderJson)
def rebuild_features_simple(self, data, domain=None):
    features = []
    for feature_data in data:
        cls_str = feature_data.pop("class")
        if domain:
            feature_data["domain"] = domain
        if "feature_path_file" in self._header:
            raise NotImplementedError
        features.append(
            self.cls_feature.factory().from_class_string(cls_str, **feature_data)
        )
    return features


FeatureTrackIOReaderJson.rebuild_features = rebuild_features_simple
FeatureTrackIOReaderJson.cls_feature = FeatureSimple
FeatureTrackIOReaderJson.cls_track = TrackSimple

# TEST TRACKABLE FEATURE

# RELATIONS BETWEEN MULTIPLE OBJECTS


class TestOverlapFraction(TestCase):
    def test_not_overlapping(s):
        """Two non-overlapping features."""
        obj1 = FeatureSimple([(1, 1), (3, 1), (2, 2), (1, 1)])
        obj2 = FeatureSimple([(3, 2), (5, 2), (4, 3), (3, 2)])
        res1 = obj1.overlap_fraction(obj2)
        res2 = obj2.overlap_fraction(obj1)
        assert_almost_equal(res1, 0.0, 3)
        assert_almost_equal(res2, 0.0, 3)

    def test_overlapping_squares(s):
        """Two overlapping squares."""
        obj1 = FeatureSimple([(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)])
        obj2 = FeatureSimple([(2, 2), (5, 2), (5, 4), (2, 4), (2, 2)])
        res1 = obj1.overlap_fraction(obj2)
        res2 = obj2.overlap_fraction(obj1)
        area1, area2, area_overlap = 4, 6, 1
        sol = 2 * 1 / (4 + 6)
        assert_almost_equal(res1, sol, 3)
        assert_almost_equal(res2, sol, 3)


# PERIODIC FEATURES


class TestPeriodicFeature(TestCase):
    def setUp(s):
        s.domain_rect = RectangularDomain(1, 1, 9, 9)
        s.domain_periodic = PeriodicDomain(1, 1, 9, 9)

        s.feats = [
            # Periodic feature
            [(1, 5), (2, 6), (2, 7), (1, 8), (1, 5)],  # west
            [(8, 5), (9, 5), (9, 8), (8, 8), (7, 7), (7, 6), (8, 5)],  # east
            # Non-periodic feature inside the domain
            [(5, 4), (6, 5), (5, 6), (4, 5), (5, 4)],
            # Non-periodic feature at the eastern boundary (along line)
            [(7, 2), (9, 2), (9, 4), (8, 4), (7, 3), (7, 2)],
            # Non-periodic feature at the western boundary (in points)
            [(1, 2), (3, 2), (3, 4), (1, 4), (2, 3), (1, 2)],
        ]

        s.factory_rect = FeatureFactory(
            cls_default=FeatureSimple,
            cls_periodic=PeriodicFeatureSimple,
            domain=s.domain_rect,
        )

        s.factory_periodic = FeatureFactory(
            cls_default=FeatureSimple,
            cls_periodic=PeriodicFeatureSimple,
            domain=s.domain_periodic,
        )

    def test_factory_non_periodic(s):
        features = s.factory_rect.run([{"contour": p} for p in s.feats])
        s.assertEqual(len(features), 5)
        res = set([f.__class__ for f in features])
        sol = {FeatureSimple}
        s.assertSetEqual(res, sol)

    def test_factory_periodic(s):
        features = s.factory_periodic.run([{"contour": p} for p in s.feats])
        s.assertEqual(len(features), 4)
        res = set([f.__class__ for f in features])
        sol = {FeatureSimple, PeriodicFeatureSimple}
        s.assertSetEqual(res, sol)

    def test_center(s):
        features = s.factory_periodic.run([{"contour": p} for p in s.feats[:2]])
        s.assertEqual(len(features), 1)
        feature = features[0]
        assert_almost_equal(feature.center(), (8.5, 6.5))

    def test_center_distance(s):
        features = sorted(
            s.factory_periodic.run([{"contour": p} for p in s.feats]),
            key=lambda f: f.id(),
        )
        s.assertEqual(len(features), 4)
        s.assertFalse(any(f.is_periodic() for f in features[:3]))
        s.assertTrue(features[3].is_periodic())
        b, c, d, a = features
        s.assertAlmostEqual(a.center_distance(b), 3.808, 2)
        s.assertAlmostEqual(a.center_distance(c), 3.536, 2)
        s.assertAlmostEqual(a.center_distance(d), 3.808, 2)
        s.assertAlmostEqual(b.center_distance(c), 3.606, 2)
        s.assertAlmostEqual(b.center_distance(d), 3.606, 2)
        s.assertAlmostEqual(c.center_distance(d), 2.000, 2)


# TEST FEATURE COMBINATION


class TestFeatureCombination(TestCase):
    def setUp(s):
        """Features 1 and 2 don't overlap, but both overlap 3."""
        s.feat1 = FeatureSimple([(1, 1), (3, 1), (3, 2), (1, 2), (1, 1)])
        s.feat2 = FeatureSimple([(4, 2), (6, 2), (6, 5), (4, 5), (4, 2)])
        s.feat3 = FeatureSimple([(2, 1), (5, 1), (5, 4), (2, 4), (2, 1)])
        s.feat12 = FeatureCombination([s.feat1, s.feat2])

    def test_invalid_overlap(s):
        """Features must not overlap."""
        f = lambda: FeatureCombination([s.feat1, s.feat3])
        s.assertRaises(ValueError, f)

    def test_area(s):
        sol = s.feat1.area() + s.feat2.area()
        s.assertEqual(s.feat12.area(), sol)

    def test_center(s):
        """Area-weighted mean center coordinates."""
        f1 = s.feat1.area() / (s.feat1.area() + s.feat2.area())
        f2 = s.feat2.area() / (s.feat1.area() + s.feat2.area())
        sol = f1 * s.feat1.center() + f2 * s.feat2.center()
        assert_array_equal(s.feat12.center(), sol)

    def test_radius(s):
        """Area-equivalent radius."""
        sol = np.sqrt((s.feat1.area() + s.feat2.area()) / np.pi)
        s.assertAlmostEqual(s.feat12.radius(), sol)

    def test_intersection_commutative(s):
        inter1 = s.feat12.intersection(s.feat3)
        inter2 = s.feat3.intersection(s.feat12)
        assert_multipolygon_equal(s, inter1, inter2)

    def test_intersection(s):
        sol = MultiPolygon(
            [
                Polygon([(2, 1), (3, 1), (3, 2), (2, 2), (2, 1)]),
                Polygon([(4, 2), (5, 2), (5, 4), (4, 4), (4, 2)]),
            ]
        )
        res = s.feat12.intersection(s.feat3)
        assert_multipolygon_equal(s, res, sol)


# TEST TRACK EXTENDER


class TestSuccessorProbabilityCalculation(TestCase):
    """Compute the various successor probabilities in isolation."""

    def setUp(s):
        thresholds = {"min_overlap": 0.5, "max_area": 1.5}
        s.tracker = TrackerSimple(**thresholds)
        s.extender = FeatureTrackExtender(s.tracker)

    # Overlap

    def test_overlap_equal(s):
        obj1 = AttributeObject(overlap_fraction=lambda o: 1)
        obj2 = AttributeObject()
        res = s.extender.successor_probability_overlap(obj1, obj2)
        sol = 1.0
        assert_almost_equal(res, sol)

    def test_overlap_smaller_in_range(s):
        obj1 = AttributeObject(overlap_fraction=lambda o: 0.55)
        obj2 = AttributeObject()
        res = s.extender.successor_probability_overlap(obj1, obj2)
        sol = 0.1
        assert_almost_equal(res, sol)

    def test_overlap_smaller_out_of_range(s):
        obj1 = AttributeObject(overlap_fraction=lambda o: 0.45)
        obj2 = AttributeObject()
        res = s.extender.successor_probability_overlap(obj1, obj2)
        sol = 0.0
        assert_almost_equal(res, sol)

    # Area

    def test_a_equal(s):
        obj1 = AttributeObject(area=lambda: 4, area_ratio=lambda o: 1)
        obj2 = AttributeObject(area=lambda: 4)
        res = s.extender.successor_probability_area(obj1, obj2)
        sol = 1.0
        assert_almost_equal(res, sol)

    def test_a_smaller_in_range(s):
        obj1 = AttributeObject(area=lambda: 4, area_ratio=lambda o: 4 / 3)
        obj2 = AttributeObject(area=lambda: 3)
        res = s.extender.successor_probability_area(obj1, obj2)
        sol = 1 / 3
        assert_almost_equal(res, sol)

    def test_a_larger_in_range(s):
        obj1 = AttributeObject(area=lambda: 4, area_ratio=lambda o: 5.5 / 4)
        obj2 = AttributeObject(area=lambda: 5.5)
        res = s.extender.successor_probability_area(obj1, obj2)
        sol = 1 - 1.5 / 2  # 0.25
        assert_almost_equal(res, sol)

    def test_a_smaller_out_of_range(s):
        obj1 = AttributeObject(area=lambda: 4, area_ratio=lambda o: 4 / 1)
        obj2 = AttributeObject(area=lambda: 1)
        res = s.extender.successor_probability_area(obj1, obj2)
        sol = 0.0
        assert_almost_equal(res, sol)

    def test_a_larger_out_of_range(s):
        obj1 = AttributeObject(area=lambda: 4, area_ratio=lambda o: 7 / 4)
        obj2 = AttributeObject(area=lambda: 7)
        res = s.extender.successor_probability_area(obj1, obj2)
        sol = 0.0
        assert_almost_equal(res, sol)


# SUCCESSOR PROBABILITY


class TestSuccessorProbability(TestCase):
    def setUp(s):

        s.thresholds = {"min_overlap": 0.05, "max_area": 1.7}

        # Shortcut function for the weigth arguments
        def kw(overlap, area):
            kwargs = {"f_overlap": overlap, "f_area": area}
            kwargs.update(s.thresholds)
            return kwargs

        s.tracker_overlap = TrackerSimple(**kw(1, 0))
        s.tracker_area = TrackerSimple(**kw(0, 1))
        s.tracker_mixed = TrackerSimple(**kw(1, 1))

        s.extender_overlap = FeatureTrackExtender(s.tracker_overlap)
        s.extender_area = FeatureTrackExtender(s.tracker_area)
        s.extender_mixed = FeatureTrackExtender(s.tracker_mixed)

        # The largest obj1 overlaps both obj2 (smaller) and obj3 (smallest)
        path1 = [(5, 0), (7, 0), (9, 2), (9, 4), (7, 6), (5, 6), (3, 4), (3, 2), (5, 0)]
        path2 = [
            (8, 1),
            (10, 1),
            (11, 2),
            (11, 4),
            (9, 6),
            (7, 6),
            (6, 5),
            (6, 3),
            (8, 1),
        ]
        path3 = [(2, 1), (4, 3), (2, 5), (0, 3), (2, 1)]
        s.obj1 = FeatureSimple(path1)
        s.obj2 = FeatureSimple(path2)
        s.obj3 = FeatureSimple(path3)
        s.a1 = 28
        s.a2 = 20
        s.a3 = 8

        # Overlap areas and fractions between the features
        s.ovlp_a12 = 10
        s.ovlp_a13 = 1
        s.ovlp_a23 = 0
        s.f_ov12 = 2 * s.ovlp_a12 / (s.a1 + s.a2)
        s.f_ov13 = 2 * s.ovlp_a13 / (s.a1 + s.a3)
        s.f_ov32 = 2 * s.ovlp_a23 / (s.a2 + s.a3)

    def p_o(s, frac):
        min_overlap = s.thresholds["min_overlap"]
        return max([frac - min_overlap, 0]) / (1 - min_overlap)

    def p_a(s, area1, area2):
        area_min = min(area1, area2)
        area_max = max(area1, area2)
        max_area = s.thresholds["max_area"]
        return (max_area - min(area_max / area_min, max_area)) / (max_area - 1)

    def test_overlap_only(s):
        res12 = s.extender_overlap.successor_probability(s.obj1, s.obj2)
        res13 = s.extender_overlap.successor_probability(s.obj1, s.obj3)
        res32 = s.extender_overlap.successor_probability(s.obj3, s.obj2)
        sol12 = [s.p_o(s.f_ov12), s.p_o(s.f_ov12), s.p_a(s.a1, s.a2)]
        sol13 = [s.p_o(s.f_ov13), s.p_o(s.f_ov13), s.p_a(s.a1, s.a3)]
        sol32 = [s.p_o(s.f_ov32), s.p_o(s.f_ov32), s.p_a(s.a3, s.a2)]
        assert_almost_equal(res12[:3], sol12)
        assert_almost_equal(res13[:3], sol13)
        assert_almost_equal(res32[:3], sol32)

    def test_area_only(s):
        res12 = s.extender_area.successor_probability(s.obj1, s.obj2)
        res13 = s.extender_area.successor_probability(s.obj1, s.obj3)
        res32 = s.extender_area.successor_probability(s.obj3, s.obj2)
        sol12 = [s.p_a(s.a1, s.a2), s.p_o(s.f_ov12), s.p_a(s.a1, s.a2)]
        sol13 = [s.p_a(s.a1, s.a3), s.p_o(s.f_ov13), s.p_a(s.a1, s.a3)]
        sol32 = [s.p_a(s.a3, s.a2), s.p_o(s.f_ov32), s.p_a(s.a3, s.a2)]
        assert_almost_equal(res12[:3], sol12)
        assert_almost_equal(res13[:3], sol13)
        assert_almost_equal(res32[:3], sol32)

    def test_mixed(s):
        res12 = s.extender_mixed.successor_probability(s.obj1, s.obj2)
        res13 = s.extender_mixed.successor_probability(s.obj1, s.obj3)
        res32 = s.extender_mixed.successor_probability(s.obj3, s.obj2)
        prob12 = (s.p_o(s.f_ov12) + s.p_a(s.a1, s.a2)) / 2
        prob13 = (s.p_o(s.f_ov13) + s.p_a(s.a1, s.a3)) / 2
        prob32 = (s.p_o(s.f_ov32) + s.p_a(s.a3, s.a2)) / 2
        sol12 = [prob12, s.p_o(s.f_ov12), s.p_a(s.a1, s.a2)]
        sol13 = [prob13, s.p_o(s.f_ov13), s.p_a(s.a1, s.a3)]
        sol32 = [prob32, s.p_o(s.f_ov32), s.p_a(s.a3, s.a2)]
        assert_almost_equal(res12[:3], sol12)
        assert_almost_equal(res13[:3], sol13)
        assert_almost_equal(res32[:3], sol32)


# SUCCESSOR CANDIDATES


class TestFindSuccessorCandidates(TestCase):
    """Successor candidates must either overlap or be at most 2*r_eq away."""

    def setUp(s):
        s.tracker = TrackerSimple(min_overlap=0.1, max_area=2.0, max_dist_rel=1.5,)
        s.extender = FeatureTrackExtender(s.tracker)

    def test_overlap(s):
        """Overlap, so candidate in both directions."""
        path1 = [(1, 1), (5, 1), (5, 7), (1, 7), (1, 1)]
        path2 = [(4, 3), (6, 3), (6, 5), (4, 5), (4, 3)]
        obj1 = FeatureSimple(path1, center=lambda: (2, 4))
        obj2 = FeatureSimple(path2, center=lambda: (5, 4))
        s.assertTrue(s.extender.is_successor_candidate(obj1, obj2))
        s.assertTrue(s.extender.is_successor_candidate(obj2, obj1))
        res = s.extender.successor_candidates([obj1], [obj2])
        sol = {obj1: [obj2], obj2: [obj1]}
        s.assertEqual(res, sol)

    def test_no_overlap_true(s):
        """No overlap, but still candidate because near enough in one case."""
        path1 = [(1, 1), (5, 1), (5, 7), (1, 7), (1, 1)]
        path2 = [(6, 3), (8, 3), (8, 5), (6, 5), (6, 3)]
        obj1 = FeatureSimple(path1, center=lambda: (2, 4))
        obj2 = FeatureSimple(path2, center=lambda: (7, 4))
        s.assertTrue(s.extender.is_successor_candidate(obj1, obj2))
        s.assertFalse(s.extender.is_successor_candidate(obj2, obj1))
        res = s.extender.successor_candidates([obj1], [obj2])
        sol = {obj1: [obj2], obj2: []}
        s.assertEqual(res, sol)

    def test_no_overlap_false(s):
        """No overlap, and also too far away to be a candidate."""
        path1 = [(1, 1), (5, 1), (5, 7), (1, 7), (1, 1)]
        path2 = [(7, 3), (9, 3), (9, 5), (7, 5), (7, 3)]
        obj1 = FeatureSimple(path1, center=lambda: (2, 4))
        obj2 = FeatureSimple(path2, center=lambda: (8, 4))
        s.assertFalse(s.extender.is_successor_candidate(obj1, obj2))
        s.assertFalse(s.extender.is_successor_candidate(obj2, obj1))
        res = s.extender.successor_candidates([obj1], [obj2])
        sol = {obj1: [], obj2: []}
        s.assertEqual(res, sol)


# TRACK EVENTS CHAINS


def feat(testcase):
    feature = AttributeObject(link_event=lambda i: None, link_track=lambda i: None)
    testcase.features.append(feature)
    return feature


class TestEventChainStraight(TestCase):
    def setUp(s):

        s.features = []

        factory = FeatureTrackEventFactory()

        s.track = TrackSimple()

        s.chain0 = factory.genesis(s.track, feat(s), 0)
        event0 = factory.continuation(s.track, feat(s), 1)
        event1 = factory.continuation(s.track, feat(s), 2)
        event2 = factory.continuation(s.track, feat(s), 3)
        event3 = factory.lysis(s.track, feat(s), 4)

        s.chain0.link_forward(event0)
        event0.link_forward(event1)
        event1.link_forward(event2)
        event2.link_forward(event3)

        s.chain = [s.chain0]
        s.track._starts = s.chain

    def test_count_events(s):
        res = s.track.n()
        s.assertEqual(res, 5)

    def test_iter_track(s):
        events = list(iter(s.track))
        res = sorted([e.feature() for e in events], key=lambda f: id(f))
        sol = sorted(s.features, key=lambda f: id(f))
        s.assertCountEqual(res, sol)


class TestEventChainMerging(TestCase):
    def setUp(s):

        s.features = []

        factory = FeatureTrackEventFactory()

        s.track = TrackSimple()
        s.chain0 = factory.genesis(s.track, feat(s), 0)
        event0 = factory.continuation(s.track, feat(s), 1)
        event1 = factory.continuation(s.track, feat(s), 2)
        s.chain1 = factory.genesis(s.track, feat(s), 1)
        event2 = factory.continuation(s.track, feat(s), 2)
        event3 = factory.merging(s.track, feat(s), 3)
        event4 = factory.continuation(s.track, feat(s), 4)
        event5 = factory.lysis(s.track, feat(s), 5)

        s.chain0.link_forward(event0)
        event0.link_forward(event1)
        s.chain1.link_forward(event2)
        event0.link_forward(event3)
        event2.link_forward(event3)
        event3.link_forward(event4)
        event4.link_forward(event5)

        s.chain = [s.chain0, s.chain1]
        s.track._starts = s.chain

    def test_count_events(s):
        res = s.track.n()
        s.assertEqual(res, 8)

    def test_iter_track(s):
        events = list(iter(s.track))
        res = sorted([e.feature() for e in events], key=lambda f: id(f))
        sol = sorted(s.features, key=lambda f: id(f))
        s.assertCountEqual(res, sol)


# ------------------------------------------------------------------------------


class TestEventChainSplitting(TestCase):
    def setUp(s):

        s.features = []

        factory = FeatureTrackEventFactory()

        s.track = TrackSimple()
        s.chain0 = factory.genesis(s.track, feat(s), 0)

        event0 = factory.continuation(s.track, feat(s), 1)
        event1 = factory.splitting(s.track, feat(s), 2)
        event2 = factory.continuation(s.track, feat(s), 3)
        event3 = factory.continuation(s.track, feat(s), 4)
        event4 = factory.lysis(s.track, feat(s), 5)
        event5 = factory.continuation(s.track, feat(s), 4)
        event6 = factory.continuation(s.track, feat(s), 5)
        event7 = factory.continuation(s.track, feat(s), 6)
        event8 = factory.lysis(s.track, feat(s), 7)

        s.chain0.link_forward(event0)
        event0.link_forward(event1)
        event1.link_forward(event2)
        event2.link_forward(event3)
        event3.link_forward(event4)
        event1.link_forward(event5)
        event5.link_forward(event6)
        event6.link_forward(event7)
        event7.link_forward(event8)

        s.chain = [s.chain0]
        s.track._starts = s.chain

    def test_count_events(s):
        res = s.track.n()
        s.assertEqual(res, 10)

    def test_iter_track(s):
        events = list(iter(s.track))
        res = sorted([e.feature() for e in events], key=lambda f: id(f))
        sol = sorted(s.features, key=lambda f: id(f))
        s.assertCountEqual(res, sol)


class TestEventChainComplex(TestCase):
    def setUp(s):

        s.features = []

        factory = FeatureTrackEventFactory()

        s.track = TrackSimple()

        gene0 = factory.genesis(s.track, feat(s), 0)
        cont0 = factory.continuation(s.track, feat(s), 1)
        splt0 = factory.splitting(s.track, feat(s), 2)
        cont1 = factory.continuation(s.track, feat(s), 3)
        lysi0 = factory.lysis(s.track, feat(s), 5)
        cont2 = factory.continuation(s.track, feat(s), 3)
        gene1 = factory.genesis(s.track, feat(s), 3)
        cont3 = factory.continuation(s.track, feat(s), 4)
        merg0 = factory.merging(s.track, feat(s), 5)
        cont4 = factory.continuation(s.track, feat(s), 6)
        splt1 = factory.splitting(s.track, feat(s), 7)
        cont5 = factory.continuation(s.track, feat(s), 8)
        lysi1 = factory.lysis(s.track, feat(s), 9)
        cont6 = factory.continuation(s.track, feat(s), 8)
        lysi2 = factory.lysis(s.track, feat(s), 9)

        gene0.link_forward(cont0)
        cont0.link_forward(splt0)
        splt0.link_forward(cont1)
        cont1.link_forward(lysi0)
        splt0.link_forward(cont2)
        gene1.link_forward(cont3)
        cont2.link_forward(merg0)
        cont3.link_forward(merg0)
        merg0.link_forward(cont4)
        cont4.link_forward(splt1)
        splt1.link_forward(cont5)
        cont5.link_forward(lysi1)
        splt1.link_forward(cont6)
        cont6.link_forward(lysi2)

        s.chain = [gene0, gene1]
        s.track._starts = s.chain

    def test_count_events(s):
        res = s.track.n()
        s.assertEqual(res, 15)

    def test_iter_track(s):
        events = list(iter(s.track))
        res = sorted([e.feature() for e in events], key=lambda f: id(f))
        sol = sorted(s.features, key=lambda f: id(f))
        s.assertCountEqual(res, sol)


# SIMPLE TRACK (NO MERGING/SPLITTING)

# SR_TODO User wherever appropriate!
class TestTracks(TestCase):
    """Base class for tracks tests."""

    def assert_tracks_features(s, tracks, features_tracks, ignore_ts=[]):
        """Compare a list of tracks to a list of features.

        The features are those expected in the tracks. Please note that the
        order in which the list of features are passed matters: They must be
        reversely sorted by the number features (i.e. their lengths when
        ignoring any None objects).
        """

        try:
            iter(ignore_ts)
        except TypeError:
            ignore_ts = [ignore_ts]

        tracks.sort(key=lambda track: track.n(), reverse=True)
        s.assertEqual(len(tracks), len(features_tracks))

        for track, features in zip(tracks, features_tracks):
            res, sol = [], []
            feats = [[f for f in ff] for ff in features if any(f for f in ff)]

            for events in track.events_ts():
                features_res_ts = sorted([e.feature() for e in events])
                features_sol_ts = sorted(feats.pop(0))
                if events[0].timestep() in ignore_ts:
                    continue
                res.extend(features_res_ts)
                sol.extend(features_sol_ts)
            try:
                s.assertEqual(sorted(res), sorted(sol))
            except AssertionError:
                err = ("Features don't match:\n\nSolution:\n{}\n\nResult:\n{}").format(
                    "\n".join([str(f) for f in sol]), "\n".join([str(f) for f in res])
                )
                raise AssertionError(err)

    def run_test(s, objs_all, *, event_id=0, tracker=None):
        if not tracker:
            tracker = s.tracker
        # SR_TMP<
        # tracker.set_event_id(event_id)
        FeatureTrackEventBase._next_id = event_id
        # SR_TMP>
        for objs in objs_all:
            tracker.extend_tracks(objs)
        tracker.finish_tracks()
        tracks = tracker.pop_finished_tracks()
        return tracks


class TestBuildTrackSimple(TestTracks):
    """Build simple tracks (i.e. no merging/splitting)."""

    plot = False

    def track1(s):
        return [
            None,
            FeatureSimple(circle(0.9, 0.5, 0.4), id=0),
            FeatureSimple(circle(1.0, 0.8, 0.6), id=1),
            FeatureSimple(circle(1.3, 1.2, 0.8), id=2),
            FeatureSimple(circle(1.6, 1.6, 1.0), id=3),
            FeatureSimple(circle(2.1, 2.0, 1.2), id=4),
            FeatureSimple(circle(2.8, 2.2, 1.3), id=5),
            FeatureSimple(circle(3.6, 2.3, 1.4), id=6),
            FeatureSimple(circle(4.4, 2.3, 1.4), id=7),
            FeatureSimple(circle(5.1, 2.6, 1.3), id=8),
            FeatureSimple(circle(5.6, 3.2, 1.0), id=9),
            FeatureSimple(circle(5.8, 3.6, 0.7), id=10),
            FeatureSimple(circle(5.9, 4.0, 0.5), id=11),
            FeatureSimple(circle(6.0, 4.2, 0.4), id=12),
            None,
        ]

    def track2(s):
        return [
            None,
            None,
            None,
            FeatureSimple(circle(3.9, 4.9, 0.4), id=21),
            FeatureSimple(circle(4.4, 4.8, 0.4), id=22),
            FeatureSimple(circle(4.9, 4.6, 0.5), id=23),
            FeatureSimple(circle(5.4, 4.4, 0.5), id=24),
            FeatureSimple(circle(5.9, 4.1, 0.6), id=25),
            FeatureSimple(circle(6.4, 3.8, 0.7), id=26),
            FeatureSimple(circle(6.9, 3.5, 0.6), id=27),
            FeatureSimple(circle(7.4, 3.2, 0.5), id=28),
            FeatureSimple(circle(7.9, 2.8, 0.5), id=29),
            FeatureSimple(circle(8.2, 2.3, 0.4), id=30),
            None,
            None,
        ]

    def track3(s):
        return [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            FeatureSimple(circle(2.0, 4.0, 0.4), id=13),
            FeatureSimple(circle(2.2, 4.1, 0.4), id=14),
            FeatureSimple(circle(2.4, 4.2, 0.5), id=15),
            FeatureSimple(circle(2.8, 4.4, 0.6), id=16),
            FeatureSimple(circle(3.3, 4.6, 0.8), id=17),
            FeatureSimple(circle(4.0, 5.1, 1.1), id=18),
            FeatureSimple(circle(4.3, 5.6, 1.1), id=19),
            FeatureSimple(circle(4.5, 6.3, 1.2), id=20),
        ]

    def setUp(s):
        s.features12 = lambda: [s.track1(), s.track2()]
        s.features23 = lambda: [s.track2(), s.track3()]
        s.features123 = lambda: [s.track1(), s.track2(), s.track3()]

        s.objs12 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features12()))
        ]

        s.objs23 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features23()))
        ]

        s.objs123 = lambda: [
            [f for f in t if f is not None] for t in list(zip(*s.features123()))
        ]

        if s.plot:
            objs = [o for t in s.objs123() for o in t]
            plot_contours("tracks_simple.png", objs)

        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.00001,
            max_area=2.0,
            max_dist_rel=1.5,
        )

    def test_single_track_1(s):
        features = s.track1()
        features = [[f] if f else [] for f in features]
        tracks = s.run_test(features, event_id=0)
        features = [[f] for f in s.track1()]
        s.assert_tracks_features(tracks, [features])

    def test_single_track_2(s):
        features = s.track2()
        features = [[f] if f else [] for f in features]
        tracks = s.run_test(features, event_id=21)
        features = [[f] for f in s.track2()]
        s.assert_tracks_features(tracks, [features])

    def test_single_track_3(s):
        features = s.track3()
        features = [[f] if f else [] for f in features]
        tracks = s.run_test(features, event_id=13)
        features = [[f] for f in s.track3()]
        s.assert_tracks_features(tracks, [features])

    def test_two_tracks_12(s):
        features = s.objs12()
        tracks = s.run_test(features)
        features = [[[f] for f in ff if f] for ff in s.features12()]
        s.assert_tracks_features(tracks, features)

    def test_two_tracks_23(s):
        s.tracker.set_timestep(2)
        features = s.objs23()
        tracks = s.run_test(features)
        features = [[[f] for f in ff if f] for ff in s.features23()]
        s.assert_tracks_features(tracks, features)

    def test_all_tracks(s):
        features = s.objs123()
        tracks = s.run_test(features)
        features = [[[f] for f in ff if f] for ff in s.features123()]
        s.assert_tracks_features(tracks, features)


class TestMissingFeature(TestTracks):

    plot = False

    def setUp(s):

        s.track1 = [
            FeatureSimple(circle(2.1, 2.0, 1.2), id=0),
            FeatureSimple(circle(2.8, 2.2, 1.3), id=1),
            FeatureSimple(circle(3.6, 2.3, 1.4), id=2),
            FeatureSimple(circle(4.4, 2.3, 1.4), id=3),
            FeatureSimple(circle(5.1, 2.6, 1.3), id=4),
            FeatureSimple(circle(5.6, 3.2, 1.0), id=5),
        ]

        kwargs = dict(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.00001,
            max_area=2.0,
            max_dist_rel=1.5,
        )
        s.tracker0 = TrackerSimple(allow_missing=False, **kwargs)
        s.tracker1 = TrackerSimple(allow_missing=True, **kwargs)

    def test_no_missing(s):
        """Control test with all features."""
        features = [[f] if f else [] for f in s.track1]
        tracks = s.run_test(features, event_id=0, tracker=s.tracker0)
        features = [[f] for f in s.track1]
        s.assert_tracks_features(tracks, [features])

    @pytest.mark.skip("TODO")
    def test_one_missing_split(s):
        """Split the track at the missing feature."""

    def test_one_missing_continue(s):
        """Connect the track across the missing feature."""
        features = [[f] if f else [] for f in s.track1]
        features[3] = []
        tracks = s.run_test(features, event_id=0, tracker=s.tracker1)

        # Check regular features
        s.assertEqual(len(tracks), 1)
        features = [[f] for f in s.track1]
        s.assert_tracks_features(tracks, [features], ignore_ts=3)

        # Check features in mean feature (dummy event)
        sol = [features[2][0], features[4][0]]
        res = tracks[0].events_ts(3)[0].feature().features()
        s.assertEqual(sol, res)

        # Check event types
        ets = [es[0] for es in tracks[0].events_ts()]
        s.assertFalse(ets[0].is_dummy())
        s.assertTrue(ets[0].is_genesis())
        s.assertFalse(ets[1].is_dummy())
        s.assertTrue(ets[1].is_continuation())
        s.assertFalse(ets[2].is_dummy())
        s.assertTrue(ets[2].is_continuation())
        s.assertTrue(ets[3].is_dummy())
        s.assertTrue(ets[3].is_continuation())
        s.assertFalse(ets[4].is_dummy())
        s.assertTrue(ets[4].is_continuation())
        s.assertFalse(ets[5].is_dummy())
        s.assertTrue(ets[5].is_end())

    def test_two_missing_split(s):
        """Track is split because of too many missing features."""
        features = [[f] if f else [] for f in s.track1]
        features[3] = []
        features[4] = []
        tracks = s.run_test(features, event_id=0, tracker=s.tracker1)

        s.assertEqual(len(tracks), 2)
        features = [[[f] for f in ff] for ff in (s.track1[:3], s.track1[5:])]
        s.assert_tracks_features(tracks, features)

        s.assertEqual(s.track1[4].event(), None)


# MERGING/SPLITTING


class TestMergeSplitIsolated(TestCase):
    """Test merging and splitting in isolation."""

    plot = False
    plotted = False

    def setUp(s):

        s.feat0 = [
            feature_circle((2.0, 0.5, 0.6), 0, 0),  # split 0
            feature_circle((3.5, 0.2, 0.4), 1, 0),
        ]  # isolated 0
        s.feat1 = [
            feature_circle((1.8, 0.9, 0.3), 2, 1),  # split 1
            feature_circle((2.5, 0.6, 0.4), 3, 1),  # split 1
            feature_circle((3.5, 0.7, 0.5), 4, 1),
        ]  # isolated 1

        if s.plot and not s.plotted:
            objs = s.feat0 + s.feat1
            plot_contours("tracks_merge-split_isolated.png", objs)
            s.plotted = True

        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.1,
            max_area=2.0,
            max_dist_rel=1.5,
        )

    def test_splitting_isolated(s):
        s.tracker.extend_tracks(s.feat0[0:1])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks(s.feat1[0:2])
        s.assertEqual(len(s.tracker.active_tracks()), 1)

    def test_merging_isolated(s):
        s.tracker.extend_tracks(s.feat1[0:2])
        s.assertEqual(len(s.tracker.active_tracks()), 2)
        s.tracker.extend_tracks(s.feat0[0:1])
        s.assertEqual(len(s.tracker.active_tracks()), 1)

    def test_splitting_full(s):
        s.tracker.extend_tracks(s.feat0)
        s.assertEqual(len(s.tracker.active_tracks()), 2)
        s.tracker.extend_tracks(s.feat1)
        s.assertEqual(len(s.tracker.active_tracks()), 2)

    def test_merging_full(s):
        s.tracker.extend_tracks(s.feat1)
        s.assertEqual(len(s.tracker.active_tracks()), 3)
        s.tracker.extend_tracks(s.feat0)
        s.assertEqual(len(s.tracker.active_tracks()), 2)


class TestMissingFeatureMergeSplit(TestTracks):

    plot = False
    plotted = False

    def setUp(s):

        s.feats = [
            [feature_circle((1.7, 1.0, 0.5), 0, 0)],
            [feature_circle((2.0, 1.3, 0.6), 0, 0)],
            [
                feature_circle((1.8, 1.7, 0.3), 2, 1),
                feature_circle((2.5, 1.3, 0.4), 3, 1),
            ],
        ]

        if s.plot and not s.plotted:
            objs = [f for ts in s.feats for f in ts]
            plot_contours("tracks_missing-feature_merge-split.png", objs)
            s.plotted = True

        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.1,
            max_area=2.0,
            max_dist_rel=1.5,
            allow_missing=True,
        )

    def test_splitting_no_missing(s):
        s.tracker.extend_tracks(s.feats[0])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks(s.feats[1])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks(s.feats[2])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        tracks = s.tracker.active_tracks()
        s.assert_tracks_features(tracks, [s.feats])

    def test_merging_no_missing(s):
        s.tracker.extend_tracks(s.feats[2])
        s.assertEqual(len(s.tracker.active_tracks()), 2)
        s.tracker.extend_tracks(s.feats[1])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks(s.feats[0])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        tracks = s.tracker.active_tracks()
        s.assert_tracks_features(tracks, [s.feats])

    def test_splitting_one_missing_continue(s):
        """Continue the track across the missing feature at a splitting."""
        s.tracker.extend_tracks(s.feats[0])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks([])
        s.assertEqual(len(s.tracker.active_tracks()), 1)
        s.tracker.extend_tracks(s.feats[2])
        s.assertEqual(len(s.tracker.active_tracks()), 1)

        # Check features in track
        tracks = s.tracker.active_tracks()
        s.assert_tracks_features(tracks, [s.feats], ignore_ts=1)

        # Check number of events per timestep
        lengths = [len(events) for events in tracks[0].events_ts()]
        s.assertEqual(lengths, [1, 1, 2])

        # Check event types
        ets = tracks[0].events_ts()
        s.assertFalse(ets[0][0].is_dummy())
        s.assertTrue(ets[0][0].is_genesis())
        s.assertTrue(ets[1][0].is_dummy())
        s.assertTrue(ets[1][0].is_splitting())
        s.assertFalse(ets[1][0].is_continuation())
        s.assertFalse(ets[2][0].is_dummy())
        s.assertTrue(ets[2][0].is_continuation())
        s.assertFalse(ets[2][1].is_dummy())
        s.assertTrue(ets[2][1].is_continuation())

    def test_merging_one_missing_continue(s):
        """Continue the track across the missing feature at a merging."""
        s.tracker.extend_tracks(s.feats[2])
        s.assertEqual(len(s.tracker.active_tracks()), 2)
        s.tracker.extend_tracks([])
        s.assertEqual(len(s.tracker.active_tracks()), 2)
        s.tracker.extend_tracks(s.feats[0])
        s.assertEqual(len(s.tracker.active_tracks()), 1)

        # Check features in track
        tracks = s.tracker.active_tracks()
        s.assert_tracks_features(tracks, [s.feats], ignore_ts=1)

        # Check number of events per timestep
        lengths = [len(events) for events in tracks[0].events_ts()]
        s.assertEqual(lengths, [2, 1, 1])

        # Check event types
        ets = tracks[0].events_ts()
        s.assertFalse(ets[0][0].is_dummy())
        s.assertTrue(ets[0][0].is_genesis())
        s.assertFalse(ets[0][1].is_dummy())
        s.assertTrue(ets[0][1].is_genesis())
        s.assertTrue(ets[1][0].is_dummy())
        s.assertTrue(ets[1][0].is_merging())
        s.assertFalse(ets[1][0].is_continuation())
        s.assertFalse(ets[2][0].is_dummy())
        s.assertTrue(ets[2][0].is_continuation())


class TestMergeSplitComplexSeparate(TestCase):
    """Complex case of multiple mergings/splittings, but only one at a time.

    Per test, there are either only splittings, or only mergings.
    """

    plot = False

    def setUp(s):

        # Note how the track is set up from the perspective of splitting.
        # For the merging tests, the "splitting" events have to be replaced
        # by "merging" events for the comparison with the test results.
        # Accordingly, also the timesteps are reversed.

        # Initial track
        s.seg10 = [
            feature_event_circle((0.0, 0.0, 0.20), 0, 0, "genesis"),
            feature_event_circle((0.3, 0.1, 0.35), 1, 1, "continuation"),
            feature_event_circle((0.7, 0.3, 0.45), 2, 2, "continuation"),
            feature_event_circle((0.9, 0.6, 0.50), 3, 3, "continuation"),
            feature_event_circle((1.1, 0.9, 0.60), 4, 4, "splitting"),
        ]

        # Continuations of splitting of seg10
        s.seg20 = [  # to the left
            feature_event_circle((0.6, 1.3, 0.30), 5, 5, "continuation"),
            feature_event_circle((0.4, 1.6, 0.40), 6, 6, "continuation"),
            feature_event_circle((0.5, 2.0, 0.40), 7, 7, "splitting"),
        ]
        s.seg21 = [  # upward
            feature_event_circle((1.5, 1.3, 0.45), 8, 5, "continuation"),
            feature_event_circle((2.0, 1.6, 0.55), 9, 6, "continuation"),
            feature_event_circle((2.6, 1.6, 0.70), 10, 7, "splitting"),
        ]
        s.seg22 = [  # to the right
            feature_event_circle((1.4, 0.6, 0.25), 11, 5, "continuation"),
            feature_event_circle((1.6, 0.4, 0.20), 12, 6, "continuation"),
            feature_event_circle((1.6, 0.2, 0.10), 13, 7, "lysis"),
        ]

        # Continuations of splitting of seg20
        s.seg30 = [  # to the left
            feature_event_circle((0.2, 2.0, 0.20), 14, 8, "continuation"),
            feature_event_circle((0.0, 2.0, 0.15), 15, 9, "lysis"),
        ]
        s.seg31 = [  # upward
            feature_event_circle((0.7, 2.3, 0.30), 16, 8, "continuation"),
            feature_event_circle((0.7, 2.6, 0.25), 17, 9, "continuation"),
            feature_event_circle((0.6, 2.8, 0.15), 18, 10, "lysis"),
        ]

        # Continuations of splitting of seg21
        s.seg32 = [  # upward
            feature_event_circle((2.6, 2.2, 0.50), 19, 8, "continuation"),
            feature_event_circle((2.9, 2.5, 0.40), 20, 9, "continuation"),
            feature_event_circle((3.2, 2.7, 0.35), 21, 10, "continuation"),
            feature_event_circle((3.5, 2.9, 0.30), 22, 11, "continuation"),
            feature_event_circle((3.7, 3.0, 0.25), 23, 12, "continuation"),
            feature_event_circle((3.9, 3.1, 0.15), 24, 13, "lysis"),
        ]
        s.seg33 = [  # to the right
            feature_event_circle((3.1, 1.3, 0.50), 25, 8, "continuation"),
            feature_event_circle((3.4, 1.1, 0.40), 26, 9, "continuation"),
            feature_event_circle((3.5, 0.8, 0.30), 27, 10, "continuation"),
            feature_event_circle((3.5, 0.6, 0.20), 28, 11, "continuation"),
            feature_event_circle((3.5, 0.5, 0.10), 29, 12, "lysis"),
        ]
        s.seg = (
            s.seg10
            + s.seg20
            + s.seg21
            + s.seg22
            + s.seg30
            + s.seg31
            + s.seg32
            + s.seg33
        )

        # Max. timestep necessary to revert events
        s.ts_max = 13

        # Group all features and events by timestep
        grouped = group_by_timestep(s.seg)
        s.features_split = [[e[0] for e in group] for group in grouped]
        s.events_split = [[e[1] for e in group] for group in grouped]

        # Revert the events
        s.features_merge, s.events_merge = revert_features_events(
            s.ts_max, s.features_split, s.events_split
        )

        # Set up tracker
        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.00001,
            max_area=2.0,
            max_dist_rel=1.5,
        )

        if s.plot:
            features = [x[0] for x in s.seg]
            outfile = "test_merge_split_complex_separate.png"
            plot_contours(outfile, features, labels=lambda i: i.id())

    def run_test(s, feature_groups, event_groups):

        # Run tracking and get tracks
        for features in feature_groups:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        tracks = s.tracker.pop_finished_tracks()

        # Make sure there's only a single track
        s.assertEqual(len(tracks), 1)

        # Check the events for every timestep
        fct = lambda g: set([e.feature() for e in g])
        res = [fct(g) for g in tracks[0].iter_ts()]
        sol = [fct(g) for g in event_groups]
        s.assertEqual(res, sol)

    def test_splitting(s):
        """Full track with both 2-way and 3-way splitting."""
        s.run_test(s.features_split, s.events_split)

    def test_merging(s):
        """Full track with both 2-way and 3-way merging."""
        s.run_test(s.features_merge, s.events_merge)


class TestMergeSplitComplexMixed(TestCase):
    """Complex case with both mergins and splittings in the same case."""

    plot = False

    def track_segments(s):

        fec = feature_event_circle  # abbreviation

        # genesis -> lysis
        s.seg0 = [
            fec((0.0, 0.5, 0.20), 0, 0, ("start", "genesis")),
            fec((0.2, 0.7, 0.30), 1, 1, "continuation"),
            fec((0.5, 1.0, 0.50), 2, 2, "continuation"),
            fec((0.8, 1.3, 0.60), 3, 3, "continuation"),
            fec((1.2, 1.6, 0.70), 4, 4, "continuation"),
            fec((1.5, 1.9, 0.80), 5, 5, "continuation"),
            fec((1.9, 2.2, 0.90), 6, 6, "splitting"),  # 0
            fec((2.5, 2.5, 0.50), 7, 7, "continuation"),
            fec((2.7, 2.8, 0.50), 8, 8, "continuation"),
            fec((2.9, 3.1, 0.50), 9, 9, "continuation"),
            fec((3.2, 3.4, 0.50), 10, 10, "continuation"),
            fec((3.5, 3.7, 0.50), 11, 11, "continuation"),
            fec((3.8, 4.0, 0.55), 12, 12, "continuation"),
            fec((4.1, 4.3, 0.60), 13, 13, "continuation"),
            fec((4.4, 4.6, 0.65), 14, 14, "continuation"),
            fec((4.7, 4.9, 0.70), 15, 15, "continuation"),
            fec((5.1, 5.2, 0.80), 16, 16, "continuation"),
            fec((5.8, 5.7, 1.00), 17, 17, "merging"),  # 0
            fec((6.0, 6.5, 0.70), 18, 18, "continuation"),
            fec((6.0, 7.0, 0.50), 19, 19, "continuation"),
            fec((6.0, 7.4, 0.30), 20, 20, "continuation"),
            fec((6.0, 7.6, 0.20), 21, 21, ("end", "lysis")),
        ]

        # seg0/splitting0 -> lysis
        s.seg1 = [
            fec((1.6, 2.9, 0.40), 22, 7, "continuation"),
            fec((1.6, 3.3, 0.40), 23, 8, "continuation"),
            fec((1.6, 3.7, 0.30), 24, 9, "continuation"),
            fec((1.6, 4.0, 0.20), 25, 10, ("end", "lysis")),
        ]

        # seg0/splitting0 -> seg0/merging0
        s.seg2 = [
            fec((2.4, 1.5, 0.40), 26, 7, "continuation"),
            fec((3.1, 1.5, 0.45), 27, 8, "continuation"),
            fec((3.8, 1.5, 0.50), 28, 9, "continuation"),
            fec((4.6, 0.9, 0.90), 29, 10, "merging"),  # 0
            fec((5.8, 1.0, 0.90), 30, 11, "continuation"),
            fec((6.8, 1.3, 0.90), 31, 12, "splitting"),  # 0
            fec((7.3, 2.4, 0.70), 32, 13, "continuation"),
            fec((7.5, 3.4, 0.70), 33, 14, "continuation"),
            fec((7.3, 4.3, 0.70), 34, 15, "continuation"),
            fec((6.7, 5.1, 0.70), 35, 16, "continuation"),
        ]

        # seg2/splitting0 -> lysis
        s.seg3 = [
            fec((7.4, 1.0, 0.60), 36, 13, "continuation"),
            fec((8.1, 0.9, 0.60), 37, 14, "continuation"),
            fec((8.8, 0.8, 0.60), 38, 15, "continuation"),
            fec((9.4, 0.8, 0.60), 39, 16, "continuation"),
            fec((10.1, 0.9, 0.60), 40, 17, "continuation"),
            fec((11.0, 1.2, 0.80), 41, 18, "merging"),  # 0
            fec((11.4, 1.8, 0.80), 42, 19, "continuation"),
            fec((11.4, 2.5, 0.80), 43, 20, "continuation"),
            fec((11.4, 3.2, 0.80), 44, 21, "continuation"),
            fec((11.4, 3.9, 0.80), 45, 22, "continuation"),
            fec((11.4, 4.6, 0.80), 46, 23, "splitting"),  # 0
            fec((11.0, 5.2, 0.60), 47, 24, "continuation"),
            fec((10.8, 5.8, 0.60), 48, 25, "continuation"),
            fec((10.6, 6.4, 0.60), 49, 26, "continuation"),
            fec((10.5, 7.0, 0.60), 50, 27, "continuation"),
            fec((10.6, 7.6, 0.60), 51, 28, "continuation"),
            fec((11.1, 8.1, 0.60), 52, 29, "continuation"),
            fec((11.6, 8.5, 0.70), 53, 30, ("merging", "splitting")),  # 1/1
            fec((12.1, 9.0, 0.50), 54, 31, "continuation"),
            fec((12.6, 9.4, 0.40), 55, 32, "continuation"),
            fec((13.0, 9.6, 0.30), 56, 33, "continuation"),
            fec((13.3, 9.7, 0.20), 57, 34, ("end", "stop")),
        ]

        # seg3/splitting0 -> seg3/merging1
        s.seg4 = [
            fec((12.1, 4.8, 0.40), 58, 24, "continuation"),
            fec((12.5, 5.3, 0.40), 59, 25, "continuation"),
            fec((12.6, 6.0, 0.40), 60, 26, "continuation"),
            fec((12.6, 6.7, 0.40), 61, 27, "continuation"),
            fec((12.5, 7.4, 0.40), 62, 28, "continuation"),
            fec((12.2, 8.0, 0.40), 63, 29, "continuation"),
        ]

        # seg3/splitting1 -> lysis
        s.seg5 = [
            fec((11.2, 9.1, 0.30), 64, 31, "continuation"),
            fec((11.0, 9.3, 0.25), 65, 32, "continuation"),
            fec((10.7, 9.4, 0.20), 66, 33, ("end", "lysis")),
        ]

        # genesis -> seg3/merging0
        s.seg6 = [
            fec((11.9, 0.0, 0.20), 67, 15, ("start", "genesis")),
            fec((11.5, 0.1, 0.35), 68, 16, "continuation"),
            fec((11.2, 0.4, 0.50), 69, 17, "continuation"),
        ]

        # genesis -> seg2/merging0
        s.seg7 = [
            fec((1.6, -1.0, 0.20), 70, 4, ("start", "genesis")),
            fec((1.9, -1.0, 0.30), 71, 5, "continuation"),
            fec((2.3, -0.9, 0.40), 72, 6, "continuation"),
            fec((2.9, -0.7, 0.50), 73, 7, "continuation"),
            fec((3.5, -0.4, 0.60), 74, 8, "continuation"),
            fec((4.2, 0.2, 0.70), 75, 9, "continuation"),
        ]

        return s.seg0 + s.seg1 + s.seg2 + s.seg3 + s.seg4 + s.seg5 + s.seg6 + s.seg7

    def setUp(s):

        s.seg = s.track_segments()

        # Max. timestep (necessary to revert reference events)
        s.ts_max = 34

        # Group all features and events by timestep
        grouped = group_by_timestep(s.seg)
        s.features_forward = [[e[0] for e in group] for group in grouped]
        s.events_forward = [[e[1] for e in group] for group in grouped]

        # Revert the events
        s.features_backward, s.events_backward = revert_features_events(
            s.ts_max, s.features_forward, s.events_forward
        )

        # Set up tracker
        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.00001,
            max_area=2.0,
            max_dist_rel=1.5,
        )
        s.tracker.reset_ids()

        if s.plot:
            features = [x[0] for x in s.seg]
            outfile = "test_merge_split_complex_mixed.png"
            plot_contours(outfile, features, labels=lambda f: "{}".format(f.id()))

    def run_test(s, feature_groups, event_groups_sol):

        # Run tracking and get tracks
        for features in feature_groups:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        tracks = s.tracker.pop_finished_tracks()

        # Make sure there's only a single track
        s.assertEqual(len(tracks), 1)

        # Get events for every timestep
        event_groups_res = [events for events in tracks[0].iter_ts()]
        s.assertEqual(len(event_groups_res), len(event_groups_sol))

        def assert_event_type(res, sol):
            attrs = [
                "is_continuation",
                "is_merging",
                "is_splitting",
                "is_start",
                "is_genesis",
                "is_end",
                "is_lysis",
                "is_stop",
                "is_isolated",
            ]
            for attr in attrs:
                try:
                    s.assertEqual(getattr(res, attr)(), getattr(sol, attr)())
                except AssertionError:
                    err = "Event differ in '{}':\n{}\n{}".format(attr, res, sol)
                    raise AssertionError(err)

        # Compare events
        f = lambda e: e.feature().id()
        for events_res, events_sol in zip(event_groups_res, event_groups_sol):
            tmp = zip(sorted(events_res, key=f), sorted(events_sol, key=f))
            for event_res, event_sol in tmp:
                assert_event_type(event_res, event_sol)

        # Compare features for every timestep
        fct = lambda g: set([e.feature() for e in g])
        res = [fct(g) for g in event_groups_res]
        sol = [fct(g) for g in event_groups_sol]
        s.assertEqual(res, sol)

    def test_forward(s):
        s.run_test(s.features_forward, s.events_forward)

    def test_backward(s):
        s.run_test(s.features_backward, s.events_backward)

    def test_forward_track_merging(s):
        """Test correct event assignment in case of complex track merging.

        When a previously split track merges with another track, all events
        of the former must be assigned to the single new track.

        In this case, the initial track #0 is first split into three branches.
        A parallel track #1 then merges with one of the branches to create a
        new track #2. In the end, all events must be assigned to this track.
        """
        feature_groups = s.features_forward[:12]
        for features in feature_groups:
            s.tracker.extend_tracks(features)
        s.assertEqual(s.tracker.n_active(), 1)
        track = s.tracker.active_tracks()[0]
        events = list(iter(track))
        res = set([event.track().id() for event in events])
        s.assertSetEqual(res, {2})

    def test_duration(s):
        for features in s.features_forward:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        track = s.tracker.pop_finished_tracks()[0]
        s.assertEqual(track.duration(), 34)

    # SR_TODO: Test seems kinda hidden here...
    # SR_TODO: Need to reorganize the whole test collection at some point!
    def test_linear_segments(s):
        """Return all linear segments of the track (as branch objects)."""
        for features in s.features_forward:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        track = s.tracker.pop_finished_tracks()[0]
        segments = track.linear_segments()

        # Information about segments: (id, prev, next, features)
        # Note that the ID is not equal to the actual segment IDs, but only
        # used to reference the segments, which are identified in the test
        # by the IDs of the features they contain.
        seg_info = {
            0: [(), (1, 2, 3), (0, 1, 2, 3, 4, 5, 6)],
            1: [(0,), (), (22, 23, 24, 25)],
            2: [(0,), (7,), (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)],
            3: [(0,), (5,), (26, 27, 28)],
            4: [(), (5,), (70, 71, 72, 73, 74, 75)],
            5: [(3, 4), (6, 8), (29, 30, 31)],
            6: [(5,), (7,), (32, 33, 34, 35)],
            7: [(2, 6), (), (17, 18, 19, 20, 21)],
            8: [(5,), (10,), (36, 37, 38, 39, 40)],
            9: [(), (10,), (67, 68, 69)],
            10: [(8, 9), (11, 12), (41, 42, 43, 44, 45, 46)],
            11: [(10,), (13,), (47, 48, 49, 50, 51, 52)],
            12: [(10,), (13,), (58, 59, 60, 61, 62, 63)],
            13: [(11, 12), (14, 15), (53,)],
            14: [(13,), (), (64, 65, 66)],
            15: [(13,), (), (54, 55, 56, 57)],
        }

        s.assert_segments(segments, seg_info)

    def test_linear_segments_non_unique_branchings(s):
        """Return all linear segments of the track (as branch objects)."""
        for features in s.features_forward:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        track = s.tracker.pop_finished_tracks()[0]
        segments = track.linear_segments(unique_branchings=False)

        # Information about segments: (id, prev, next, features)
        # Note that the ID is not equal to the actual segment IDs, but only
        # used to reference the segments, which are identified in the test
        # by the IDs of the features they contain.
        seg_info = {
            0: [(), (1, 2, 3), (0, 1, 2, 3, 4, 5, 6)],
            1: [(0,), (), (6, 22, 23, 24, 25)],
            2: [(0,), (7,), (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)],
            3: [(0,), (5,), (6, 26, 27, 28, 29)],
            4: [(), (5,), (70, 71, 72, 73, 74, 75, 29)],
            5: [(3, 4), (6, 8), (29, 30, 31)],
            6: [(5,), (7,), (31, 32, 33, 34, 35, 17)],
            7: [(2, 6), (), (17, 18, 19, 20, 21)],
            8: [(5,), (10,), (31, 36, 37, 38, 39, 40, 41)],
            9: [(), (10,), (67, 68, 69, 41)],
            10: [(8, 9), (11, 12), (41, 42, 43, 44, 45, 46)],
            11: [(10,), (13,), (46, 47, 48, 49, 50, 51, 52, 53)],
            12: [(10,), (13,), (46, 58, 59, 60, 61, 62, 63, 53)],
            13: [(11, 12), (14, 15), (53,)],
            14: [(13,), (), (53, 64, 65, 66)],
            15: [(13,), (), (53, 54, 55, 56, 57)],
        }

        s.assert_segments(segments, seg_info)

    def assert_segments(s, segments, seg_info):

        # Check features
        res = set(tuple([f.id() for f in seg.features()]) for seg in segments)
        sol = set(i[-1] for i in seg_info.values())
        s.assertSetEqual(res, sol)

        # Function to get the local segment ID from the features
        feat2sid = lambda seg: [
            i
            for i, (p, n, fs) in seg_info.items()
            if tuple(f.id() for f in seg.features()) == fs
        ][0]

        # Check links between segments
        for segment in segments:
            sid = feat2sid(segment)
            sids_prev = {feat2sid(seg) for seg in segment.prev()}
            sids_next = {feat2sid(seg) for seg in segment.next()}
            try:
                s.assertSetEqual(sids_prev, set(seg_info[sid][0]))
                s.assertSetEqual(sids_next, set(seg_info[sid][1]))
            except AssertionError as e:
                err = (
                    "Incorrectly linked segment {} ({})"
                    "\n expected: {:>10} <-> {:<10}"
                    "\n observed: {:>10} <-> {:<10}"
                    "\n\nERROR: {}"
                ).format(
                    sid,
                    "/".join([str(i) for i in seg_info[sid][-1]]),
                    ", ".join([str(i) for i in seg_info[sid][0]]),
                    ", ".join([str(i) for i in seg_info[sid][1]]),
                    ", ".join([str(i) for i in sids_prev]),
                    ", ".join([str(i) for i in sids_next]),
                    e,
                )
                raise AssertionError(err)


# TRACK POST-PROCESSING

fc = lambda coords, id, ts: feature_circle(coords, id=id, ts=ts)


class TestRemoveStubsBase(TestCase):

    plot = False
    plotted = {}

    def setUp(s):
        s.tracker = s.setup_tracker()
        s.init_features_ts()
        s.make_plot(s.outfile, s.features)
        s.track = s.run_tracking(s.features_ts)

    def setup_tracker(self):
        return TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.01,
            max_area=2.0,
            max_dist_rel=-1,
            max_dist_abs=-1,
        )

    def init_features_ts(self):
        self.features_ts = []
        n = -1
        for i, xyr_ts in enumerate(self.feature_coords_ts):
            self.features_ts.append([])
            for xyr in xyr_ts:
                n += 1
                self.features_ts[-1].append(fc(xyr, n, i))
        self.features = [f for lvl in self.features_ts for f in lvl]

    def run_tracking(self, features_ts):
        self.tracker.reset_ids()
        FeatureTrackEventBase._next_id = 0
        # ipython(globals(), locals())
        for features in features_ts:
            self.tracker.extend_tracks(features)
        self.tracker.finish_tracks()
        tracks = self.tracker.pop_finished_tracks()
        self.assertEqual(len(tracks), 1)
        return tracks[0]

    def make_plot(self, outfile, features):
        if not self.__class__.plot or self.__class__.plotted.get(outfile, False):
            return
        self.__class__.plotted[outfile] = True
        plot_contours(
            outfile, features, labels=lambda f: "{}/{}".format(f.id(), f._timestep)
        )

    def assert_features(self, track, sol):

        # Check features
        res = sorted([feature.id() for feature in track.features()])
        self.assertEqual(res, sol)

        # Check events by timestep
        res = sorted([e.feature().id() for t in track.events_ts() for e in t])
        self.assertEqual(res, sol)

    def assert_successors(self, track, test_data):
        events = {event.feature().id(): event for event in track.events()}
        msg = lambda d, i, n: (
            "\n\nWrong number of '{d}' events found for F"
            "{i} event ({nf} instead of {ne}):\n{e}"
        ).format(
            d=d,
            i=i,
            ne=n,
            nf=len(getattr(events[i], d)()),
            e="\n".join(
                [
                    " F{:<3}: {}".format(e.feature().id(), e.__class__.__name__)
                    for e in getattr(events[i], d)()
                ]
            ),
        )
        for i, nprev, nnext in test_data:
            try:
                event = events[i]
            except KeyError:
                err = "Not enough events for index {}:\n{}".format(
                    i, "\n".join([str(n) + " " + str(e) for n, e in events.items()])
                )
                raise AssertionError(err)
            self.assertEqual(len(event.prev()), nprev, msg("prev", i, nprev))
            self.assertEqual(len(event.next()), nnext, msg("next", i, nnext))

    def assert_event_types(self, track, test_data):
        events = {event.feature().id(): event for event in track.events()}
        msg = lambda i, t: "F{} event is {}: {}".format(
            i, t, events[i].__class__.__name__
        )
        for i, types_is, types_not in test_data:
            try:
                event = events[i]
            except KeyError:
                err = "Not enough events for index {}: {}".format(i, pformat(events))
                raise AssertionError(err)
            for t in types_is:
                self.assertTrue(getattr(event, "is_" + t)(), msg(i, "not " + t))
            for t in types_not:
                self.assertFalse(getattr(event, "is_" + t)(), msg(i, t))


class TestRemoveStubs(TestRemoveStubsBase):
    """Remove insignificant branches from complex tracks.

    Basic tests without special cases such as mixed events.
    """

    def setUp(s):

        s.feature_coords_ts = [
            [(0.2, 0.1, 0.3), (2.4, 0.0, 0.4)],
            [(0.5, 0.8, 0.6), (2.3, 0.8, 0.5), (0.7, 2.0, 0.4)],
            [(1.6, 1.5, 0.9)],
            [(2.6, 2.6, 1.1)],
            [(3.9, 2.8, 1.1), (4.5, 4.2, 0.3)],
            [(5.2, 3.2, 1.2)],
            [(6.3, 3.9, 1.1), (5.7, 2.2, 0.3)],
            [(6.4, 5.3, 0.9), (7.4, 2.8, 0.9)],
            [(7.4, 5.8, 0.5), (6.0, 6.5, 0.7), (8.1, 1.8, 0.7)],
            [(7.8, 6.3, 0.4), (5.2, 6.9, 0.5), (8.3, 0.9, 0.4)],
            [(8.0, 6.7, 0.3), (8.4, 0.4, 0.2)],
            [(8.3, 7.0, 0.2)],
        ]
        s.outfile = "track_stubs.png"

        super().setUp()

        s.ids_all = list(range(22 + 1))
        s.ids_del1 = [4, 8, 11]
        s.ids_del2 = [1, 3, 4, 8, 11, 15, 18]

    def test_raw_track(s):
        events_by_fid = {e.feature().id(): e for e in s.track.events()}
        msg = lambda i: "[{}] {}".format(i, events_by_fid[i].__class__.__name__)
        for i in [0, 1, 4, 8]:
            s.assertTrue(events_by_fid[i].is_start(), msg(i))
        for i in [18, 21, 22]:
            s.assertTrue(events_by_fid[i].is_end(), msg(i))
        for i in [5, 9]:
            s.assertTrue(events_by_fid[i].is_merging(), msg(i))
        for i in [9, 10, 12]:
            s.assertTrue(events_by_fid[i].is_splitting(), msg(i))
        for i in [2, 3, 6, 7, 13, 16, 19, 14, 15, 17, 20]:
            s.assertTrue(events_by_fid[i].is_continuation(), msg(i))

    def test_features_unlinked_n2(s):
        """Check that the removed features are properly unlinked (n=2)."""
        s.track.remove_stubs(2)

        s.features = [f for lvl in s.features_ts for f in lvl]

        # Check that retained features are still linked to events
        res0 = [(f.id(), f.event()) for f in s.features if f.id() not in s.ids_del2]
        res1 = [(f, bool(e)) for f, e in res0]
        sol = [(f.id(), True) for f in s.features if f.id() not in s.ids_del2]
        err = (
            "\n\nSome retained features not linked to events anymore:\n\n"
            " ID | EVENT\n{}"
        ).format("\n".join([" {:>2} | {}".format(f, str(e)) for f, e in res0]))
        s.assertEqual(res1, sol, err)

        # Check that removed features are not linked to events anymore
        res = [(f.id(), str(f.event())) for f in s.features if f.id() in s.ids_del2]
        sol = [(f.id(), str(None)) for f in s.features if f.id() in s.ids_del2]
        err = (
            "\n\nSome removed features still linked to events:\n\n" " ID | EVENT\n{}"
        ).format("\n".join([" {:>2} | {}".format(f, str(e)) for f, e in res]))
        s.assertEqual(sorted(res), sol, err)

    def test_features_n1(s):
        """Check the features which the track contains (n=1)."""
        s.track.remove_stubs(1)
        s.assert_features(s.track, [i for i in s.ids_all if i not in s.ids_del1])

    def test_features_n2(s):
        """Check the features which the track contains (n=2)."""
        s.track.remove_stubs(2)
        s.assert_features(s.track, [i for i in s.ids_all if i not in s.ids_del2])

    def test_events_unlinked_n2(s):
        """Check that removed events are unlinked from retained events."""
        s.track.remove_stubs(2)
        s.assert_successors(s.track, [(5, 1, 1), (9, 1, 1), (10, 1, 2), (12, 1, 1)])

    def test_event_types_adapted_n2(s):
        """Check that eliminated mergings/splittings have the right type.

        If all but one branches (in one direction) have been eliminated,
        the event is no longer a merging/splitting event, but (usually)
        only a continuation.
        """
        s.track.remove_stubs(2)
        s.assert_event_types(
            s.track,
            [
                [5, ["continuation"], ["merging"]],
                [9, ["continuation"], ["merging", "splitting"]],
                [12, ["continuation"], ["splitting"]],
            ],
        )


class TestRemoveStubsSpecialSingleEndpoint(TestRemoveStubsBase):
    """
    Track starting with a GenesisSplitting and immediate merging one
    timestep later, i.e. with a pair of n=2 stubs sharing the end point.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(1.30, 1.0, 0.25)],
            [(1.65, 0.9, 0.20), (1.5, 1.2, 0.1)],
            [(1.90, 1.3, 0.40)],
            [(2.40, 1.6, 0.50)],
            [(2.80, 2.0, 0.40)],
            [(2.90, 2.4, 0.20)],
        ]
        s.outfile = "track_stubs_special_single_endpoint.png"
        super().setUp()

    def test_features_n2(s):
        """Two n=2 stubs end in a MergingLysis/Genesis event."""
        s.track.remove_stubs(2)
        s.assert_features(s.track, [0, 1, 3, 4, 5, 6])


class TestRemoveStubsSpecialSingleEndpoint2(TestRemoveStubsBase):
    """
    Add another feature at the stub end, making them n=3 stubs,
    and also invert the track, turning the splitting into a merging
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(2.90, 2.4, 0.20)],
            [(2.80, 2.0, 0.40)],
            [(2.40, 1.6, 0.50)],
            [(1.90, 1.3, 0.40)],
            [(1.65, 0.9, 0.20), (1.5, 1.2, 0.1)],
            [(1.30, 1.0, 0.25)],
            [(1.05, 1.0, 0.15)],
        ]
        s.outfile = "track_stubs_special_single_endpoint2.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 6, 7])


class TestRemoveStubsSpecialMultiBranch(TestRemoveStubsBase):
    """
    Splitting event with many successors with max. length 3,
    i.e. all of them are potential stubs (all removed for n=3)
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(1.0, 1.0, 1.0)],
            [(1.8, 1.8, 1.0)],
            [
                (1.0, 2.5, 0.2),
                (1.4, 2.8, 0.2),
                (2.2, 2.8, 0.4),
                (2.0, 2.0, 0.2),
                (2.8, 2.0, 0.3),
                (2.5, 1.0, 0.4),
            ],
            [(2.5, 3.2, 0.4), (3.1, 2.0, 0.2), (2.8, 0.8, 0.3)],
            [(2.8, 3.5, 0.3), (3.0, 0.6, 0.2)],
            [],
        ]
        s.outfile = "track_stubs_special_multibranch.png"
        super().setUp()

    def test_features_n1(s):
        s.track.remove_stubs(1)
        s.assert_features(s.track, [0, 1, 4, 6, 7, 8, 9, 10, 11, 12])

    def test_features_n2(s):
        s.track.remove_stubs(2)
        s.assert_features(s.track, [0, 1, 4, 7, 8, 10, 11, 12])

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 4, 8, 11])


class TestRemoveStubsSpecialDoubleStub(TestRemoveStubsBase):
    """
    A n=2 stub ends in a merging event with a n=1 stub,
    i.e. both should be removed for n=2 without messing things up
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(1.8, 4.0, 0.4)],
            [(2.4, 4.0, 0.5)],
            [(3.2, 4.0, 0.6)],
            [(4.1, 4.0, 0.8)],
            [(5.0, 4.3, 0.6), (4.6, 3.3, 0.3), (4.6, 2.6, 0.3)],
            [(5.7, 4.6, 0.6), (4.9, 2.8, 0.4)],
            [(6.2, 5.1, 0.5)],
            [(6.5, 5.6, 0.4)],
        ]
        s.outfile = "track_stubs_special_doublestub.png"
        super().setUp()

    def test_features_n1(s):
        s.track.remove_stubs(1)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])

    def test_features_n2(s):
        s.track.remove_stubs(2)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 7, 9, 10])


class TestRemoveStubsSpecialMergeStub(TestRemoveStubsBase):
    """
    Two long tracks merge and the new track ends one timestep later.
    Nothing is truncated, though, as that would make the track shorter.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(2.1, 1.7, 0.3), (5.4, 0.6, 0.4), (4.5, 0.5, 0.3)],
            [(2.3, 2.0, 0.2), (5.0, 1.0, 0.6)],
            [(2.5, 2.3, 0.3), (4.5, 1.5, 0.5)],
            [(2.8, 2.6, 0.2), (4.0, 2.0, 0.6), (2.2, 2.4, 0.1)],
            [(3.0, 2.7, 0.1), (3.5, 2.5, 0.3), (4.5, 2.5, 0.4)],
            [(3.3, 2.9, 0.4), (4.9, 2.8, 0.4)],
            [(3.3, 3.3, 0.1), (5.3, 3.1, 0.4)],
        ]
        s.outfile = "track_stubs_special_mergestub.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        # SR_TODO: Think about branch 12/14/16 (retained because it contains an
        # SR_TODO: event at timestep 6, but it's not the only event at that)
        # s.assert_features(s.track, [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15])
        s.assert_features(s.track, [0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])


class TestRemoveStubsSpecialOnlyStubs(TestRemoveStubsBase):
    """
    A complex merging/splitting only has stubs attached (max. length 2).
    The "most significant" forward-stub ends in a merging/lysis.
    This case is is based on a real case (PV Cut-offs).
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(7.0, 3.0, 0.8)],
            [(6.0, 4.0, 0.8), (4.0, 4.0, 0.8)],
            [(5.0, 5.0, 1.4)],
            [(4.0, 5.9, 0.3), (4.8, 6.4, 0.4), (6.5, 5.7, 0.8)],
            [(4.0, 6.6, 0.5)],
        ]
        s.outfile = "track_stubs_special_onlystubs.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 3, 5, 7])


class TestRemoveStubsSpecialDoubleSplit(TestRemoveStubsBase):
    """
    A split leads to two branches with three features each, but while
    one branch is linear, the other consists of another splitting and
    two ends. Even though the latter is larger in total area, the former
    has precedence because it's longer (for n>2, that is).
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(5.0, 1.2, 0.5)],
            [(5.0, 1.9, 0.6)],
            [(5.0, 2.7, 0.7)],
            [(5.0, 3.5, 0.8)],
            [(4.5, 4.2, 0.3), (5.5, 4.3, 0.6)],
            [(4.2, 4.5, 0.2), (5.3, 4.9, 0.4), (6.2, 4.5, 0.5)],
            [(4.0, 4.6, 0.1)],
        ]
        s.outfile = "track_stubs_special_doublesplit.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 6, 9])


class TestRemoveStubsSpecialOldStub(TestRemoveStubsBase):
    """
    Three branches merge near the beginning of a track. Two are stubs,
    but one of them goes farther back in time then the non-stub branch,
    and is thus not removed (n=3).
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(3.0, 1.4, 0.2)],
            [(3.1, 1.7, 0.2), (4.2, 0.6, 0.6), (6.0, 1.0, 0.8)],
            [
                (3.3, 1.9, 0.2),
                (4.1, 1.3, 0.5),
                (5.3, 1.5, 0.6),
                (6.5, 1.7, 0.3),
                (7.0, 1.1, 0.4),
                (3.6, 0.5, 0.3),
            ],
            [(4.3, 2.2, 1.0), (7.2, 1.8, 0.6), (3.3, 0.5, 0.2)],
            [(4.3, 3.0, 0.8), (7.7, 2.4, 0.7)],
            [(4.1, 3.8, 0.5), (7.3, 3.1, 0.6), (8.4, 2.4, 0.5)],
            [(6.9, 3.7, 0.7), (9.0, 2.2, 0.4)],
            [(6.3, 4.3, 0.6), (7.4, 4.3, 0.3), (9.2, 1.8, 0.3)],
            [(5.7, 4.4, 0.5), (6.5, 4.9, 0.3), (7.7, 4.5, 0.2)],
            [(5.2, 4.5, 0.4), (6.3, 5.2, 0.3)],
            [(4.9, 4.9, 0.4), (6.0, 5.5, 0.3)],
            [(4.8, 5.3, 0.4), (5.7, 5.8, 0.3)],
            [(4.7, 5.8, 0.5), (5.5, 6.1, 0.3)],
            [(4.9, 6.3, 0.6)],
            [(4.3, 6.5, 0.2), (5.0, 6.9, 0.5)],
            [(4.0, 6.3, 0.2), (5.2, 7.4, 0.4)],
            [(5.4, 7.8, 0.3)],
            [(5.6, 8.1, 0.2)],
        ]
        s.outfile = "track_stubs_special_oldstub.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(
            s.track,
            [i for i in range(41) if i not in (9, 12, 17, 19, 22, 21, 25, 35, 37)],
        )


class TestRemoveStubsSpecialStartSplit(TestRemoveStubsBase):
    """
    TODO: write description
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(2.0, 1.0, 0.9)],
            [(3.0, 1.6, 0.7), (1.6, 1.9, 0.5), (3.8, 3.8, 0.3)],
            [
                (3.8, 1.6, 0.4),
                (1.6, 2.6, 0.5),
                (2.8, 2.4, 0.5),
                (3.4, 3.8, 0.4),
                (1.2, 3.6, 0.3),
                (2.2, 4.7, 0.3),
            ],
            [(4.4, 1.6, 0.4), (2.3, 3.6, 1.0)],
            [(5.0, 1.6, 0.4)],
            [(5.6, 1.6, 0.4)],
        ]
        s.outfile = "track_stubs_special_startsplit.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 2, 4, 5, 6, 10, 11, 12, 13])


class TestRemoveStubsSpecialManyLysis(TestRemoveStubsBase):
    """Short track with only a single genesis but many lysis events.

    Based on a PV cut-off track that triggered an infinite recursion-crash.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(2.0, 1.8, 0.8)],
            [(1.5, 2.5, 0.4), (2.6, 2.5, 0.6)],
            [(2.3, 3.1, 0.2), (3.1, 3.0, 0.5)],
            [(3.6, 3.3, 0.3)],
            [(3.9, 3.4, 0.2), (3.5, 3.6, 0.1)],
        ]
        s.outfile = "track_stubs_special_manylysis.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 2, 4, 5, 6])


class TestRemoveStubsSpecialMergingLysisComplex(TestRemoveStubsBase):
    """Track ends in merging-lysis of 3 branches from different splittings.

    Two of the branches ending in the merging-lysis are n=2-stubs, the third
    is a n=3-stub. Based on a PV cut-off track where the merging-lysis was
    eliminated, which should not happen.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(3.0, 1.0, 1.0)],
            [(3.0, 2.3, 0.9), (4.0, 1.0, 0.3)],
            [(3.3, 3.4, 0.7), (2.1, 2.4, 0.4), (3.9, 2.3, 0.3)],
            [(2.7, 4.1, 0.5), (2.6, 3.1, 0.4), (1.9, 2.9, 0.3)],
            [(1.8, 3.8, 0.8)],
        ]
        s.outfile = "test_remove_stubs_special_merging_lysis_complex.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 3, 6, 9])

    def test_successors_n3(s):
        s.track.remove_stubs(3)
        s.assert_successors(s.track, [(0, 0, 1), (1, 1, 1), (3, 1, 1), (9, 1, 0)])

    def test_events_n3(s):
        s.track.remove_stubs(3)
        s.assert_event_types(
            s.track,
            [
                [0, ["genesis"], ["splitting"]],
                [1, ["continuation"], ["splitting"]],
                [3, ["continuation"], ["splitting"]],
                [9, ["stop"], ["merging"]],
            ],
        )


class TestRemoveStubsSpecialMergingMergingLysis(TestRemoveStubsBase):
    """
    A merging of a regular branch and a stub isfollowed by a merging/lysis,
    where the other branch is another stub. The first merging was erroneously
    removed (n=3), leaving the lysis event isolated.
    """

    plot = False

    def setUp(s):
        s.feature_coords_ts = [
            [(2.0, 1.0, 0.6)],
            [(2.0, 1.8, 0.7)],
            [(2.0, 2.7, 0.8)],
            [(2.0, 3.9, 0.8), (3.5, 4.5, 0.5)],
            [(2.5, 5.0, 1.0), (4.1, 5.9, 0.5)],
            [(3.0, 6.2, 1.1)],
        ]
        s.outfile = "test_remove_stubs_special_merging_merging_lysis.png"
        super().setUp()

    def test_features_n3(s):
        s.track.remove_stubs(3)
        s.assert_features(s.track, [0, 1, 2, 3, 5, 7])

    def test_successors_n3(s):
        s.track.remove_stubs(3)
        s.assert_successors(s.track, [(5, 1, 1), (7, 1, 0)])

    def test_events_n3(s):
        s.track.remove_stubs(3)
        s.assert_event_types(
            s.track,
            [[5, ["continuation"], ["merging", "lysis"]], [7, ["end"], ["merging"]]],
        )


class TestRemoveShortSplits(TestRemoveStubsBase):
    """Only retain the most significant path of short splits.

    Often, a path splits into multiple features, only to merge again one or
    a few timesteps later. Discard all but the most significant one of such
    paths up to a certain length (e.g. n=2 means splitting, two continuations,
    and re-merging at the third step).
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(1.0, 2.0, 0.6)],
            [(2.0, 2.0, 0.8)],
            [(2.8, 1.7, 0.6), (2.8, 2.7, 0.3)],
            [(3.5, 2.2, 0.7)],
            [(4.5, 2.3, 0.8)],
            [(5.2, 1.7, 0.4), (5.4, 2.4, 0.2), (5.1, 3.0, 0.3)],
            [(5.8, 1.8, 0.4), (5.7, 2.5, 0.2), (5.6, 3.1, 0.4)],
            [(6.3, 1.9, 0.4), (6.2, 2.9, 0.5)],
            [(7.1, 2.5, 0.8)],
            [(7.9, 2.5, 0.6)],
        ]
        s.outfile = "track_minisplits.png"
        super().setUp()

    def test_features_n1(s):
        s.track.remove_short_splits(1)
        s.assert_features(s.track, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    def test_features_n2(s):
        s.track.remove_short_splits(2)
        s.assert_features(s.track, [0, 1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15])

    def test_events_unlinked_n2(s):
        s.track.remove_short_splits(2)
        s.assert_successors(s.track, [(1, 1, 1), (4, 1, 1), (5, 1, 2), (11, 1, 1)])

    def test_event_types_adapted_n2(s):
        s.track.remove_short_splits(2)
        s.assert_event_types(
            s.track,
            [
                [1, ["continuation"], ["splitting"]],
                [4, ["continuation"], ["merging"]],
                [11, ["continuation"], ["merging"]],
            ],
        )

    def test_features_n3(s):
        s.track.remove_short_splits(3)
        s.assert_features(s.track, [0, 1, 2, 4, 5, 8, 11, 13, 14, 15])


class TestRemoveShortSplitsSpecialSplit1(TestRemoveStubsBase):
    """Special cases of split branches.

    Splitting into two branches and re-merging four steps later.
    Another branch is split off of the "more significant" branch,
    i.e. the one which is retained (for n>3).
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(0.8, 3.0, 0.6)],
            [(1.7, 3.0, 0.7)],
            [(2.6, 3.0, 0.8)],
            [(3.4, 2.5, 0.4), (3.4, 3.5, 0.5)],
            [(4.0, 2.3, 0.4), (4.0, 3.7, 0.5)],
            [(4.6, 2.5, 0.4), (4.6, 3.6, 0.4), (4.3, 4.2, 0.2)],
            [(5.2, 3.0, 0.8), (4.4, 4.4, 0.2)],
            [(6.1, 3.0, 0.7), (4.6, 4.6, 0.2)],
            [(7.0, 3.0, 0.6), (4.8, 4.7, 0.2)],
        ]
        s.outfile = "track_splits_split1.png"
        super().setUp()

    def test_n4(s):
        s.track.remove_short_splits(4)
        s.assert_features(s.track, [0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15])
        s.assert_successors(s.track, [(2, 1, 1), (6, 1, 2), (10, 1, 1)])
        s.assert_event_types(
            s.track,
            [
                [2, ["continuation"], ["splitting"]],
                [6, ["splitting"], ["continuation"]],
                [10, ["continuation"], ["merging"]],
            ],
        )


class TestRemoveShortSplitsSpecialSplit2(TestRemoveStubsBase):
    """Special cases of split branches.

    Splitting into two branches and re-merging four steps later.
    Another branch is split off of the "less significant" branch,
    i.e. the one which is removed (for n>3). Only the part of the
    branch after the splitting event is removed.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(0.8, 3.0, 0.6)],
            [(1.7, 3.0, 0.7)],
            [(2.6, 3.0, 0.8)],
            [(3.4, 2.5, 0.4), (3.4, 3.5, 0.5)],
            [(4.0, 2.3, 0.4), (4.0, 3.7, 0.5)],
            [(4.5, 2.4, 0.3), (4.6, 3.5, 0.5), (4.2, 1.9, 0.2)],
            [(5.2, 3.0, 0.8), (4.4, 1.7, 0.2)],
            [(6.1, 3.0, 0.7), (4.6, 1.6, 0.2)],
            [(7.0, 3.0, 0.6), (4.8, 1.5, 0.2)],
        ]
        s.outfile = "track_splits_split2.png"
        super().setUp()

    def test_n4(s):
        s.track.remove_short_splits(4)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
        s.assert_successors(s.track, [(2, 1, 2), (5, 1, 1), (10, 1, 1)])
        s.assert_event_types(
            s.track,
            [
                [2, ["splitting"], ["continuation"]],
                [5, ["continuation"], ["splitting"]],
                [10, ["continuation"], ["merging"]],
            ],
        )


class TestRemoveShortSplitsSpecialMerge1(TestRemoveStubsBase):
    """Special cases of split branches.

    Like features_split1, but with a merging instead of a splitting.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(0.8, 3.0, 0.6), (3.4, 4.7, 0.2)],
            [(1.7, 3.0, 0.7), (3.6, 4.6, 0.2)],
            [(2.6, 3.0, 0.8), (3.8, 4.4, 0.2)],
            [(3.4, 2.5, 0.4), (3.4, 3.6, 0.4), (3.9, 4.2, 0.2)],
            [(4.0, 2.3, 0.4), (4.0, 3.7, 0.5)],
            [(4.6, 2.5, 0.4), (4.6, 3.5, 0.5)],
            [(5.2, 3.0, 0.8)],
            [(6.1, 3.0, 0.7)],
            [(7.0, 3.0, 0.6)],
        ]
        s.outfile = "track_splits_merge1.png"
        super().setUp()

    def test_n4(s):
        s.track.remove_short_splits(4)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15])
        s.assert_successors(s.track, [(4, 1, 1), (10, 2, 1), (13, 1, 1)])
        s.assert_event_types(
            s.track,
            [
                [4, ["continuation"], ["splitting"]],
                [10, ["merging"], ["continuation"]],
                [13, ["continuation"], ["merging"]],
            ],
        )


class TestRemoveShortSplitsSpecialMerge2(TestRemoveStubsBase):
    """Special cases of split branches.

    Like features_split2, but with a merging instead of a splitting.
    Only the part of the "less significant" branch after the merging
    event is retained.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(0.8, 3.0, 0.6), (3.3, 1.5, 0.2)],
            [(1.7, 3.0, 0.7), (3.5, 1.6, 0.2)],
            [(2.6, 3.0, 0.8), (3.7, 1.7, 0.2)],
            [(3.4, 2.5, 0.3), (3.4, 3.5, 0.5), (3.8, 1.9, 0.2)],
            [(3.9, 2.3, 0.4), (4.0, 3.7, 0.5)],
            [(4.5, 2.4, 0.4), (4.6, 3.5, 0.5)],
            [(5.2, 3.0, 0.8)],
            [(6.1, 3.0, 0.7)],
            [(7.0, 3.0, 0.6)],
        ]
        s.outfile = "track_splits_merge2.png"
        super().setUp()

    def test_n4(s):
        s.track.remove_short_splits(4)
        s.assert_features(s.track, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        s.assert_successors(s.track, [(4, 1, 1), (9, 1, 1), (13, 2, 1)])
        s.assert_event_types(
            s.track,
            [
                [4, ["continuation"], ["splitting"]],
                [9, ["continuation"], ["merging"]],
                [13, ["merging"], ["continuation"]],
            ],
        )


class TestRemoveShortSplitsSpecialNonStubMerge(TestRemoveStubsBase):
    """
    A pair of n=1 split stubs run parallel to a n=1 split stub, but a
    non-stub branch merges together with the n=1 split stubs. This turns
    the n=2 stub automatically into the lesser branch.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(3.9, 1.0, 0.6)],
            [(3.6, 1.5, 0.7)],
            [(3.3, 2.2, 0.8), (1.0, 4.1, 0.3)],
            [(3.1, 3.0, 0.9), (1.1, 4.5, 0.3)],
            [(3.0, 4.0, 1.0), (1.3, 4.8, 0.3)],
            [(2.0, 4.3, 0.4), (2.6, 4.9, 0.3), (3.6, 4.6, 0.6), (1.5, 5.0, 0.3)],
            [(2.2, 5.1, 0.6), (3.5, 5.3, 0.6)],
            [(2.9, 5.9, 0.9)],
            [(3.0, 6.7, 0.9)],
            [(3.2, 7.5, 0.8)],
            [(3.5, 8.1, 0.7)],
            [(3.9, 8.5, 0.6)],
        ]
        s.outfile = "track_splits_non_stub_merge.png"
        super().setUp()

    def test_n3(s):
        s.track.remove_short_splits(3)
        s.assert_features(
            s.track, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        )
        s.assert_successors(s.track, [(6, 1, 1), (12, 1, 1), (14, 2, 1)])
        s.assert_event_types(
            s.track,
            [
                [6, ["continuation"], ["splitting"]],
                [12, ["continuation"], ["merging"]],
                [14, ["merging"], []],
            ],
        )


class TestRemoveShortSplitsSpecialMergeSplit(TestRemoveStubsBase):
    """A n=1 short split originates from a merge/split, just after a splitting.

    This case originates from GenericFeature data, where the short split candidate
    branches originated from the splitting, but were both considered
    too significant to remove because they both contain the merging/splitting,
    where a significant branch merges in.
    """

    def setUp(s):
        s.feature_coords_ts = [
            [(3.0, 0.5, 0.7), (6.4, 3.4, 0.3)],
            [(3.0, 1.2, 0.8), (6.0, 3.2, 0.4)],
            [(3.0, 2.2, 0.9), (5.4, 3.2, 0.5)],
            [(2.4, 2.9, 0.7), (3.8, 3.0, 0.6), (4.9, 3.6, 0.5)],
            [(2.1, 3.5, 0.6), (4.0, 3.9, 0.9)],
            [(1.7, 3.9, 0.5), (3.3, 4.6, 0.6), (4.5, 4.7, 0.5)],
            [(1.3, 4.2, 0.4), (3.9, 5.1, 0.8)],
            [(3.8, 5.9, 0.7)],
            [(3.6, 6.6, 0.6)],
        ]
        s.outfile = "track_splits_merge_split.png"
        super().setUp()

    def test_n3(s):
        s.track.remove_short_splits(3)
        s.assert_features(
            s.track, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]
        )
        s.assert_successors(s.track, [(10, 2, 1), (15, 1, 1)])
        s.assert_event_types(
            s.track,
            [[10, ["merging"], ["splitting"]], [15, ["continuation"], ["merging"]]],
        )


class TestRemoveShortSplitsSpecialMergeSplit2(TestRemoveStubsBase):
    """An n=1 split ends in a merge/split."""

    def setUp(s):
        s.feature_coords_ts = [
            [(3.0, 3.0, 0.9)],
            [(2.4, 4.0, 0.5), (3.7, 4.0, 0.6)],
            [(3.0, 5.0, 0.9)],
            [(2.4, 6.0, 0.5), (3.7, 6.0, 0.6)],
        ]
        s.outfile = "track_splits_merge_split2.png"
        super().setUp()

    def test_n2(s):
        s.track.remove_short_splits(2)
        s.assert_features(s.track, [0, 2, 3, 4, 5])
        s.assert_successors(s.track, [(0, 0, 1), (3, 1, 2)])
        s.assert_event_types(
            s.track, [[0, ["genesis"], ["splitting"]], [3, ["splitting"], ["merging"]]]
        )


# PERIODIC DOMAINS

# PERIODIC TRACKS


class TestPeriodicTrack(TestCase):
    def setUp(s):

        s.paths_ts = [
            [((4, 5), (5, 5), (6, 6), (6, 7), (5, 8), (4, 8), (3, 7), (3, 6), (4, 5))],
            [((6, 5), (7, 5), (8, 6), (8, 7), (7, 8), (6, 8), (5, 7), (5, 6), (6, 5))],
            [
                ((8, 5), (9, 5), (9, 8), (8, 8), (7, 7), (7, 6), (8, 5)),
                ((0, 5), (1, 6), (1, 7), (0, 8), (0, 5)),
            ],
            [((1, 5), (2, 5), (3, 6), (3, 7), (2, 8), (1, 8), (0, 7), (0, 6), (1, 5))],
            [((3, 5), (4, 5), (5, 6), (5, 7), (4, 8), (3, 8), (2, 7), (2, 6), (3, 5))],
        ]

        # Create features
        s.factory = FeatureFactory(
            cls_default=FeatureSimple,
            cls_periodic=PeriodicFeatureSimple,
            domain=PeriodicDomain(0, 0, 9, 9),
        )

        # Set up tracker
        s.tracker = TrackerSimple(
            f_overlap=0.5,
            f_area=0.5,
            threshold=0.0,
            min_overlap=0.00001,
            max_area=2.0,
            max_dist_rel=1.5,
        )

    def test_simple_track(s):
        for paths in s.paths_ts:
            features = s.factory.run([{"contour": p} for p in paths])
            s.tracker.extend_tracks(features)
        s.assertEqual(s.tracker.n_active(), 1)
        s.assertEqual(s.tracker.n_finished(), 0)
        s.tracker.finish_tracks()
        track = s.tracker.pop_finished_tracks()[0]
        res = [event.feature().center() for event in track]
        sol = [[i + 0.5, 6.5] for i in [4, 6, 8, 1, 3]]
        assert_almost_equal(res, sol)


# IO


class TestTrackIO(TestCase):
    def setUp(s):

        fs = lambda coords, id: FeatureSimple(circle(*coords), id=id)

        FTE = "FeatureTrackEvent"
        FTEG, FTEC, FTES = FTE + "Genesis", FTE + "Continuation", FTE + "Stop"

        s.features_straight = [
            [fs((0.0, 0.5, 0.20), 0)],
            [fs((0.2, 0.7, 0.30), 1)],
            [fs((0.5, 1.0, 0.50), 2)],
            [fs((0.8, 1.3, 0.30), 3)],
            [fs((1.0, 1.5, 0.20), 4)],
        ]
        path = lambda i: s.features_straight[i][0].path()
        s.jdat_straight = {
            "EVENTS": [
                {
                    "class": FTEG,
                    "id": 0,
                    "track": 0,
                    "feature": 0,
                    "timestep": 0,
                    "prev": [],
                    "next": [1],
                },
                {
                    "class": FTEC,
                    "id": 1,
                    "track": 0,
                    "feature": 1,
                    "timestep": 1,
                    "prev": [0],
                    "next": [2],
                },
                {
                    "class": FTEC,
                    "id": 2,
                    "track": 0,
                    "feature": 2,
                    "timestep": 2,
                    "prev": [1],
                    "next": [3],
                },
                {
                    "class": FTEC,
                    "id": 3,
                    "track": 0,
                    "feature": 3,
                    "timestep": 3,
                    "prev": [2],
                    "next": [4],
                },
                {
                    "class": FTES,
                    "id": 4,
                    "track": 0,
                    "feature": 4,
                    "timestep": 4,
                    "prev": [3],
                    "next": [],
                },
            ],
            "FEATURES": [
                {"class": "FeatureSimple", "id": 0, "path": path(0)},
                {"class": "FeatureSimple", "id": 1, "path": path(1)},
                {"class": "FeatureSimple", "id": 2, "path": path(2)},
                {"class": "FeatureSimple", "id": 3, "path": path(3)},
                {"class": "FeatureSimple", "id": 4, "path": path(4)},
            ],
            "TRACKS": [
                {"id": 0, "starts": [0], "ends": [4], "events": [0, 1, 2, 3, 4],}
            ],
        }

        s.config = {
            "TRACKER": {
                "f_overlap": 0.5,
                "f_area": 0.5,
                "threshold": 0.0,
                "min_overlap": 0.00001,
                "max_area": 2.0,
                "max_dist_rel": 1.5,
            },
        }
        s.tracker = TrackerSimple(**s.config["TRACKER"])

        s.writer = FeatureTrackIOWriterJson()
        s.reader = FeatureTrackIOReaderJson()

    def run_tracking(s, features_ts):
        s.tracker.reset_ids()
        for features in features_ts:
            s.tracker.extend_tracks(features)
        s.tracker.finish_tracks()
        return s.tracker.pop_finished_tracks()

    def test_write_string_config(s):
        s.writer.add_config(s.config)
        jstr = s.writer.write_string()
        jdat = json.loads(jstr)["CONFIG"]
        assert_dict_contained(jdat["TRACKER"], s.config["TRACKER"])

    def test_write_string_straight_track(s):
        """Write the full track in one go."""
        tracks = s.run_tracking(s.features_straight)
        s.assertEqual(len(tracks), 1)
        s.writer.add_tracks(tracks)
        jstr = s.writer.write_string()
        res = json.loads(jstr)
        sol = s.jdat_straight
        assert_dict_contained(res, sol)

    def test_write_string_straight_features(s):
        """Only write the features (separately for each timestep)."""
        tracks = s.run_tracking(s.features_straight)
        s.assertEqual(len(tracks), 1)
        features = [event.feature() for event in tracks[0]]
        for feature in features:
            s.writer.add_features([feature])
        jstr = s.writer.write_string()
        res = json.loads(jstr)["FEATURES"]
        sol = s.jdat_straight["FEATURES"]
        assert_dict_contained(res, sol)

    def test_read_string_straight(s):
        jstr = json.dumps(s.jdat_straight)
        objs = s.reader.read_string(jstr)
        res = sorted(objs["FEATURES"])
        sol = sorted([i[0] for i in s.features_straight])
        s.assertEqual(res, sol)
        s.assertEqual(len(objs["TRACKS"]), 1)
        res = objs["TRACKS"][0]
        log.debug("===================")
        sol = s.run_tracking(s.features_straight)[0]
        s.assertEqual(res, sol)


if __name__ == "__main__":
    unittest.main()

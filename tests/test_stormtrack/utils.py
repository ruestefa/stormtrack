#!/usr/bin/env python3

import sys
import logging as log
from unittest import TestCase
from pprint import pprint
from pprint import pformat

import numpy as np

# from IPython.terminal.embed import embed; embed()  # SR_DBG
from stormtrack.core.typedefs import default_constants
from stormtrack.core.identification import Feature


# Test Features


class TestFeatures_Base(TestCase):
    def setUp(s):
        pass

    def setUpFeatures(self, *names):
        self.nxy = self.fld.shape
        if not hasattr(self, "fid"):
            self.fid = 0
        for name in names:
            self.initialize_features(name)

    def initialize_features(self, suffix):
        if not hasattr(self, "inds_features_{}".format(suffix)):
            return

        # Prepare indices of pixels, shells, and holes
        # SR_TMP<
        try:
            obj_ids = getattr(self, "obj_ids_{}".format(suffix))
        except AttributeError:
            obj_ids = getattr(self, "obj_ids_{}".format(suffix.replace("out", "in")))
        # SR_TMP>
        inds_features = getattr(self, "inds_features_{}".format(suffix))
        if hasattr(self, "shells_{}".format(suffix)):
            shells_lst = getattr(self, "shells_{}".format(suffix))
            if len(shells_lst) != len(inds_features):
                raise Exception(
                    "{}: wrong number of shells: {} != {}".format(
                        suffix, len(shells_lst), len(inds_features)
                    )
                )
        else:
            shells_lst = [None] * len(inds_features)
        if hasattr(self, "holes_{}".format(suffix)):
            holes_lst = getattr(self, "holes_{}".format(suffix))
            if len(holes_lst) != len(inds_features):
                raise Exception(
                    "{}: wrong number of holes: {} != {}".format(
                        suffix, len(holes_lst), len(inds_features)
                    )
                )
        else:
            holes_lst = [None] * len(inds_features)

        # Create features
        features = []
        for inds, shells, holes in zip(inds_features, shells_lst, holes_lst):
            pixels = self.select_pixels(obj_ids, inds)
            connectivity = {"_4c": 4, "_8c": 8}.get(suffix[-3:])
            feature = self.create_feature(
                pixels=pixels,
                id_=self.fid,
                shells=shells,
                holes=holes,
                connectivity=connectivity,
            )
            self.fid += 1
            feature._shared_boundary_pixels = None
            feature._shared_boundary_pixels_unique = None
            features.append(feature)

        # Add neighbors
        if hasattr(self, "inds_neighbors_{}".format(suffix)):
            inds_neighbors = getattr(self, "inds_neighbors_{}".format(suffix))
            for i, feature in enumerate(features):
                feature.neighbors = [features[j] for j in inds_neighbors[i]]

        # Add pixels shared with neighbors/background
        attr_name = "shared_boundary_pixels_{}".format(suffix)
        if hasattr(self, attr_name):
            attr = getattr(self, attr_name)
            assert len(attr) == len(features)
            for i, feature in enumerate(features):
                assert len(attr[i]) == len(feature.neighbors) + 2
                feature._shared_boundary_pixels = {}
                target = feature._shared_boundary_pixels
                for j, neighbor in enumerate(feature.neighbors):
                    pixels = np.array(attr[i][j], dtype=np.int32)
                    target[neighbor] = pixels
                target["bg"] = np.array(attr[i][-2], dtype=np.int32)
                target["in"] = np.array(attr[i][-1], dtype=np.int32)

        # Add pixels uniquely shared with neighbors/background
        attr_name = "shared_boundary_pixels_unique_{}".format(suffix)
        if hasattr(self, attr_name):
            attr = getattr(self, attr_name)
            assert len(attr) == len(features)
            for i, feature in enumerate(features):
                assert len(attr[i]) == len(feature.neighbors) + 2
                feature._shared_boundary_pixels_unique = {}
                target = feature._shared_boundary_pixels_unique
                for j, neighbor in enumerate(feature.neighbors):
                    pixels = np.array(attr[i][j], dtype=np.int32)
                    target[neighbor] = pixels
                target["bg"] = np.array(attr[i][-2], dtype=np.int32)
                target["in"] = np.array(attr[i][-1], dtype=np.int32)

        # Add features to test class
        setattr(self, "features_{}".format(suffix), features)

    def create_feature(self, pixels, id_, shells=None, holes=None, connectivity=None):
        values = self.fld[(pixels.T[0], pixels.T[1])]
        dummy_center = [0, 0]
        dummy_extrema = [[0, 0]]

        def arr(lst, dt=np.int32):
            if lst is None:
                lst = []
            return np.asarray(lst, dtype=dt)

        if shells is not None:
            shells = [arr(shell) for shell in shells]
        if holes is not None:
            holes = [arr(hole) for hole in holes]
        feature = Feature(
            values=arr(values, np.float32),
            pixels=arr(pixels),
            shells=shells,
            holes=holes,
            id_=id_,
        )
        if shells is None or holes is None:
            if connectivity is None:
                raise ValueError("no connectivity but shells or holes is None")
            nx, ny = self.nxy
            const = default_constants(nx=nx, ny=ny, connectivity=connectivity)
            if (shells, holes) is (None, None):
                feature.derive_boundaries_from_pixels(const)
            elif shells is None:
                feature.derive_shells_from_pixels(const)
            elif holes is None:
                feature.derive_holes_from_pixels(const)
        return feature

    def select_pixels(self, fld, inds_regions):
        return np.array(
            [
                (x, y)
                for inds in inds_regions
                for x, y in np.array(np.where(fld == inds)).T
            ]
        )

    # SR_TODO Split up into multiple assert functions (keep this one, though)
    def assertFeaturesEqual(
        self,
        features1,
        features2,
        msg=None,
        check_boundaries=True,
        check_neighbors=True,
        check_shared_pixels=True,
        sort_by_size=False,
    ):
        if sort_by_size:
            features1.sort(key=lambda f: tuple(f.pixels.min(axis=0)))
            features2.sort(key=lambda f: tuple(f.pixels.min(axis=0)))

        # Check number and size of features
        self.assertEqual(len(features1), len(features2), msg)
        lens = lambda fs: set([f.n for f in fs])
        try:
            self.assertSetEqual(lens(features1), lens(features2), msg)
        except AssertionError as e:
            err = "Features differ in size:\n"
            sizes1 = [(f.id, f.n) for f in features1]
            sizes2 = [(f.id, f.n) for f in features2]
            for i, ((fid1, n1), (fid2, n2)) in enumerate(zip(sizes1, sizes2)):
                if n1 != n2:
                    err += "({}) [{:2}] {:4} {:4} [{:2}]\n".format(
                        i, fid1, n1, n2, fid2
                    )
            err += "\nPixels in one but not the other:\n"
            pxs1 = [(f.id, [(x, y) for x, y in f.pixels]) for f in features1]
            pxs2 = [(f.id, [(x, y) for x, y in f.pixels]) for f in features2]
            for i, ((fid1, px1), (fid2, px2)) in enumerate(zip(pxs1, pxs2)):
                diff1 = [p for p in px1 if p not in px2]
                diff2 = [p for p in px2 if p not in px1]
                if not diff1 and not diff2:
                    continue
                err += "\n({}) [{:^6}] [{:^6}]\n".format(i, fid1, fid2)
                for j in range(max([len(diff1), len(diff2)])):
                    tmpl = "({:2}, {:2})"
                    xy1 = "-" if j >= len(diff1) else tmpl.format(*diff1[j])
                    xy2 = "-" if j >= len(diff2) else tmpl.format(*diff2[j])
                    err += "    {:^8} {:^8}\n".format(xy1, xy2)
            raise AssertionError(err) from None

        # Check pixels
        pxset = lambda f: set([(x, y) for x, y in f.pixels])
        pixels1 = sorted([pxset(f) for f in features1], key=lambda i: len(i))
        pixels2 = sorted([pxset(f) for f in features2], key=lambda i: len(i))
        for px1, px2 in zip(pixels1, pixels2):
            self.assertSetEqual(px1, px2, msg)

        # Check shells and holes
        if check_boundaries:
            for feature1, feature2 in zip(features1, features2):

                shells1 = sorted(feature1.shells, key=lambda h: h.min(axis=0))
                shells2 = sorted(feature2.shells, key=lambda h: h.min(axis=0))
                for shell1, shell2 in zip(shells1, shells2):
                    assertListShiftedEqual(self, shell1, shell2, "check shells")

                holes1 = sorted(feature1.holes, key=lambda h: h.min(axis=0))
                holes2 = sorted(feature2.holes, key=lambda h: h.min(axis=0))
                for hole1, hole2 in zip(holes1, holes2):
                    assertListShiftedEqual(self, hole1, hole2, "check holes")

        # Check neighbors (IDs don't have to match)
        if check_neighbors:
            for feature1, feature2 in zip(features1, features2):
                neighbors1 = feature1.neighbors
                neighbors2 = feature2.neighbors
                msg = (
                    "features {} and {} differ in neighbors:\n" "[{}] v. [{}]"
                ).format(
                    feature1.id,
                    feature2.id,
                    ", ".join(["{:2}".format(f.id) for f in neighbors1]),
                    ", ".join(["{:2}".format(f.id) for f in neighbors2]),
                )
                self.assertFeaturesEqual(
                    neighbors1,
                    neighbors2,
                    msg=msg,
                    check_neighbors=False,
                    check_boundaries=False,
                    check_shared_pixels=False,
                )

        # Check shared pixels
        if check_shared_pixels:
            if not check_neighbors:
                err = "no shared pixels check possible without neighbors"
                raise Exception(err)
            for feature1, feature2 in zip(features1, features2):
                for mode in ["complete", "unique"]:
                    self.assertFeaturesEqual_SharedPixels(feature1, feature2, mode)

    def assertFeaturesEqual_SharedPixels(self, feature1, feature2, mode):

        # Check background boundary pixels
        pixels1 = feature1.shared_boundary_pixels("bg", mode)
        pixels2 = feature2.shared_boundary_pixels("bg", mode)
        err = ("number of shared_boundary_pixels({}, {}) differs").format("bg", mode)
        self.assertEqual(len(pixels1), len(pixels2), err)
        pixels1 = set([(x, y) for x, y in pixels1])
        pixels2 = set([(x, y) for x, y in pixels2])
        err = "shared_boundary_pixels differ ({}, {})".format("bg", mode)
        self.assertSetEqual(pixels1, pixels2, err)

        # Check interior boundary pixels ("complete" mode)
        # (only relevant for 4-connectivity)
        pixels1 = feature1.shared_boundary_pixels("in", mode)
        pixels2 = feature2.shared_boundary_pixels("in", mode)
        err = ("number of shared_boundary_pixels({}, {}) differs").format("in", mode)
        self.assertEqual(len(pixels1), len(pixels2), err)
        pixels1 = set([(x, y) for x, y in pixels1])
        pixels2 = set([(x, y) for x, y in pixels2])
        err = "shared_boundary_pixels({}, {}) differ".format("in", mode)
        self.assertSetEqual(pixels1, pixels2, err)

        # Check neighbors
        neighbors1 = sorted(feature1.neighbors, key=lambda f: f.n)
        neighbors2 = sorted(feature2.neighbors, key=lambda f: f.n)
        for neighbor1, neighbor2 in zip(neighbors1, neighbors2):
            pixels1 = feature1.shared_boundary_pixels(neighbor1, mode)
            pixels2 = feature2.shared_boundary_pixels(neighbor2, mode)
            err = ("number of shared_boundary_pixels({}/{}, {}) " "differs").format(
                neighbor1.id, neighbor2.id, mode
            )
            self.assertEqual(len(pixels1), len(pixels2), err)
            pixels1 = set([(x, y) for x, y in pixels1])
            pixels2 = set([(x, y) for x, y in pixels2])
            err = ("shared_boundary_pixels({}/{}, {}) differ").format(
                neighbor1.id, neighbor2.id, mode
            )
            self.assertSetEqual(pixels1, pixels2, err)


def assertListShiftedEqual(self, list1, list2, msg=None):
    """Check that the elements of two list are in the same relative order.

    The elements can be shifted and even in reverse order.

    If the first and last elements of the lists are identical, one is removed.
    """
    if msg is None:
        msg = ""
    else:
        msg = "{}: ".format(msg)

    # SR_TODO use ndarray asserts
    self.assertEqual(len(list1), len(list2), "{}lists differ in length".format(msg))
    try:
        # SR_TMP
        # self.assertEqual(list1[0], list1[-1], "edge elements of list 1 differ")
        # self.assertEqual(list2[0], list2[-1], "edge elements of list 2 differ")
        self.assertEqual(
            list(list1[0]),
            list(list1[-1]),
            "{}edge elements of list 1 differ".format(msg),
        )
        self.assertEqual(
            list(list2[0]),
            list(list2[-1]),
            "{}edge elements of list 2 differ".format(msg),
        )
    except AssertionError:
        pass
    else:
        list1 = list1[:-1]
        list2 = list2[:-1]
    # SR_TMP
    # self.assertSetEqual(set(list1), set(list2), "list elements differ")
    self.assertSetEqual(
        set([tuple(i) for i in list1]),
        set([tuple(i) for i in list2]),
        "{}list elements differ".format(msg),
    )
    list1 = [i for i in list1]
    for i in range(2 * len(list1)):
        try:
            # SR_TMP
            # self.assertEqual(list1, list2)
            self.assertEqual(
                [list(i) for i in list1],
                [list(i) for i in list2],
                "{}list elements differ".format(msg),
            )
        except AssertionError:
            if i + 1 == len(list1):
                list1 = list1[::-1]
            else:
                list1.append(list1.pop(0))
        else:
            return
    else:
        raise AssertionError("{}list element order differs".format(msg))


def assertBoundaries(self, inds_lst1, inds_lst2, name="inds"):
    """Chech that two sets of boundaries (shells, holes) are the same."""

    self.assertEqual(len(inds_lst1), len(inds_lst2))

    def sort_inds_lst(inds_lst):
        inds_lst = sorted(inds_lst, key=lambda i: min(i[:, 0]))
        inds_lst = sorted(inds_lst, key=lambda i: max(i[:, 1]), reverse=True)
        inds_lst = sorted(inds_lst, key=lambda i: len(i), reverse=True)
        return inds_lst

    inds_lst1 = sort_inds_lst(inds_lst1)
    inds_lst2 = sort_inds_lst(inds_lst2)
    for i, (inds1, inds2) in enumerate(zip(inds_lst1, inds_lst2)):
        inds1 = [(x, y) for x, y in inds1]
        inds2 = [(x, y) for x, y in inds2]
        try:
            assertListShiftedEqual(self, inds1, inds2)
        except AssertionError as e:
            err = ("shifted list assertion failed for {n} " "{i}: {e}\n").format(
                n=name, i=i, e=e
            )
            err += "\n {:^8s} {:^8s}\n".format(name + "1", name + "2")
            for j in range(max([len(inds1), len(inds2)])):
                x1, y1 = inds1[j] if j < len(inds1) else (-1, -1)
                x2, y2 = inds2[j] if j < len(inds2) else (-1, -1)
                err += " ({:2d}, {:2d}) ({:2d}, {:2d})\n".format(x1, y1, x2, y2)
            err += "\n"
            raise AssertionError(err)


# Test Tracks


class TestTracks_Base(TestCase):
    """Base class for tracks tests."""

    def assert_tracks_features(s, tracks, features_tracks, ignore_ts=[]):
        """Compare a list of tracks to a list of features.

        The features are those expected in the tracks. Please note that the
        order in which the list of features are passed matters: They must be
        reversely sorted by the number features (i.e. their lengths when
        ignoring any None objects).
        """

        # Sort by combined size of features
        tracks = sorted(tracks, key=lambda t: sum([f.n for f in t.features()]))
        features_tracks = sorted(
            features_tracks,
            key=lambda ft: sum([f.n for ff in ft for f in ff if f is not None]),
        )

        # Make sure ignore_ts is a list
        try:
            iter(ignore_ts)
        except TypeError:
            ignore_ts = [ignore_ts]

        # Compare track features
        for track, features in zip(tracks, features_tracks):
            res, sol = [], []
            feats = [[f for f in ff] for ff in features if any(f for f in ff)]

            for features_ts in track.features_ts():
                features_res_ts = sorted(features_ts)
                features_sol_ts = sorted(feats.pop(0))
                res.extend(features_res_ts)
                sol.extend(features_sol_ts)
            try:
                s.assertEqual(sorted(res), sorted(sol))
            except AssertionError:
                err = ("Features don't match:\n\nSolution:\n{}\n\nResult:\n{}").format(
                    "\n".join([str(f) for f in sol]), "\n".join([str(f) for f in res])
                )
                raise AssertionError(err)

    def run_tracking(s, objs_all, *, event_id=0, tracker=None):
        if not tracker:
            tracker = s.tracker
        for ts, objs in enumerate(objs_all):
            tracker.extend_tracks(objs, ts)
        tracker.finish_tracks()
        tracks = tracker.pop_finished_tracks()
        return tracks


# SR_TODO merge with TestTracks_Base (or pull out common base class)
class TestTrackFeatures_Base(TestCase):
    def run_test(s, grouped_features, n_exp=None):
        """Run tracking and fetch tracks."""
        for ts, typed_features in grouped_features:
            features = [feature for feature, type in typed_features]
            s.tracker.extend_tracks(features, ts)
        s.tracker.finish_tracks()
        tracks = s.tracker.pop_finished_tracks()
        if n_exp is not None:
            s.assertEqual(len(tracks), n_exp)
            if n_exp == 1:
                return tracks[0]
        return tracks

    def check_tracks(s, tracks, grouped_features):
        """Check tracks against reference features"""

        # Make sure there's only a single track
        s.assertEqual(len(tracks), 1)
        track = tracks[0]

        # Check the events types for every timestep
        for ts, typed_features in grouped_features:
            res = set(track.vertices_ts(ts)["type"])
            sol = set([type for feature, type in typed_features])
            s.assertSetEqual(
                res,
                sol,
                (
                    "feature types as timestep {} differ:" "\n\nres: {}\n\nsol: {}"
                ).format(ts, pformat(res), pformat(sol)),
            )

        # Check the number of neighbors of all vertices
        for vertex in track.graph.vs:
            type = vertex["type"]
            if "/" not in type:
                if type in ["start", "stop", "genesis", "lysis"]:
                    s.assertTrue(len(vertex.neighbors()) == 1)
                elif type == "continuation":
                    s.assertTrue(len(vertex.neighbors()) == 2)
                elif type in ["merging", "splitting"]:
                    s.assertTrue(len(vertex.neighbors()) >= 3)
                else:
                    err = "vertex type '{}'".format(type)
                    raise NotImplementedError(err)
            else:
                if all(i in type for i in ["merging", "splitting"]):
                    s.assertTrue(len(vertex.neighbors()) >= 4)
                elif "splitting" in type and any(
                    i in type for i in ["start", "genesis"]
                ):
                    s.assertTrue(len(vertex.neighbors()) >= 3)
                elif "merging" in type and any(i in type for i in ["stop", "lysis"]):
                    s.assertTrue(len(vertex.neighbors()) >= 3)
                else:
                    err = "vertex type '{}'".format(type)
                    raise NotImplementedError(err)

    def group_by_timestep(s, segments, dts=1):
        timesteps = sorted(set([seg[0] for seg in segments]))
        features_grouped = []
        for ts in timesteps:
            features_grouped.append((ts, []))
            for ts_seg, feature, type in segments:
                if ts_seg == ts:
                    features_grouped[-1][1].append((feature, type))
        return features_grouped

    def revert_grouped_features(s, features_grouped_fw):
        fw = features_grouped_fw
        bw = []
        for i in range(len(features_grouped_fw)):
            features_ts = []
            for feature, type_fw in fw[-(i + 1)][1]:
                type_bw = s.revert_type(type_fw)
                features_ts.append((feature, type_bw))
            bw.append((fw[i][0], features_ts))
        return bw

    def revert_type(s, type_fw):
        fw2bw = {
            "start": "stop",
            "stop": "start",
            "genesis": "lysis",
            "lysis": "genesis",
            "merging": "splitting",
            "splitting": "merging",
        }
        if isinstance(type_fw, str):
            return fw2bw.get(type_fw, type_fw)
        return (fw2bw.get(fw, fw) for fw in type_fw)


# Some utility functions/classes


def circle(px, py, rad, shell=None, connectivity=8):
    """Compute the coordinates of a circle with radius rad around (px, py).

    If an empty list is passed as last argument, the it is filled with the
    shell pixels.
    """
    _name_ = "circle"
    if shell is not None and (not isinstance(shell, list) or len(shell) > 0):
        raise ValueError("{}: shell must be None or []".format(_name_))
    if connectivity != 8:
        raise NotImplementedError("{}: connectivity != 8".format(_name_))
    center = np.array([px, py])
    pllc = np.floor(center - rad).astype(int)
    purc = np.ceil(center + rad).astype(int)
    fld = np.zeros(purc + 1)
    pts = []
    for i in range(pllc[0], purc[0] + 1):
        for j in range(pllc[1], purc[1] + 1):
            dist = np.sqrt((px - i) ** 2 + (py - j) ** 2)
            if dist <= rad:
                fld[i, j] = 1
                pts.append((i, j))
    if shell is not None:
        for i in range(purc[0] + 1):
            for j in range(purc[1] + 1):
                if fld[i, j] == 1:
                    try:
                        if any(
                            v == 0
                            for v in (
                                fld[i - 1, j],
                                fld[i, j - 1],
                                fld[i, j + 1],
                                fld[i + 1, j],
                            )
                        ):
                            shell.append((i, j))
                    except IndexError:
                        shell.append((i, j))
    return pts


def feature_circle(px, py, rad, id_, ts):
    shell_points = []
    points = circle(px, py, rad, shell_points)
    pixels = np.asarray(points, np.int32)
    shell = np.array(shell_points, np.int32)
    feature = Feature(pixels=pixels, shells=[shell], id_=id_, timestep=ts)
    return feature


def feature_rectangle(xymin, xymax, id, ts=0):

    # Create feature pixels
    x = np.arange(xymin[0], xymax[0] + 1)
    y = np.arange(xymin[1], xymax[1] + 1)
    nx, ny = len(x), len(y)
    pixels = np.array([(i, j) for i in x for j in y], np.int32)

    # Create shell pixels
    xl, yl = x.tolist(), y.tolist()
    shell_xy = np.array(
        [
            xl + [xymax[0]] * (ny - 2) + xl[::-1] + ([xymin[0]] * (ny - 2)),
            [xymax[1]] * nx + yl[1:-1][::-1] + [xymin[1]] * nx + yl[1:-1],
        ],
        np.int32,
    )
    shell = shell_xy.T

    # Create feature
    feature = Feature(pixels=pixels, shells=[shell], id_=id, timestep=ts)
    return feature


if __name__ == "__main__":

    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)

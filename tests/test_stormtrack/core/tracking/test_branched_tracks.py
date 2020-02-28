#!/usr/bin/env python3

# Standard library
import sys
import unittest
from datetime import datetime
from datetime import timedelta
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.tracking import FeatureTracker

# Local
from ...utils import feature_circle
from ...utils import TestTrackFeatures_Base
from ...utils import TestTracks_Base


class MergeSplit_Isolated(TestCase):
    """Test merging and splitting in isolation."""

    plot = False
    plotted = False

    def setUp(s):
        nx, ny = 40, 40

        s.feat0 = [
            feature_circle(10, 15, 6, 0, 0),  # split 0
            feature_circle(25, 12, 4, 1, 0),
        ]  # isolated 0
        s.feat1 = [
            feature_circle(8, 19, 3, 2, 1),  # split 1
            feature_circle(15, 16, 4, 3, 1),  # split 1
            feature_circle(25, 17, 5, 4, 1),
        ]  # isolated 1

        if s.plot and not s.plotted:
            raise NotImplementedError("{}.plot".format(s.__class__.__name__))
            objs = s.feat0 + s.feat1
            plot_contours("tracks_merge-split_isolated.png", objs)
            s.plotted = True

        # Set up tracker
        s.tracker = FeatureTracker(
            f_overlap=0.5,
            f_size=0.5,
            max_children=10,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            minsize=0,
            maxsize=0,
            connectivity=8,
            split_tracks_n=-1,
            merge_features=False,
            nx=nx,
            ny=ny,
        )

    def test_splitting_isolated(s):
        s.tracker.extend_tracks(s.feat0[0:1], 0)
        s.assertEqual(len(s.tracker.active_tracks), 1)
        s.tracker.extend_tracks(s.feat1[0:2], 1)
        s.assertEqual(len(s.tracker.active_tracks), 1)

    def test_merging_isolated(s):
        for f in s.feat1:
            f.timestep = 0
        for f in s.feat0:
            f.timestep = 1
        s.tracker.extend_tracks(s.feat1[0:2], 0)
        s.assertEqual(len(s.tracker.active_tracks), 2)
        s.tracker.extend_tracks(s.feat0[0:1], 1)
        s.assertEqual(len(s.tracker.active_tracks), 1)

    def test_splitting_full(s):
        s.tracker.extend_tracks(s.feat0, 0)
        s.assertEqual(len(s.tracker.active_tracks), 2)
        s.tracker.extend_tracks(s.feat1, 1)
        s.assertEqual(len(s.tracker.active_tracks), 2)

    def test_merging_full(s):
        for f in s.feat1:
            f.timestep = 0
        for f in s.feat0:
            f.timestep = 1
        s.tracker.extend_tracks(s.feat1, 0)
        s.assertEqual(len(s.tracker.active_tracks), 3)
        s.tracker.extend_tracks(s.feat0, 1)
        s.assertEqual(len(s.tracker.active_tracks), 2)


@unittest.skip("not implemented")
class MergeSplit_MissingFeature(TestTracks_Base):

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
            min_p_tot=0.0,
            min_p_overlap=0.1,
            min_p_size=0.0,
            max_area=2.0,
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


class MergeSplit_Complex_Separate(TestTrackFeatures_Base):
    """Complex case of multiple mergings/splittings, but only one at a time.

    Per test, there are either only splittings, or only mergings.
    """

    plot = False

    def setup_features(s, dir, ts0, dts=1):
        """"
        Note how the track is set up from the perspective of splitting.
        For the merging tests, the "splitting" events have to be replaced
        by "merging" events for the comparison with the test results.
        Accordingly, also the timesteps are reversed.
        """

        def ts(i):
            return ts0 + i * dts

        # Initial track
        seg10 = [
            (ts(0), feature_circle(2, 3, 2, 0, ts(0)), "start"),
            (ts(1), feature_circle(5, 4, 3, 1, ts(1)), "continuation"),
            (ts(2), feature_circle(9, 6, 4, 2, ts(2)), "continuation"),
            (ts(3), feature_circle(11, 9, 5, 3, ts(3)), "continuation"),
            (ts(4), feature_circle(13, 12, 6, 4, ts(4)), "splitting"),
        ]

        # Continuations of splitting of seg10
        seg20 = [  # to the left
            (ts(5), feature_circle(8, 16, 3, 5, ts(5)), "continuation"),
            (ts(6), feature_circle(6, 19, 4, 6, ts(6)), "continuation"),
            (ts(7), feature_circle(7, 23, 4, 7, ts(7)), "splitting"),
        ]
        seg21 = [  # upward
            (ts(5), feature_circle(17, 16, 4, 8, ts(5)), "continuation"),
            (ts(6), feature_circle(22, 19, 5, 9, ts(6)), "continuation"),
            (ts(7), feature_circle(28, 19, 7, 10, ts(7)), "splitting"),
        ]
        seg22 = [  # to the right
            (ts(5), feature_circle(16, 9, 2, 11, ts(5)), "continuation"),
            (ts(6), feature_circle(18, 7, 2, 12, ts(6)), "continuation"),
            (ts(7), feature_circle(18, 5, 1, 13, ts(7)), "lysis"),
        ]

        # Continuations of splitting of seg20
        seg30 = [  # to the left
            (ts(8), feature_circle(4, 23, 2, 14, ts(8)), "continuation"),
            (ts(9), feature_circle(2, 23, 1, 15, ts(9)), "lysis"),
        ]
        seg31 = [  # upward
            (ts(8), feature_circle(9, 26, 3, 16, ts(8)), "continuation"),
            (ts(9), feature_circle(9, 29, 2, 17, ts(9)), "continuation"),
            (ts(10), feature_circle(8, 31, 1, 18, ts(10)), "lysis"),
        ]

        # Continuations of splitting of seg21
        seg32 = [  # upward
            (ts(8), feature_circle(28, 25, 5, 19, ts(8)), "continuation"),
            (ts(9), feature_circle(31, 28, 4, 20, ts(9)), "continuation"),
            (ts(10), feature_circle(33, 30, 3, 21, ts(10)), "continuation"),
            (ts(11), feature_circle(37, 32, 3, 22, ts(11)), "continuation"),
            (ts(12), feature_circle(39, 33, 2, 23, ts(12)), "continuation"),
            (ts(13), feature_circle(41, 34, 1, 24, ts(13)), "stop"),
        ]
        seg33 = [  # to the right
            (ts(8), feature_circle(33, 16, 5, 25, ts(8)), "continuation"),
            (ts(9), feature_circle(36, 14, 4, 26, ts(9)), "continuation"),
            (ts(10), feature_circle(37, 11, 3, 27, ts(10)), "continuation"),
            (ts(11), feature_circle(37, 9, 2, 28, ts(11)), "continuation"),
            (ts(12), feature_circle(37, 8, 1, 29, ts(12)), "lysis"),
        ]

        segs = seg10 + seg20 + seg21 + seg22 + seg30 + seg31 + seg32 + seg33
        segs = s.group_by_timestep(segs)

        if dir == "fw":
            return segs
        elif dir == "bw":
            return s.revert_grouped_features(segs)

    def setUp(s):
        nx, ny = 50, 40

        # Set up features
        s.grouped_features_split = s.setup_features("fw", 0)
        s.grouped_features_merge = s.setup_features("bw", 0)

        # Set up tracker
        s.tracker = FeatureTracker(
            f_overlap=0.5,
            f_size=0.5,
            max_children=10,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            minsize=0,
            maxsize=0,
            connectivity=8,
            split_tracks_n=-1,
            merge_features=False,
            nx=nx,
            ny=ny,
        )

        if s.plot:
            raise NotImplementedError("{}.plot".format(s.__class__.__name__))
            features = [x[0] for x in seg]
            outfile = "test_merge_split_complex_separate.png"
            plot_contours(outfile, features, labels=lambda i: i.id())

    def test_split(s):
        """Full track with both 2-way and 3-way splitting."""
        tracks = s.run_test(s.grouped_features_split)
        s.check_tracks(tracks, s.grouped_features_split)

    def test_merge(s):
        """Full track with both 2-way and 3-way merging."""
        tracks = s.run_test(s.grouped_features_merge)
        s.check_tracks(tracks, s.grouped_features_merge)


class MergeSplit_Complex_Mixed_base(TestTrackFeatures_Base):
    """Complex case with both mergins and splittings in the same case."""

    plot = False

    nx = 77
    ny = 66

    def setup_features(s, dir, ts0, dts=1):
        def ts(i):
            if len(str(ts0)) > 12:
                raise NotImplementedError(ts0)
            if len(str(ts0)) == 12:
                ts0_dt = datetime.strptime(str(ts0), "%Y%m%d%H%M")
                ts_dt = ts0_dt + i * timedelta(minutes=dts)
                ts = int(ts_dt.strftime("%Y%m%d%H%M"))
                return ts
            return ts0 + i * dts

        # genesis -> lysis
        seg0 = [
            (ts(0), feature_circle(5, 13, 1, 0, ts(0)), "start"),
            (ts(1), feature_circle(6, 14, 2, 1, ts(1)), "continuation"),
            (ts(2), feature_circle(8, 15, 3, 2, ts(2)), "continuation"),
            (ts(3), feature_circle(9, 17, 3, 3, ts(3)), "continuation"),
            (ts(4), feature_circle(11, 18, 4, 4, ts(4)), "continuation"),
            (ts(5), feature_circle(12, 19, 4, 5, ts(5)), "continuation"),
            (ts(6), feature_circle(15, 21, 5, 6, ts(6)), "splitting"),  # 0
            (ts(7), feature_circle(18, 23, 3, 7, ts(7)), "continuation"),
            (ts(8), feature_circle(19, 24, 3, 8, ts(8)), "continuation"),
            (ts(9), feature_circle(20, 25, 3, 9, ts(9)), "continuation"),
            (ts(10), feature_circle(21, 27, 3, 10, ts(10)), "continuation"),
            (ts(11), feature_circle(23, 28, 3, 11, ts(11)), "continuation"),
            (ts(12), feature_circle(24, 30, 3, 12, ts(12)), "continuation"),
            (ts(13), feature_circle(26, 31, 3, 13, ts(13)), "continuation"),
            (ts(14), feature_circle(27, 33, 3, 14, ts(14)), "continuation"),
            (ts(15), feature_circle(28, 34, 4, 15, ts(15)), "continuation"),
            (ts(16), feature_circle(31, 36, 4, 16, ts(16)), "continuation"),
            (ts(17), feature_circle(34, 39, 5, 17, ts(17)), "merging"),  # 0
            (ts(18), feature_circle(35, 42, 4, 18, ts(18)), "continuation"),
            (ts(19), feature_circle(35, 45, 3, 19, ts(19)), "continuation"),
            (ts(20), feature_circle(35, 47, 2, 20, ts(20)), "continuation"),
            (ts(21), feature_circle(35, 48, 1, 21, ts(21)), "lysis"),
        ]

        # seg0/splitting0 -> lysis
        seg1 = [
            (ts(7), feature_circle(13, 24, 2, 22, ts(7)), "continuation"),
            (ts(8), feature_circle(13, 26, 2, 23, ts(8)), "continuation"),
            (ts(9), feature_circle(13, 29, 1, 24, ts(9)), "continuation"),
            (ts(10), feature_circle(13, 30, 1, 25, ts(10)), "lysis"),
        ]

        # seg0/splitting0 -> seg0/merging0
        seg2 = [
            (ts(7), feature_circle(17, 17, 2, 26, ts(7)), "continuation"),
            (ts(8), feature_circle(21, 17, 2, 27, ts(8)), "continuation"),
            (ts(9), feature_circle(24, 17, 3, 28, ts(9)), "continuation"),
            (ts(10), feature_circle(28, 14, 5, 29, ts(10)), "merging"),  # 0
            (ts(11), feature_circle(34, 15, 5, 30, ts(11)), "continuation"),
            (ts(12), feature_circle(39, 17, 5, 31, ts(12)), "splitting"),  # 0
            (ts(13), feature_circle(41, 22, 4, 32, ts(13)), "continuation"),
            (ts(14), feature_circle(42, 27, 4, 33, ts(14)), "continuation"),
            (ts(15), feature_circle(42, 32, 4, 34, ts(15)), "continuation"),
            (ts(16), feature_circle(38, 35, 3, 35, ts(16)), "continuation"),
        ]

        # seg2/splitting0 -> lysis
        seg3 = [
            (ts(13), feature_circle(42, 15, 3, 36, ts(13)), "continuation"),
            (ts(14), feature_circle(45, 14, 3, 37, ts(14)), "continuation"),
            (ts(15), feature_circle(49, 14, 3, 38, ts(15)), "continuation"),
            (ts(16), feature_circle(52, 14, 3, 39, ts(16)), "continuation"),
            (ts(17), feature_circle(55, 15, 3, 40, ts(17)), "continuation"),
            (ts(18), feature_circle(60, 16, 4, 41, ts(18)), "merging"),  # 0
            (ts(19), feature_circle(62, 19, 4, 42, ts(19)), "continuation"),
            (ts(20), feature_circle(62, 22, 4, 43, ts(20)), "continuation"),
            (ts(21), feature_circle(62, 26, 4, 44, ts(21)), "continuation"),
            (ts(22), feature_circle(62, 30, 4, 45, ts(22)), "continuation"),
            (ts(23), feature_circle(62, 33, 4, 46, ts(23)), "splitting"),  # 0
            (ts(24), feature_circle(60, 36, 3, 47, ts(24)), "continuation"),
            (ts(25), feature_circle(59, 39, 3, 48, ts(25)), "continuation"),
            (ts(26), feature_circle(58, 42, 3, 49, ts(26)), "continuation"),
            (ts(27), feature_circle(57, 45, 3, 50, ts(27)), "continuation"),
            (ts(28), feature_circle(58, 48, 3, 51, ts(28)), "continuation"),
            (ts(29), feature_circle(61, 50, 3, 52, ts(29)), "continuation"),
            (ts(30), feature_circle(63, 52, 4, 53, ts(30)), "merging/splitting"),  # 1/1
            (ts(31), feature_circle(66, 55, 2, 54, ts(31)), "continuation"),
            (ts(32), feature_circle(68, 57, 2, 55, ts(32)), "continuation"),
            (ts(33), feature_circle(70, 58, 2, 56, ts(33)), "continuation"),
            (ts(34), feature_circle(72, 58, 1, 57, ts(34)), "stop"),
        ]

        # seg3/splitting0 -> seg3/merging1
        seg4 = [
            (ts(24), feature_circle(66, 34, 2, 58, ts(24)), "continuation"),
            (ts(25), feature_circle(67, 37, 2, 59, ts(25)), "continuation"),
            (ts(26), feature_circle(68, 40, 2, 60, ts(26)), "continuation"),
            (ts(27), feature_circle(68, 44, 2, 61, ts(27)), "continuation"),
            (ts(28), feature_circle(67, 47, 2, 62, ts(28)), "continuation"),
            (ts(29), feature_circle(66, 49, 2, 63, ts(29)), "continuation"),
        ]

        # seg3/splitting1 -> lysis
        seg5 = [
            (ts(31), feature_circle(61, 55, 2, 64, ts(31)), "continuation"),
            (ts(32), feature_circle(60, 56, 1, 65, ts(32)), "continuation"),
            (ts(33), feature_circle(59, 57, 1, 66, ts(33)), "lysis"),
        ]

        # genesis -> seg3/merging0
        seg6 = [
            (ts(15), feature_circle(65, 15, 1, 67, ts(15)), "genesis"),
            (ts(16), feature_circle(63, 15, 2, 68, ts(16)), "continuation"),
            (ts(17), feature_circle(61, 17, 3, 69, ts(17)), "continuation"),
        ]

        # genesis -> seg2/merging0
        seg7 = [
            (ts(4), feature_circle(13, 5, 1, 70, ts(4)), "genesis"),
            (ts(5), feature_circle(15, 5, 2, 71, ts(5)), "continuation"),
            (ts(6), feature_circle(17, 6, 2, 72, ts(6)), "continuation"),
            (ts(7), feature_circle(20, 7, 3, 73, ts(7)), "continuation"),
            (ts(8), feature_circle(23, 8, 3, 74, ts(8)), "continuation"),
            (ts(9), feature_circle(26, 9, 4, 75, ts(9)), "continuation"),
        ]

        segs = seg0 + seg1 + seg2 + seg3 + seg4 + seg5 + seg6 + seg7
        segs = s.group_by_timestep(segs, dts)

        if dir == "fw":
            return segs
        elif dir == "bw":
            return s.revert_grouped_features(segs)

        raise ValueError("invalid dir: " + dir)

    def setup_tracker(s, **kwas):
        conf = dict(
            f_overlap=0.5,
            f_size=0.5,
            max_children=10,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            minsize=0,
            maxsize=0,
            connectivity=8,
            split_tracks_n=-1,
            merge_features=False,
            nx=s.nx,
            ny=s.ny,
        )
        conf.update(kwas)
        return FeatureTracker(**conf)


class MergeSplit_Complex_Mixed(MergeSplit_Complex_Mixed_base):
    def setUp(s):
        # Set up features
        s.grouped_features_fw = s.setup_features("fw", 0)
        s.grouped_features_bw = s.setup_features("bw", 0)

        # Set up tracker
        s.tracker = s.setup_tracker()

        if s.plot:
            raise NotImplementedError("{}.plot".format(s.__class__.__name__))
            features = [x[0] for x in seg]
            outfile = "test_merge_split_complex_mixed.png"
            plot_contours(outfile, features, labels=lambda f: "{}".format(f.id()))

    def test_forward(s):
        tracks = s.run_test(s.grouped_features_fw)
        s.check_tracks(tracks, s.grouped_features_fw)

    def test_backward(s):
        tracks = s.run_test(s.grouped_features_bw)
        s.check_tracks(tracks, s.grouped_features_bw)

    # SR_TODO check tracks, not only their number
    def test_forward_nobranch(s):
        """n_children=0 -> no branching"""
        s.tracker.max_children = 1
        tracks = s.run_test(s.grouped_features_fw)
        s.assertEqual(len(tracks), 8)

    def test_forward_postproc_split_unconditional(s):
        """Split tracks at all branchings."""

        # Create track and split it
        tracks = s.run_test(s.grouped_features_fw)
        s.assertEqual(len(tracks), 1)
        track_big = tracks[0]
        tracks_res = track_big.split(0)

        # Create solution track (w/o any branchings)
        s.tracker.reset()
        s._reset_features(s.grouped_features_fw)
        s.tracker.max_children = 1
        tracks_sol = s.run_test(s.grouped_features_fw)

        # Check results
        tracks_res.sort(key=lambda t: sum(f.n for f in t.features()))
        tracks_sol.sort(key=lambda t: sum(f.n for f in t.features()))
        for track_res, track_sol in zip(tracks_res, tracks_sol):

            # Check features (order doesn't matter)
            res = set(track_res.features())
            sol = set(track_sol.features())
            s.assertSetEqual(res, sol)

            # Check vertex types (order doesn't matter)
            res = sorted(track_res.graph.vs["type"])
            sol = sorted(track_sol.graph.vs["type"])
            s.assertEqual(res, sol)

    # SR_TODO add checks for features/feature types
    def test_forward_postproc_split_n8(s):
        """Split tracks at all branchings unless re-merge w/i 8 timesteps.

        This happens once for this track (gap of 7 timesteps), i.e. one
        splitting and one merging are retained.
        """

        # Create track and split it
        tracks = s.run_test(s.grouped_features_fw)
        s.assertEqual(len(tracks), 1)
        track_big = tracks[0]
        tracks_res = track_big.split(8)

        s.assertEqual(len(tracks_res), 3)

    # SR_TODO add checks for features/feature types
    def test_forward_postproc_split_n12(s):
        """Split at all branchings unless re-merge/finish in 8 timesteps.

        This happens once for this track (gap of 7 timesteps), i.e. one
        splitting and one merging are retained.
        """

        # Create track and split it
        tracks = s.run_test(s.grouped_features_fw)
        s.assertEqual(len(tracks), 1)
        track_big = tracks[0]

        tracks_res = track_big.split(12)

        s.assertEqual(len(tracks_res), 2)

    def _reset_features(s, features_ts):
        """Reset Feature.cregion to NULL (otherwise warnings in 2nd run)."""
        for ts, features in features_ts:
            for feature, type in features:
                feature.cleanup_cregion()


class MergeSplit_Complex_TempRes(MergeSplit_Complex_Mixed_base):
    """Test different temporal resolutions."""

    def setUp(s):
        pass

    def test_forward_hour(s):
        """Test with hourly timesteps."""
        s.tracker = s.setup_tracker()
        grouped_features = s.setup_features("fw", 2007102100, 1)
        track = s.run_test(grouped_features, 1)

    def test_forward_minute_hour(s):
        """Test with hourly timesteps."""
        s.tracker = s.setup_tracker()
        grouped_features = s.setup_features("fw", 200710210000, 60)
        track = s.run_test(grouped_features, 1)
        s.check_tracks([track], grouped_features)

    def test_forward_minute_minute(s):
        """Test with hourly timesteps."""
        s.tracker = s.setup_tracker()
        grouped_features = s.setup_features("fw", 200710210000, 1)
        track = s.run_test(grouped_features, 1)
        s.check_tracks([track], grouped_features)

        # tss = track.timesteps()
        # ipython(globals(), locals())


if __name__ == "__main__":

    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)

    unittest.main()

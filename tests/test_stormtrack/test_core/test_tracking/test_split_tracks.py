#!/usr/bin/env python3

# Standard library
import logging as log
import sys
import unittest
from collections import OrderedDict as odict
from unittest import TestCase

# Thirt-party
import numpy as np
import igraph as ig

# First-party
from stormtrack.core.tracking import FeatureTrack
from stormtrack.core.tracking import FeatureTracker
from stormtrack.core.tracking import FeatureTrackSplitter


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


class DummyFeature:
    def __init__(self, id_, ts, type):
        self.id = id_
        self.timestep = ts
        type_reg = dict(
            ge="genesis",
            ly="lysis",
            gl="genesis/lysis",
            co="continuation",
            me="merging",
            sp="splitting",
            ms="merging/splitting",
            gs="genesis/splitting",
            ml="merging/lysis",
        )
        self.type = type_reg[type]

        self.n = -1

    def set_track(self, track):
        self.track = track


class SplitTracks_Base(TestCase):
    def init_track(self, tid, features_ts, edges_attrs):
        features = [f for ff in features_ts for f in ff]

        # Initialize graph
        graph = ig.Graph(directed=True)

        # Add vertices (attributes: feature, name, ts, type, head)
        graph.add_vertices(len(features))
        graph.vs["feature"] = features
        graph.vs["ts"] = [f.timestep for f in features]
        graph.vs["name"] = [str(f.id) for f in features]
        graph.vs["type"] = [str(f.type) for f in features]
        graph.vs["missing_predecessors"] = None
        graph.vs["missing_successors"] = None

        # Add edges (attributes: p_tot, p_size, p_overlap, p_share)
        for (vid0, vid1), edge_attrs in edges_attrs.items():
            for key in FeatureTrack.es_attrs:
                if key not in edge_attrs:
                    edge_attrs[key] = None
            vx0 = graph.vs.find(str(vid0))
            vx1 = graph.vs.find(str(vid1))
            graph.add_edge(vx0.index, vx1.index, **edge_attrs)

        # SR_TMP<
        config = dict(
            f_overlap=0.5, f_size=0.5, min_p_overlap=0.0, min_p_size=0.0, min_p_tot=0.0,
        )
        # SR_TMP>

        # Initialize track
        track = FeatureTrack(id_=0, graph=graph, config=config)

        return track

    def assertTracksEdgesEqual(self, tracks, tracks_edges_sol):
        tracks.sort(key=lambda t: t.n)
        tracks_edges_sol.sort(key=lambda es: len(es))

        # SR_TODO more elegant and robust structure than nested try/except

        # Check number of tracks
        try:
            self.assertEqual(len(tracks), len(tracks_edges_sol))
        except AssertionError:
            err = "wrong number of tracks: {} instead of {}".format(
                len(tracks), len(tracks_edges_sol)
            )
        else:

            # Check track edges
            nerr = 0
            for track, sol in zip(tracks, tracks_edges_sol):
                try:
                    self.assertTrackEdgesEqual(track, sol)
                except AssertionError:
                    nerr += 1
            if nerr == 0:
                return  # !!! NO MORE CHECKS AFTER THIS !!!
            err = "edges of {}/{} tracks wrong".format(nerr, len(tracks))

        # Error message
        tracks_edges_res = [self._get_edge_tuples(t.graph) for t in tracks]
        err += "\n\nexpected:\n{}".format(
            "\n".join(["  {}".format(sorted(i)) for i in tracks_edges_sol])
        )
        err += "\n\nfound:\n{}".format(
            "\n".join(["  {}".format(sorted(i)) for i in tracks_edges_res])
        )
        err += "\n"
        raise AssertionError(err)

    def assertTrackEdgesEqual(self, track, edge_tuples_sol):
        res = self._get_edge_tuples(track.graph)
        self.assertSetEqual(res, edge_tuples_sol)

    def _get_edge_tuples(self, graph):
        vid2fid = lambda vid: graph.vs[vid]["feature"].id
        return {(vid2fid(e.source), vid2fid(e.target)) for e in graph.es}


# SR_TODO check subtracks, not just their number
class SplitTracks_TwoBranches_SplitReMerge(SplitTracks_Base):
    def setUp(s):
        r"""
                      [ 2]-[ 4]-[ 6]-[ 8]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 9]-[10]-[11]-[12]

        """
        features_ts = [  #     id ts  type
            [DummyFeature(0, 0, "ge")],
            [DummyFeature(1, 1, "sp")],
            [DummyFeature(2, 2, "co"), DummyFeature(3, 2, "co")],
            [DummyFeature(4, 3, "co"), DummyFeature(5, 3, "co")],
            [DummyFeature(6, 4, "co"), DummyFeature(7, 4, "co")],
            [DummyFeature(8, 5, "co"), DummyFeature(9, 5, "co")],
            [DummyFeature(10, 6, "me")],
            [DummyFeature(11, 7, "co")],
            [DummyFeature(12, 8, "ly")],
        ]
        #
        #             [ 2]-[ 4]-[ 6]-[ 8]
        #            x                   \
        #   [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 9]x[10]-[11]-[12]
        #
        edges_attrs = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 2): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (2, 4): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 7): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (6, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (7, 9): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (8, 10): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (9, 10): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (10, 11): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (11, 12): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track = s.init_track(0, features_ts, edges_attrs)

        used_tids = {0}
        s.splitter = FeatureTrackSplitter(used_tids)

    def test_n6_nosplit(s):
        r""" nt=6
                      [ 2]-[ 4]-[ 6]-[ 8]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 9]-[10]-[11]-[12]
        """
        subtracks = s.splitter.split(s.track, n=6)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {
                    (0, 1),
                    (1, 2),
                    (1, 3),
                    (3, 5),
                    (5, 7),
                    (7, 9),
                    (2, 4),
                    (4, 6),
                    (6, 8),
                    (8, 10),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                }
            ],
        )

    def test_n5_nosplit(s):
        r""" nt=5
                      [ 2]-[ 4]-[ 6]-[ 8]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 9]-[10]-[11]-[12]
        """
        subtracks = s.splitter.split(s.track, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {
                    (0, 1),
                    (1, 2),
                    (1, 3),
                    (3, 5),
                    (5, 7),
                    (7, 9),
                    (2, 4),
                    (4, 6),
                    (6, 8),
                    (8, 10),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                }
            ],
        )

    def test_n4_split(s):
        r""" nt=4
                      [ 2]-[ 4]-[ 6]-[ 8]
                     x                   \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 9]x[10]-[11]-[12]
        """
        subtracks = s.splitter.split(s.track, n=4)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {(0, 1), (1, 3), (3, 5), (5, 7), (7, 9)},
                {(2, 4), (4, 6), (6, 8), (8, 10), (10, 11), (11, 12)},
            ],
        )


# SR_TODO check subtracks, not just their number
class SplitTracks_TwoBranches_Split(SplitTracks_Base):
    def setUp(s):
        r"""
                      [ 2]-[ 4]-[ 6]-[ 8]
                     /
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]

        """
        features_ts = [  #     id ts  n  type
            [DummyFeature(0, 0, "ge")],
            [DummyFeature(1, 1, "sp")],
            [DummyFeature(2, 2, "co"), DummyFeature(3, 2, "co")],
            [DummyFeature(4, 3, "co"), DummyFeature(5, 3, "co")],
            [DummyFeature(6, 4, "co"), DummyFeature(7, 4, "ly")],
            [DummyFeature(8, 5, "ly")],
        ]
        #
        #              [ 2]-[ 4]-[ 6]-[ 8]
        #             x
        #    [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        #
        edges_attrs1 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 2): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (2, 4): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 7): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (6, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track1 = s.init_track(1, features_ts, edges_attrs1)
        #
        #              [ 2]-[ 4]-[ 6]-[ 8]
        #             /
        #    [ 0]-[ 1]x[ 3]-[ 5]-[ 7]
        #
        edges_attrs2 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 2): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (2, 4): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 7): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (6, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track2 = s.init_track(2, features_ts, edges_attrs2)

        used_tids = {1, 2}
        s.splitter = FeatureTrackSplitter(used_tids)

    def test_track1_n5_nosplit(s):
        r""" nt=5
                       [ 2]-[ 4]-[ 6]-[ 8]
                      /
             [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track1, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (1, 3), (3, 5), (5, 7)},],
        )

    def test_track1_n4_split(s):
        """ nt=4
                       [ 2]-[ 4]-[ 6]-[ 8]
                      x
             [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track1, n=4)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5), (5, 7)}, {(2, 4), (4, 6), (6, 8)},]
        )

    def test_track1_n3_split(s):
        r""" nt=3
                       [ 2]-[ 4]-[ 6]-[ 8]
                      x
             [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track1, n=3)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5), (5, 7)}, {(2, 4), (4, 6), (6, 8)},]
        )

    def test_track2_n5_nosplit(s):
        """ nt=5
                       [ 2]-[ 4]-[ 6]-[ 8]
                      /
             [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track2, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (1, 3), (3, 5), (5, 7)},],
        )

    def test_track2_n4_nosplit(s):
        r""" nt=4
                       [ 2]-[ 4]-[ 6]-[ 8]
                      /
             [ 0]-[ 1]-[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track2, n=4)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 2), (2, 4), (4, 6), (6, 8), (1, 3), (3, 5), (5, 7)},],
        )

    def test_track2_n3_split(s):
        r""" nt=2
                       [ 2]-[ 4]-[ 6]-[ 8]
                      /
             [ 0]-[ 1]x[ 3]-[ 5]-[ 7]
        """
        subtracks = s.splitter.split(s.track2, n=3)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 2), (2, 4), (4, 6), (6, 8)}, {(3, 5), (5, 7)},]
        )


# SR_TODO check subtracks, not just their number
class SplitTracks_TwoBranches_Merge(SplitTracks_Base):
    def setUp(s):
        r"""
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        features_ts = [  #     id ts  type
            [DummyFeature(0, 0, "ge")],
            [DummyFeature(1, 1, "co"), DummyFeature(2, 1, "ge")],
            [DummyFeature(3, 2, "co"), DummyFeature(4, 2, "co")],
            [DummyFeature(5, 3, "co"), DummyFeature(6, 3, "co")],
            [DummyFeature(7, 4, "me")],
            [DummyFeature(8, 5, "ly")],
        ]
        #
        #        [ 2]-[ 4]-[ 6]
        #                      x
        #   [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        #
        edges_attrs1 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (2, 4): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 7): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (6, 7): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (7, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track1 = s.init_track(1, features_ts, edges_attrs1)
        #
        #        [ 2]-[ 4]-[ 6]
        #                      \
        #   [ 0]-[ 1]-[ 3]-[ 5]x[ 7]-[ 8]
        #
        edges_attrs2 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (2, 4): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 7): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (6, 7): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (7, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track2 = s.init_track(2, features_ts, edges_attrs2)

        used_tids = {1, 2}
        s.splitter = FeatureTrackSplitter(used_tids)

    def test_track1_n5_nosplit(s):
        r""" nt=5
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track1, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 3), (3, 5), (5, 7), (7, 8), (2, 4), (4, 6), (6, 7)},],
        )

    def test_track1_n4_nosplit(s):
        r""" nt=4
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track1, n=4)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 3), (3, 5), (5, 7), (7, 8), (2, 4), (4, 6), (6, 7)},],
        )

    def test_track1_n3_split(s):
        r""" nt=3
                 [ 2]-[ 4]-[ 6]
                               x
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track1, n=3)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5), (5, 7), (7, 8)}, {(2, 4), (4, 6)},]
        )

    def test_track1_n2_split(s):
        r""" nt=2
                 [ 2]-[ 4]-[ 6]
                               x
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track1, n=2)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5), (5, 7), (7, 8)}, {(2, 4), (4, 6)},]
        )

    def test_track2_n5_nosplit(s):
        r""" nt=5
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]-[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track2, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [{(0, 1), (1, 3), (3, 5), (5, 7), (7, 8), (2, 4), (4, 6), (6, 7)},],
        )

    def test_track2_n4_split(s):
        r""" nt=4
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]x[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track2, n=4)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5)}, {(2, 4), (4, 6), (6, 7), (7, 8)},]
        )

    def test_track2_n3_split(s):
        r""" nt=3
                 [ 2]-[ 4]-[ 6]
                               \
            [ 0]-[ 1]-[ 3]-[ 5]x[ 7]-[ 8]
        """
        subtracks = s.splitter.split(s.track2, n=3)
        s.assertTracksEdgesEqual(
            subtracks, [{(0, 1), (1, 3), (3, 5)}, {(2, 4), (4, 6), (6, 7), (7, 8)},]
        )


class SplitTracks_ThreeBranches_SplitReMergeReMerge(SplitTracks_Base):
    """Splitting into three branches, which remerge in two steps."""

    def setUp(s):
        r"""
                      [ 2]-[ 5]-[ 8]-[10]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 6]-[ 9]-[11]-[12]-[13]
                     \         /
                      [ 4]-[ 7]
        """
        DF = DummyFeature

        features_ts = [
            #   id ts  type
            [DF(0, 0, "ge")],
            [DF(1, 1, "sp")],
            [DF(2, 2, "co"), DF(3, 2, "co"), DF(4, 2, "co")],
            [DF(5, 3, "co"), DF(6, 3, "co"), DF(7, 3, "co")],
            [DF(8, 4, "co"), DF(9, 4, "co"), DF(10, 4, "co")],
            [DF(11, 5, "co"), DF(12, 5, "me")],
            [DF(13, 6, "me")],
            [DF(14, 7, "ly")],
        ]
        #
        #             [ 2]-[ 5]-[ 8]-[11]
        #            x                   x
        #   [ 0]-[ 1]-[ 3]-[ 6]-[ 9]x[12]-[13]-[14]
        #            x              /
        #             [ 4]-[ 7]-[10]
        #
        edges_attrs1 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 2): dict(p_share=0.3, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=0.5, n_overlap=0, p_tot=0.0),
            (1, 4): dict(p_share=0.2, n_overlap=0, p_tot=0.0),
            (2, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 7): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (6, 9): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (7, 10): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (8, 11): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (9, 12): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (10, 12): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (11, 13): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (12, 13): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (13, 14): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track1 = s.init_track(1, features_ts, edges_attrs1)
        #
        #             [ 2]-[ 5]-[ 8]-[11]
        #            /                   x
        #   [ 0]-[ 1]x[ 3]-[ 6]-[ 9]x[12]-[13]-[14]
        #            x              /
        #             [ 4]-[ 7]-[10]
        #
        edges_attrs2 = {
            (0, 1): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (1, 2): dict(p_share=0.5, n_overlap=0, p_tot=0.0),
            (1, 3): dict(p_share=0.3, n_overlap=0, p_tot=0.0),
            (1, 4): dict(p_share=0.2, n_overlap=0, p_tot=0.0),
            (2, 5): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (3, 6): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (4, 7): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (5, 8): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (6, 9): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (7, 10): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (8, 11): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
            (9, 12): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (10, 12): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (11, 13): dict(p_share=0.4, n_overlap=0, p_tot=0.0),
            (12, 13): dict(p_share=0.6, n_overlap=0, p_tot=0.0),
            (13, 14): dict(p_share=1.0, n_overlap=0, p_tot=0.0),
        }
        s.track2 = s.init_track(2, features_ts, edges_attrs2)

        used_tids = {1, 2}
        s.splitter = FeatureTrackSplitter(used_tids)

    def test_track1_n5_nosplit(s):
        r""" nt=5
                      [ 2]-[ 5]-[ 8]-[11]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 6]-[ 9]-[12]-[13]-[14]
                     \              /
                      [ 4]-[ 7]-[10]

        """
        subtracks = s.splitter.split(s.track1, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {
                    (0, 1),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 5),
                    (3, 6),
                    (4, 7),
                    (5, 8),
                    (6, 9),
                    (7, 10),
                    (8, 11),
                    (9, 12),
                    (10, 12),
                    (11, 13),
                    (12, 13),
                    (13, 14),
                },
            ],
        )

    def test_track1_n4_split2(s):
        r""" nt=4
                      [ 2]-[ 5]-[ 8]-[11]
                     x                   x
            [ 0]-[ 1]-[ 3]-[ 6]-[ 9]-[12]-[13]-[14]
                     \              /
                      [ 4]-[ 7]-[10]

        """
        subtracks = s.splitter.split(s.track1, n=4)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {
                    (0, 1),
                    (1, 3),
                    (1, 4),
                    (3, 6),
                    (4, 7),
                    (6, 9),
                    (7, 10),
                    (9, 12),
                    (10, 12),
                    (12, 13),
                    (13, 14),
                },
                {(2, 5), (5, 8), (8, 11)},
            ],
        )

    def test_track1_n3_split4(s):
        r""" nt=3
                      [ 2]-[ 5]-[ 8]-[11]
                     x                   x
            [ 0]-[ 1]-[ 3]-[ 6]-[ 9]x[12]-[13]-[14]
                     x              /
                      [ 4]-[ 7]-[10]

        """
        subtracks = s.splitter.split(s.track1, n=3)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {(0, 1), (1, 3), (3, 6), (6, 9)},
                {(2, 5), (5, 8), (8, 11)},
                {(4, 7), (7, 10), (10, 12), (12, 13), (13, 14)},
            ],
        )

    def test_track2_n5_nosplit(s):
        r""" nt=5
                      [ 2]-[ 5]-[ 8]-[11]
                     /                   \
            [ 0]-[ 1]-[ 3]-[ 6]-[ 9]-[12]-[13]-[14]
                     \              /
                      [ 4]-[ 7]-[10]

        """
        subtracks = s.splitter.split(s.track2, n=5)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {
                    (0, 1),
                    (1, 2),
                    (1, 3),
                    (1, 4),
                    (2, 5),
                    (3, 6),
                    (4, 7),
                    (5, 8),
                    (6, 9),
                    (7, 10),
                    (8, 11),
                    (9, 12),
                    (10, 12),
                    (11, 13),
                    (12, 13),
                    (13, 14),
                },
            ],
        )

    def test_track2_n4_split3(s):
        r""" nt=4
                      [ 2]-[ 5]-[ 8]-[11]
                     /                   x
            [ 0]-[ 1]x[ 3]-[ 6]-[ 9]-[12]-[13]-[14]
                     x              /
                      [ 4]-[ 7]-[10]

        """
        subtracks = s.splitter.split(s.track2, n=4)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {(0, 1), (1, 2), (2, 5), (5, 8), (8, 11)},
                {
                    (3, 6),
                    (6, 9),
                    (9, 12),
                    (4, 7),
                    (7, 10),
                    (10, 12),
                    (12, 13),
                    (13, 14),
                },
            ],
        )

    def test_track2_n3_split4(s):
        r""" nt=3
                      [ 2]-[ 5]-[ 8]-[11]
                     /                   x
            [ 0]-[ 1]x[ 3]-[ 6]-[ 9]x[12]-[13]-[14]
                     x              /
                      [ 4]-[ 7]-[10]
        """
        subtracks = s.splitter.split(s.track2, n=3)
        s.assertTracksEdgesEqual(
            subtracks,
            [
                {(0, 1), (1, 2), (2, 5), (5, 8), (8, 11)},
                {(3, 6), (6, 9)},
                {(4, 7), (7, 10), (10, 12), (12, 13), (13, 14)},
            ],
        )


class SplitTrack_Probabilities_Simple(TestCase):
    """Split very simple track and check tracking probabilities."""

    def setUp(s):
        def feature_rect(fid, llc, urc, ts):
            pxs = np.arange(llc[0], urc[0] + 1)
            pys = np.arange(llc[1], urc[1] + 1)
            pixels = np.array([(x, y) for x in pxs for y in pys], np.int32)
            shell = np.array(
                [(px, pys[0]) for px in pxs]
                + [(pxs[-1], py) for py in pys[1:-1]]
                + [(px, pys[-1]) for px in pxs[::-1]]
                + [(pxs[0], py) for py in pys[-2:0:-1]],
                np.int32,
            )

            return Feature(id_=fid, timestep=ts, pixels=pixels, shells=[shell])

        nx, ny = 9, 9
        f_size = 0.4
        f_overlap = 0.6

        X, Y, Z = 10, 11, 12
        #
        #                       6 - - - -  6 - - - -  6 - - - -  6 - - - -
        #          5 - - - - -  5 - 5 5 5  5 7 7 7 -  5 - - X -  5 - - Z Z
        # 4 - - -  4 2 2 - 3 3  4 - 5 5 5  4 7 7 7 -  4 - - X -  4 - - Z Z
        # 3 0 0 -  3 2 2 - 3 3  3 - 5 5 5  3 - 8 8 8  3 - - - -  3 - - Z Z
        # 2 0 0 -  2 2 2 - - -  2 - - - -  2 - 8 8 8  2 - - Y Y  2 - - Z Z
        # 1 0 0 -  1 - - - - -  1 - - - -  1 - - - 9  1 - - Y Y  1 - - - -
        # 0 - - 1  0 - - 4 4 -  0 - - - 6  0 - - - 9  0 - - Y Y  0 - - - -
        # (0)0 1 2 (1)0 1 2 3 4 (2)0 1 2 3 (3)0 1 2 3 (4)0 1 2 3 (5)0 1 2 3
        #
        #
        # (0)   [0]       [1]
        #        |         |
        # (1)   [2]  [3]  [4]
        #        \\  /     |
        # (2)     [5]     [6]
        #        //  \     |
        # (3)   [7]  [8]  [9]
        #        |    \\  /
        # (4)   [X]    [Y]
        #          \  //
        # (5)       [Z]
        #
        s.features_tss = odict()
        s.features_tss[0] = [
            feature_rect(0, (0, 1), (1, 3), 0),
            feature_rect(1, (2, 0), (2, 0), 0),
        ]
        s.features_tss[1] = [
            feature_rect(2, (0, 2), (1, 4), 1),
            feature_rect(3, (3, 3), (4, 4), 1),
            feature_rect(4, (2, 0), (3, 0), 1),
        ]
        s.features_tss[2] = [
            feature_rect(5, (1, 3), (3, 5), 2),
            feature_rect(6, (3, 0), (3, 0), 2),
        ]
        s.features_tss[3] = [
            feature_rect(7, (0, 4), (2, 5), 3),
            feature_rect(8, (1, 2), (3, 3), 3),
            feature_rect(9, (3, 0), (3, 1), 3),
        ]
        s.features_tss[4] = [
            feature_rect(X, (2, 4), (2, 5), 4),
            feature_rect(Y, (2, 0), (3, 2), 4),
        ]
        s.features_tss[5] = [
            feature_rect(Z, (2, 2), (3, 5), 5),
        ]
        s.features_fid = {f.id: f for fs in s.features_tss.values() for f in fs}

        # Define overlaps (in both directions)
        s.overlaps = {
            (0, 2): 4,
            (1, 4): 1,
            (2, 5): 2,
            (3, 5): 2,
            (4, 6): 1,
            (5, 7): 4,
            (5, 8): 3,
            (6, 9): 1,
            (7, X): 2,
            (8, Y): 2,
            (9, Y): 2,
            (X, Z): 2,
            (Y, Z): 2,
        }
        s.overlaps.update({(b, a): o for (a, b), o in s.overlaps.items()})

        # Helper functions to compute tracking probabilities
        O = lambda fa, fb: s.overlaps.get((fa, fb), 0)
        S = lambda f: s.features_fid[f].n

        def PS(fids1, fids2):
            """p_size"""
            size1 = sum([S(i) for i in fids1])
            size2 = sum([S(i) for i in fids2])
            return min([size1, size2]) / max([size1, size2])

        def PO(fids1, fids2):
            """p_overlap"""
            ovlp = sum([O(f1, f2) for f1 in fids1 for f2 in fids2])
            size1 = sum([S(i) for i in fids1])
            size2 = sum([S(i) for i in fids2])
            return 2 * ovlp / (size1 + size2)

        def PT(fids1, fids2):
            """p_tot"""
            return f_size * PS(fids1, fids2) + f_overlap * PO(fids1, fids2)

        def PR(fid1, fids1, fid2):
            """p_share"""
            pt1 = PT([fid1], [fid2])
            ptt = sum([PT([i], [fid2]) for i in [fid1] + fids1])
            return pt1 / ptt

        s.Ps = lambda fids1, fids2: dict(
            p_tot=PT(fids1, [fid2]), p_size=PS(fids1, fids2), p_ovlp=PO(fids1, [fid2])
        )

        def PTSO(fids1, fids2):
            return dict(
                p_tot=PT(fids1, fids2),
                p_size=PS(fids1, fids2),
                p_overlap=PO(fids1, fids2),
            )

        def merge(dict1, dict2):
            dict_out = {k: v for k, v in dict1.items()}
            dict_out.update(dict2)
            return dict_out

        # Define tracking probabilities for n=-1 (original track)
        s.probs_ini = {
            (0, 2): merge(PTSO([0], [2]), dict(p_share=1.0)),
            (1, 4): merge(PTSO([1], [4]), dict(p_share=1.0)),
            (2, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(2, [3], 5))),
            (3, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(3, [2], 5))),
            (4, 6): merge(PTSO([4], [6]), dict(p_share=1.0)),
            (5, 7): merge(PTSO([5], [7, 8]), dict(p_share=PR(7, [8], 5))),
            (5, 8): merge(PTSO([5], [7, 8]), dict(p_share=PR(8, [7], 5))),
            (6, 9): merge(PTSO([6], [9]), dict(p_share=1.0)),
            (7, X): merge(PTSO([7], [X]), dict(p_share=1.0)),
            (8, Y): merge(PTSO([8, 9], [Y]), dict(p_share=PR(8, [9], Y))),
            (9, Y): merge(PTSO([8, 9], [Y]), dict(p_share=PR(9, [8], Y))),
            (X, Z): merge(PTSO([X, Y], [Z]), dict(p_share=PR(X, [Y], Z))),
            (Y, Z): merge(PTSO([X, Y], [Z]), dict(p_share=PR(Y, [X], Z))),
        }
        for (v0, v1), attrs in s.probs_ini.items():
            attrs["n_overlap"] = O(v0, v1)

        # Define tracking probabilities for n=0 (linear tracks)
        s.n_subtracks_n0 = 4
        s.probs_n0 = {
            (0, 2): merge(PTSO([0], [2]), dict(p_share=1.0)),
            (1, 4): merge(PTSO([1], [4]), dict(p_share=1.0)),
            (2, 5): merge(PTSO([2], [5]), dict(p_share=1.0)),
            (5, 7): merge(PTSO([5], [7]), dict(p_share=1.0)),
            (4, 6): merge(PTSO([4], [6]), dict(p_share=1.0)),
            (6, 9): merge(PTSO([6], [9]), dict(p_share=1.0)),
            (7, X): merge(PTSO([7], [X]), dict(p_share=1.0)),
            (8, Y): merge(PTSO([8], [Y]), dict(p_share=1.0)),
            (Y, Z): merge(PTSO([Y], [Z]), dict(p_share=1.0)),
        }

        # Define tracking probabilities for n=1
        s.n_subtracks_n1 = 4
        s.probs_n1 = s.probs_n0

        # Define tracking probabilities for n=2
        s.n_subtracks_n2 = 3
        s.probs_n2 = {
            (0, 2): merge(PTSO([0], [2]), dict(p_share=1.0)),
            (1, 4): merge(PTSO([1], [4]), dict(p_share=1.0)),
            (2, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(2, [3], 5))),
            (3, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(3, [2], 5))),
            (4, 6): merge(PTSO([4], [6]), dict(p_share=1.0)),
            (5, 7): merge(PTSO([5], [7]), dict(p_share=1.0)),
            (6, 9): merge(PTSO([6], [9]), dict(p_share=1.0)),
            (7, X): merge(PTSO([7], [X]), dict(p_share=1.0)),
            (8, Y): merge(PTSO([8], [Y]), dict(p_share=1.0)),
            (Y, Z): merge(PTSO([Y], [Z]), dict(p_share=1.0)),
        }

        # Define tracking probabilities for n=3
        s.n_subtracks_n3 = 2
        s.probs_n3 = {
            (0, 2): merge(PTSO([0], [2]), dict(p_share=1.0)),
            (1, 4): merge(PTSO([1], [4]), dict(p_share=1.0)),
            (2, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(2, [3], 5))),
            (3, 5): merge(PTSO([2, 3], [5]), dict(p_share=PR(3, [2], 5))),
            (4, 6): merge(PTSO([4], [6]), dict(p_share=1.0)),
            (5, 7): merge(PTSO([5], [7, 8]), dict(p_share=PR(7, [8], 5))),
            (5, 8): merge(PTSO([5], [7, 8]), dict(p_share=PR(8, [7], 5))),
            (6, 9): merge(PTSO([6], [9]), dict(p_share=1.0)),
            (7, X): merge(PTSO([7], [X]), dict(p_share=1.0)),
            (8, Y): merge(PTSO([8], [Y]), dict(p_share=1.0)),
            (X, Z): merge(PTSO([X, Y], [Z]), dict(p_share=PR(X, [Y], Z))),
            (Y, Z): merge(PTSO([X, Y], [Z]), dict(p_share=PR(Y, [X], Z))),
        }

        # Define tracking probabilities for n=4
        s.n_subtracks_n4 = 2
        s.probs_n4 = s.probs_n3

        # Define tracking probabilities for n=5
        s.n_subtracks_n5 = 1
        s.probs_n5 = s.probs_ini

        # Initialize feature tracker
        s.tracker = FeatureTracker(
            f_overlap=f_overlap,
            f_size=f_size,
            min_p_tot=0.0,
            min_p_overlap=0.0,
            min_p_size=0.0,
            connectivity=4,
            split_tracks_n=-1,
            nx=9,
            ny=9,
        )

    def track_features(s, features_tss):
        for ts, features in sorted(features_tss.items()):
            s.tracker.extend_tracks(features, ts)
        s.tracker.finish_tracks()
        return s.tracker.pop_finished_tracks()

    def collect_edge_attrs(s, tracks):
        attrs = {}
        for track in tracks:
            for eg in track.graph.es:
                vx_source = track.graph.vs[eg.source]
                vx_target = track.graph.vs[eg.target]
                fid_source = vx_source["feature"].id
                fid_target = vx_target["feature"].id
                attrs[(fid_source, fid_target)] = eg.attributes()
        return attrs

    def assert_edges_equal(s, tracks, attrs_sol):

        # Collect edge probabilities
        attrs_res = s.collect_edge_attrs(tracks)

        # Check edges
        s.assertEqual(set(attrs_res.keys()), set(attrs_sol.keys()))

        # Check tracking probabilities and overlap
        for edge, sol in sorted(attrs_sol.items()):
            for key, v_sol in sorted(sol.items()):
                v_res = attrs_res[edge][key]
                if not np.isclose(v_sol, v_res):
                    err = "edge {}: {}: expected {}, got {}".format(
                        edge, key, v_sol, v_res
                    )
                    raise AssertionError(err)

    def test_nosplit(s):
        """Don't split the track."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        s.assert_edges_equal(tracks, s.probs_ini)

    def test_split_n0(s):
        """Fully split the track into linear segments."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(0)
        s.assertEqual(len(subtracks), s.n_subtracks_n0)
        s.assert_edges_equal(subtracks, s.probs_n0)

    def test_split_n1(s):
        """Split the track into subtracks based on nt=1."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(1)
        s.assertEqual(len(subtracks), s.n_subtracks_n1)
        s.assert_edges_equal(subtracks, s.probs_n1)

    def test_split_n2(s):
        """Split the track into subtracks based on nt=2."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(2)
        s.assertEqual(len(subtracks), s.n_subtracks_n2)
        s.assert_edges_equal(subtracks, s.probs_n2)

    def test_split_n3(s):
        """Split the track into subtracks based on nt=3."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(3)
        s.assertEqual(len(subtracks), s.n_subtracks_n3)
        s.assert_edges_equal(subtracks, s.probs_n3)

    def test_split_n4(s):
        """Split the track into subtracks based on nt=4."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(4)
        s.assertEqual(len(subtracks), s.n_subtracks_n4)
        s.assert_edges_equal(subtracks, s.probs_n4)

    def test_split_n5(s):
        """Split the track into subtracks based on nt=5."""
        tracks = s.track_features(s.features_tss)
        s.assertEqual(len(tracks), 1)
        subtracks = tracks[0].split(5)
        s.assertEqual(len(subtracks), s.n_subtracks_n5)
        s.assert_edges_equal(subtracks, s.probs_n5)


if __name__ == "__main__":
    unittest.main()

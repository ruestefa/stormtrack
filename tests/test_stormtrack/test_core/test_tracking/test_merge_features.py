#!/usr/bin/env python3

# Standard library
import logging as log
import sys
import unittest
from unittest import TestCase

# Third-party
import numpy as np

# First-party
from stormtrack.core.tracking import FeatureTrack
from stormtrack.core.tracking import TrackFeatureMerger

# Local
from ...utils import feature_rectangle


# log.getLogger().addHandler(log.StreamHandler(sys.stdout))
# log.getLogger().setLevel(log.DEBUG)


class MergeFeatures_Base(TestCase):

    _next_tid = 0

    def setUp(self):

        # Check domain consistency (otherwise segfault if nx/ny too small)
        if not all(
            0 <= x < self.nx and 0 <= y < self.ny
            for (xy0, xy1, _, _), _ in self.data_features_in
            for x, y in (xy0, xy1)
        ):
            err = (
                "feature points outside domain [({}, {}), ({}, {})] "
                "defined by nx={}, ny={}; adapt nx and/or ny!"
            ).format(0, 0, self.nx - 1, self.ny - 1, self.nx, self.ny)
            raise ValueError(err)

        # Create track and features
        self.create_empty_track(self.f_size, self.f_ovlp, self.next_tid())
        self.create_features(self.data_features_in, self.data_neighbors)

        # Add features to graph as vertices
        self.set_vs_attrs()

        # Add edges to graph (links between features over time)
        self.track.graph.add_edges(self.data_es_inds_in)
        eprobs_in = self.compute_edge_probabilities_in(self.data_size_ovlp_in)
        self.track.graph.es["p_size"] = eprobs_in["p_size"]
        self.track.graph.es["p_overlap"] = eprobs_in["p_overlap"]
        self.track.graph.es["p_tot"] = eprobs_in["p_tot"]
        self.track.graph.es["p_share"] = eprobs_in["p_share"]

    def next_tid(self):
        next_tid = self.__class__._next_tid
        self.__class__._next_tid += 1
        return next_tid

    def create_empty_track(self, f_size, f_ovlp, tid):
        config = dict(f_size=f_size, f_overlap=f_ovlp)
        self.track = FeatureTrack(id_=tid, config=config)

    def create_features(self, data_features, data_neighbors):
        self.features = [feature_rectangle(*f) for f, t in data_features]
        self.types = [t for f, t in data_features]
        self.link_neighbors(data_neighbors)
        for feature, type in zip(self.features, self.types):
            feature.set_track(self.track)
        timesteps = sorted(set(f.timestep for f in self.features))
        get_fts = lambda ts: [f for f in self.features if f.timestep == ts]
        self.features_ts = [(ts, get_fts(ts)) for ts in timesteps]

    def link_neighbors(self, inds):
        for ind0, ind1 in inds:
            self.features[ind0].neighbors.append(self.features[ind1])
            self.features[ind1].neighbors.append(self.features[ind0])

    def set_vs_attrs(self):

        n_features = len(self.features)
        min_ts = min([f.timestep for f in self.features])
        max_ts = max([f.timestep for f in self.features])
        n_starts = len([f for f in self.features if f.timestep == min_ts])
        n_heads = len([f for f in self.features if f.timestep == max_ts])
        attr_head = [False] * (n_features - n_heads) + [True] * n_heads

        self.track.graph.add_vertices(len(self.features))
        self.track.graph.vs["feature"] = [f for f in self.features]
        self.track.graph.vs["name"] = [str(f.id) for f in self.features]
        self.track.graph.vs["ts"] = [f.timestep for f in self.features]
        self.track.graph.vs["type"] = [t for t in self.types]
        # self.track.graph.vs["_active_head"] = attr_head
        self.track.graph.vs["missing_predecessors"] = None
        self.track.graph.vs["missing_successors"] = None

        for i, feature in enumerate(self.features):
            feature.set_vertex(self.track.graph.vs[i])

    def compute_edge_probabilities_in(self, data):
        p_sizes, p_overlaps, p_tots, p_shares = [], [], [], []
        for f_sizes0, f_sizes1, ovlps, ind in data:

            sizes0, sizes1 = f_sizes0(), f_sizes1()
            size0, size1 = np.sum(sizes0), np.sum(sizes1)
            ovlp = np.sum(ovlps)

            p_size = self.cmp_p_size(size0, size1)
            p_ovlp = self.cmp_p_ovlp(size0, size1, ovlp)
            p_tot = self.cmp_p_tot(size0, size1, ovlp)
            p_share = self.cmp_p_share(sizes0, sizes1, ovlps, ind)

            p_sizes.append(p_size)
            p_overlaps.append(p_ovlp)
            p_tots.append(p_tot)
            p_shares.append(p_share)

        return dict(
            p_size=np.array(p_sizes),
            p_overlap=np.array(p_overlaps),
            p_tot=np.array(p_tots),
            p_share=np.array(p_shares),
        )

    def compute_edge_probabilities_out(self, data_in, data_out):
        p_sizes, p_overlaps, p_tots, p_shares = [], [], [], []
        for data_in_i, data_out_i in zip(data_in, data_out):

            # Determine wheter in- or output features
            f_name0 = str(data_out_i[0]).split()[1]
            f_name1 = str(data_out_i[1]).split()[1]
            self.assertEqual(f_name0, f_name1, "mixed in/out features")
            if "get_features_n_in" in f_name0:
                inout = "in"
                f_sizes0, f_sizes1, ovlps, ind = data_in_i
            elif "get_features_n_out" in f_name0:
                inout = "out"
                f_sizes0, f_sizes1, ovlps, ind = data_out_i

            sizes0, sizes1 = f_sizes0(), f_sizes1()
            size0, size1 = np.sum(sizes0), np.sum(sizes1)
            ovlp = np.sum(ovlps)

            p_size = self.cmp_p_size(size0, size1)
            p_ovlp = self.cmp_p_ovlp(size0, size1, ovlp)
            p_tot = self.cmp_p_tot(size0, size1, ovlp)
            p_share = self.cmp_p_share(sizes0, sizes1, ovlps, ind)

            p_sizes.append(p_size)
            p_overlaps.append(p_ovlp)
            p_tots.append(p_tot)
            p_shares.append(p_share)

        return dict(
            p_size=np.array(p_sizes),
            p_overlap=np.array(p_overlaps),
            p_tot=np.array(p_tots),
            p_share=np.array(p_shares),
        )

    def cmp_p_size(self, parent_n, child_n):
        return min([parent_n, child_n]) / max([parent_n, child_n])

    def cmp_p_ovlp(self, parent_n, child_n, ovlp_n):
        return 2 * ovlp_n / (parent_n + child_n)

    def cmp_p_tot(self, parent_n, child_n, ovlp_n):
        p_size = self.cmp_p_size(parent_n, child_n)
        p_ovlp = self.cmp_p_ovlp(parent_n, child_n, ovlp_n)
        return self.f_size * p_size + self.f_ovlp * p_ovlp

    def cmp_p_shares(self, sizes0, sizes1, ovlps):
        if len(sizes0) > len(sizes1):
            sizes0, sizes1 = sizes1, sizes0
        assert len(sizes0) == 1
        assert len(sizes1) == len(ovlps)
        size_parent, size_children = sizes0[0], sizes1
        p_tots = np.array(
            [
                self.cmp_p_tot(size_parent, size_child, ovlp)
                for size_child, ovlp in zip(size_children, ovlps)
            ]
        )
        return p_tots / p_tots.sum()

    def cmp_p_share(self, sizes0, sizes1, ovlps, ind):
        return self.cmp_p_shares(sizes0, sizes1, ovlps)[ind]

    def get_features_n_in(self, *ids):
        """Returns function to retrieve the sizes of input features.

        The function can be used at a later point once the features exist.
        """

        def fct(*, get_ids=False):
            if get_ids:
                return ids
            return [f.n for f in self.features if f.id in ids]

        return fct

    def get_features_n_out(self, *ids):
        """Returns function to retrieve the sizes of output features.

        The function can be used at a later point once the features exist.
        """

        def fct(*, get_ids=False):
            if get_ids:
                return ids
            return [f.n for f in self.track.features() if f.id in ids]

        return fct

    def run_test(self):
        TrackFeatureMerger(
            self.track, nx=self.nx, ny=self.ny, connectivity=self.connectivity
        ).run()

    def check_results(self):

        # Check no. features per timestep
        for features, sol in zip(
            self.track.features_ts(), self.n_features_ts_postmerge
        ):
            self.assertEqual(len(features), sol)

        # Check some properties of the new feature
        ind = self.new_feature_ind_ts
        new_feature = self.track.features_ts(self.ts_merge)[ind]
        sol = sum([self.features[i].n for i in self.merged_features])
        self.assertEqual(new_feature.n, sol)
        new_vertex = new_feature.vertex()

        # Check Feature-vertex associations
        self.assertFeatureVertexValid(self.features + [new_feature])
        self.assertFeatureVertexExists(self.track.features())

        # Check some feature/vertex properties
        for vertex in self.track.graph.vs:
            fid = vertex["feature"].id
            if vertex["type"] != self.data_feature_types_out[fid]:
                err = "feature {}: wrong type: expected {}, found {}".format(
                    fid, self.data_feature_types_out[fid], vertex["type"]
                )
                raise AssertionError(err)

        # Collect edges
        self.assertEqual(len(self.track.graph.es), len(self.data_es_inds_out))
        edges = {}
        id2feature = lambda id_: [f for f in self.track.features() if f.id == id_][0]
        for i0, i1 in self.data_es_inds_out:
            edge = self.track.graph.es.find(
                _between=(
                    (id2feature(i0).vertex().index,),
                    (id2feature(i1).vertex().index,),
                )
            )
            edges[(i0, i1)] = edge

        # Compare expected and actual edge probabilities
        edge_order = self.get_edge_order(self.data_size_ovlp_out)
        eprobs_sol = self.compute_edge_probabilities_out(
            self.data_size_ovlp_in, self.data_size_ovlp_out
        )
        eprobs_in = self.compute_edge_probabilities_in(self.data_size_ovlp_in)
        eprobs_res = self.extract_edge_probabilities(edge_order)
        self.assertEdgeProbabilities(eprobs_sol, eprobs_res, eprobs_in)

    def assertEdgeProbabilities(self, eprobs_sol, eprobs_res, eprobs_in):
        for key, sol in sorted(eprobs_sol.items()):
            res = eprobs_res[key]
            epi = eprobs_in[key]
            if not np.allclose(res, sol):
                err = (
                    "edge probabilities differ:\n\n{}:\n"
                    "edge {}\nin    {}\nsol   {}\nres   {}"
                ).format(
                    key,
                    "".join(
                        [
                            "{:>3}-{:<3}".format(io[0] + str(e0), io[0] + str(e1))
                            for e0, e1, io in edge_order
                        ]
                    ),
                    "  ".join(["{:5.3f}".format(p) for p in epi]),
                    "  ".join(["{:5.3f}".format(p) for p in sol]),
                    "  ".join(["{:5.3f}".format(p) for p in res]),
                )
                raise AssertionError(err)

    def get_edge_order(self, data):
        """Extract the order of edges from edge probability data table."""
        edges = []
        for (f_sizes0, f_sizes1, ovlps, ind) in data:

            # Determine wheter in- or output features
            f_name0 = str(f_sizes0).split()[1]
            f_name1 = str(f_sizes1).split()[1]
            self.assertEqual(f_name0, f_name1, "mixed in/out features")
            if "get_features_n_in" in f_name0:
                inout = "in"
            elif "get_features_n_out" in f_name0:
                inout = "out"

            # Collect feature ids for all edge combinations
            ids0 = f_sizes0(get_ids=True)
            ids1 = f_sizes1(get_ids=True)
            if len(ids0) == 1:
                edges_i = [(ids0[0], i, inout) for i in ids1]
            elif len(ids1) == 1:
                edges_i = [(i, ids1[0], inout) for i in ids0]
            else:
                raise Exception("invalid edge data")
            edges.append(edges_i[ind])

        return edges

    def extract_edge_probabilities(self, edge_order):
        probs = dict(p_size=[], p_overlap=[], p_tot=[], p_share=[])
        for ind0, ind1, inout in edge_order:
            vx0 = self.track.graph.vs.find(str(ind0))
            vx1 = self.track.graph.vs.find(str(ind1))
            edge = self.track.graph.es.find(_between=((vx0.index,), (vx1.index,)))
            for key, val in probs.items():
                val.append(edge[key])
        return {key: np.array(val) for key, val in probs.items()}

    def assertFeatureVertexValid(self, features):
        """Check that feature vertices are valid if they exist."""
        for feature in features:
            try:
                feature.vertex()
            except Exception as e:
                err = "Feature {}: invalid vertex (error: {})".format(feature.id, e)
                raise AssertionError(err)

    def assertFeatureVertexExists(self, features):
        """Check that every feature is associated with a vertex."""
        for feature in features:
            if feature.vertex() is None:
                err = "feature {} is not associated with a vertex"
                raise AssertionError(err)


class MergeFeatures(MergeFeatures_Base):

    # Call super().setUp() in tests
    def setUp(s):
        pass

    def test_1(s):

        # Set some parameters
        s.nx, s.ny = 10, 14
        s.connectivity = 8
        s.f_size, s.f_ovlp = 0.5, 0.5

        #
        # 13 - - - - - - / - - - - - - / - - - - - -
        # 12 - - - 0 0 0 / - - - - - - / - - - - - -
        # 11 - - - 0 0 0 / - - 3 3 - - / - 5 5 5 - -
        # 10 - - - 0 0 0 / - - 3 3 - - / - 5 5 5 - -
        #  9 - - - - - - / - - 3 3 - - / - 5 5 5 - -
        #  8 - - 1 1 1 - / - - 3 3 - - / - - - - - -
        #  7 - - 1 1 1 - / - 4 4 4 4 - / - 6 6 6 6 -
        #  6 - - 1 1 1 - / - 4 4 4 4 - / - 6 6 6 6 -
        #  5 - - 1 1 1 - / - 4 4 4 4 - / - - - - - -
        #  4 - - - - - - / - 4 4 4 4 - / - 7 7 7 7 -
        #  3 - 2 2 2 2 - / - 4 4 4 4 - / - 7 7 7 7 -
        #  2 - 2 2 2 2 - / - 4 4 4 4 - / - 7 7 7 7 -
        #  1 - 2 2 2 2 - / - - - - - - / - 7 7 7 7 -
        #  0 - - - - - - / - - - - - - / - - - - - -
        #    0 1 2 3 4 5 / 1 3 4 5 6 7 / 4 5 6 7 8 9
        #      (ts 0)        (ts 1)        (ts 2)
        #
        s.n_features_ts_postmerge = [3, 1, 3]
        s.new_feature_ind_ts = 0

        # Define features
        s.data_features_in = [
            (((3, 10), (5, 12), 0, 0), "genesis"),
            (((2, 5), (4, 8), 1, 0), "genesis"),
            (((1, 1), (4, 3), 2, 0), "genesis"),
            (((4, 8), (5, 11), 3, 1), "continuation"),
            (((3, 2), (6, 7), 4, 1), "merging/splitting"),
            (((5, 9), (7, 11), 5, 2), "lysis"),
            (((5, 6), (8, 7), 6, 2), "lysis"),
            (((5, 1), (8, 4), 7, 2), "lysis"),
        ]
        s.data_neighbors = [(3, 4)]
        s.data_feature_types_out = {
            0: "genesis",
            1: "genesis",
            2: "genesis",
            3: "merging/splitting",
            5: "lysis",
            6: "lysis",
            7: "lysis",
        }

        # Feature index pairs for edges (note that the merged feature will
        # inherit the lowest id of the original features)
        s.data_es_inds_in = [(0, 3), (1, 4), (2, 4), (3, 5), (4, 6), (4, 7)]
        s.data_es_inds_out = [(0, 3), (1, 3), (2, 3), (3, 5), (3, 6), (3, 7)]

        # Data for successor probabilities
        # Format: ((<fids0>), (<fids1>), <ovlp>, <ind>)
        fi, fo = s.get_features_n_in, s.get_features_n_out
        s.data_size_ovlp_in = [
            (fi(0), fi(3), (4,), 0),  # 0 <-> 3
            (fi(1, 2), fi(4), (6, 4), 0),  # [12]<-> 4 / 0
            (fi(1, 2), fi(4), (6, 4), 1),  # [12]<-> 4 / 1
            (fi(3), fi(5), (3,), 0),  # 3 <-> 5
            (fi(4), fi(6, 7), (4, 6), 0),  # 4 <->[67] / 0
            (fi(4), fi(6, 7), (4, 6), 1),  # 4 <->[67] / 1
        ]
        s.data_size_ovlp_out = [
            (fo(0, 1, 2), fo(3), (4, 7, 4), 0),  # [012]<-> 3 / 0
            (fo(0, 1, 2), fo(3), (4, 7, 4), 1),  # [012]<-> 3 / 1
            (fo(0, 1, 2), fo(3), (4, 7, 4), 2),  # [012]<-> 3 / 2
            (fo(3), fo(5, 6, 7), (3, 4, 6), 0),  # 3 <->[567] / 0
            (fo(3), fo(5, 6, 7), (3, 4, 6), 1),  # 3 <->[567] / 1
            (fo(3), fo(5, 6, 7), (3, 4, 6), 2),  # 3 <->[567] / 2
        ]

        super().setUp()

        # Some parameter for the checks
        s.ts_merge = 1
        s.merged_features = [3, 4]
        s.merged_feature_type = "merging/splitting"

        # Run test
        s.run_test()
        s.check_results()

    def test_2(s):

        # Set some parameters
        s.nx, s.ny = 10, 18
        s.connectivity = 8
        s.f_size, s.f_ovlp = 0.5, 0.5

        #
        # 18 - - - - - - / - - - - - / - - - - - -
        # 17 - - 0 0 0 - / - 4 4 - - / - - - - - -
        # 16 - - 0 0 0 - / - 4 4 - - / - 8 8 8 8 -
        # 15 - - 0 0 0 - / - - - - - / - 8 8 8 8 -
        # 14 - - 0 0 0 - / - 5 5 5 - / - 8 8 8 8 -
        # 13 - - - - - - / - 5 5 5 - / - - - - - -
        # 12 - - - - - - / - 5 5 5 - / - 9 9 9 - -
        # 11 - 1 1 1 1 1 / - 6 6 6 - / - 9 9 9 - -
        # 10 - 1 1 1 1 1 / - 6 6 6 - / - 9 9 9 - -
        #  9 - 1 1 1 1 1 / - 6 6 6 - / - 9 9 9 - -
        #  8 - - - - - - / - 6 6 6 - / - 9 9 9 - -
        #  7 - - 2 2 2 - / - 6 6 6 - / - 9 9 9 - -
        #  6 - - 2 2 2 - / - 6 6 6 - / - 9 9 9 - -
        #  5 - - 2 2 2 - / - 6 6 6 - / - 9 9 9 - -
        #  4 - - 2 2 2 - / - 6 6 6 - / - 9 9 9 - -
        #  3 - - - - - - / - 6 6 6 - / - 9 9 9 - -
        #  2 - - 3 3 3 - / - 7 7 7 - / - 9 9 9 - -
        #  1 - - 3 3 3 - / - 7 7 7 - / - - - - - -
        #  0 - - - - - - / - - - - - / - - - - - -
        #    0 1 2 3 4 5 / 1 3 4 5 6 / 3 4 5 6 7 8
        #      (ts 0)        (ts 1)      (ts 2)
        #
        s.n_features_ts_postmerge = [4, 2, 2]
        s.new_feature_ind_ts = 1

        # Define features: (((x0, y0), (x1, y1), id, ts), type)
        s.data_features_in = [
            (((2, 14), (4, 17), 0, 0), "genesis/splitting"),
            (((1, 9), (5, 11), 1, 0), "genesis"),
            (((2, 4), (4, 7), 2, 0), "genesis"),
            (((2, 1), (4, 2), 3, 0), "genesis"),
            (((3, 16), (4, 17), 4, 1), "continuation"),
            (((3, 12), (5, 14), 5, 1), "continuation"),
            (((3, 3), (5, 11), 6, 1), "merging"),
            (((3, 1), (5, 2), 7, 1), "continuation"),
            (((4, 14), (7, 16), 8, 2), "merging/lysis"),
            (((4, 2), (6, 12), 9, 2), "merging/lysis"),
        ]
        s.data_neighbors = [(5, 6), (6, 7)]
        s.data_feature_types_out = {
            0: "genesis/splitting",
            1: "genesis",
            2: "genesis",
            3: "genesis",
            4: "continuation",
            5: "merging/splitting",
            8: "merging/lysis",
            9: "lysis",
        }

        # Feature index pairs for edges (note that the merged feature will
        # inherit the lowest id of the original features)
        s.data_es_inds_in = [
            (0, 4),
            (0, 5),
            (1, 6),
            (2, 6),
            (3, 7),
            (4, 8),
            (5, 8),
            (6, 9),
            (7, 9),
        ]
        s.data_es_inds_out = [
            (0, 4),
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 8),
            (5, 8),
            (5, 9),
        ]

        # Data for successor probabilities
        # Format: ((<fids0>), (<fids1>), <ovlps> <ind>)
        fi, fo = s.get_features_n_in, s.get_features_n_out
        s.data_size_ovlp_in = [
            (fi(0), fi(4, 5), (4, 2), 0),  # 0 <->[45] / 0
            (fi(0), fi(4, 5), (4, 2), 1),  # 0 <->[45] / 1
            (fi(1, 2), fi(6), (9, 8), 0),  # [12]<-> 6 / 0
            (fi(1, 2), fi(6), (9, 8), 1),  # [12]<-> 6 / 1
            (fi(3), fi(7), (4,), 0),  # 3 <-> 7
            (fi(4, 5), fi(8), (1, 2), 0),  # [45]<-> 8 / 0
            (fi(4, 5), fi(8), (1, 2), 1),  # [45]<-> 8 / 1
            (fi(6, 7), fi(9), (18, 2), 0),  # [67]<-> 9 / 0
            (fi(6, 7), fi(9), (18, 2), 1),  # [67]<-> 9 / 1
        ]
        #
        # The output successor probabilities are tricky in this case,
        # because the merging results in many-to-many-relationships
        # ([0123]<->[45] and [45]<->[89]) which would otherwise not
        # exists with our tracking algorithm!
        # We handle these pragmatically by recomputing the probabilities
        # for all edges directly involved with the merged feature while
        # ignoring other, indirectly involved edges (0<->4 and 4<->8).
        # This means that the edges are not all consistent anymore.
        #
        s.data_size_ovlp_out = [
            (fi(0), fi(4, 5), (4, 2), 0),  # 0 <->[45] / 0
            (fo(0, 1, 2, 3), fo(5), (2, 9, 8, 4), 0),  # [0123]<-> 5 / 0
            (fo(0, 1, 2, 3), fo(5), (2, 9, 8, 4), 1),  # [0123]<-> 5 / 1
            (fo(0, 1, 2, 3), fo(5), (2, 9, 8, 4), 2),  # [0123]<-> 5 / 2
            (fo(0, 1, 2, 3), fo(5), (2, 9, 8, 4), 3),  # [0123]<-> 5 / 3
            (fi(4), fi(8), (3,), 0),  # 4 <-> 8
            (fo(5), fo(8, 9), (2, 22), 0),  # 5 <->[89] / 0
            (fo(5), fo(8, 9), (2, 22), 1),  # 5 <->[89] / 1
        ]

        super().setUp()

        # Some parameter for the checks
        s.ts_merge = 1
        s.merged_features = [5, 6, 7]

        # Run test
        s.run_test()
        s.check_results()

    def test_3(s):

        # Set some parameters
        s.nx, s.ny = 10, 14
        s.connectivity = 8
        s.f_size, s.f_ovlp = 0.5, 0.5

        #
        #  8 - - - - - - / - - - - - -
        #  7 - - - - - - / - 1 1 1 1 -
        #  6 - 0 0 0 0 - / - 1 1 1 1 -
        #  5 - 0 0 0 0 - / - 1 1 1 1 -
        #  4 - 0 0 0 0 - / - - 2 2 2 -
        #  3 - 0 0 0 0 - / - - 2 2 2 -
        #  2 - 0 0 0 0 - / - - 2 2 2 -
        #  1 - 0 0 0 0 - / - - 2 2 2 -
        #  0 - - - - - - / - - - - - -
        #    0 1 2 3 4 5 / 1 3 4 5 6 7
        #      (ts 0)        (ts 1)
        #
        s.n_features_ts_postmerge = [1, 1]
        s.new_feature_ind_ts = 0

        # Define features: (((x0, y0), (x1, y1), id, ts), type)
        s.data_features_in = [
            (((1, 1), (4, 6), 0, 0), "genesis/splitting"),
            (((3, 5), (6, 7), 1, 1), "lysis"),
            (((4, 1), (6, 4), 2, 1), "lysis"),
        ]
        s.data_neighbors = [(1, 2)]
        s.data_feature_types_out = {
            0: "genesis",
            1: "lysis",
        }

        # Feature index pairs for edges (note that the merged feature will
        # inherit the lowest id of the original features)
        s.data_es_inds_in = [(0, 1), (0, 2)]
        s.data_es_inds_out = [(0, 1)]

        # Data for successor probabilities
        # Format: ((<fids0>), (<fids1>), <ovlp>, <ind>)
        fi, fo = s.get_features_n_in, s.get_features_n_out
        s.data_size_ovlp_in = [
            (fi(0), fi(1, 2), (4, 4), 0),  # 0 <->[12] / 0
            (fi(0), fi(1, 2), (4, 4), 1),  # 0 <->[12] / 1
        ]
        s.data_size_ovlp_out = [
            (fo(0), fo(1), (8,), 0),  # 0 <->[12] / 0
        ]

        super().setUp()

        # Some parameter for the checks
        s.ts_merge = 1
        s.merged_features = [1, 2]

        # Run test
        s.run_test()
        s.check_results()


if __name__ == "__main__":
    unittest.main()

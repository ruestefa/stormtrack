#!/usr/bin/env python3

# Standard library
import os
import unittest
import sys
from pprint import pprint as pp
from unittest import TestCase

# Third-pary
import numpy as np

# First-party
from stormtrack.extra.front_surgery import front_surgery_isolated
from stormtrack.extra.front_surgery import front_surgery_temporal

# Local
from ..utils import TestFeatures_Base


class Isolated(TestFeatures_Base):
    """Run the first major step of front surgery on an isolated field.

    Input:
    A list of front subfeatures which have already been internally split,
    i.e. not the raw output of the initial front identification tool.
    The features are internally split, i.e. can touch each other.

    Output:
    Multiple categorized lists of front features (clutter and non-clutter).
    Features in the same category have been merged, i.e. the remaining
    distinct features don't touch each other. However, features in different
    categories may touch each other.

    Algorithm:
     - Sort features into three size categories using the two thresholds
     - identify_obvious_clutter: All small features w/o large neighbors
     - merge_features_in_categories: Merge all neighbors in categories
     - merge_categories: Merge small into medium category
     - identify_clutter_neighbors: All medium with any clutter neighbors
     - merge_categories: Merge medium into large category
     - merge_features_in_categories: Merge all neighbors in categories
     - separate_big_clutter_clusters: Separate largest clutter features
       (above upper threshold) from others (exclude from further steps)

    """

    def setUp(self):

        # fmt: on
        A =  0; B =  1; C =  2; D =  3; E =  4; F =  5; G =  6; H =  7; I =  8; J =  9;
        K = 10; L = 11; M = 12; N = 13; O = 14; P = 15; Q = 16; R = 17; S = 18; T = 19;
        U = 20; V = 21; W = 22; X = 23; Y = 24; Z = 25;
        # fmt: on

        _ = np.nan
        # fmt: off
        self.obj_ids_in_8c = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,_, _,_,_,_,_, _,B,B,B,_, _,_,_,Z,_], #  3
                [_,_,_,_,A, A,A,A,A,A, C,C,C,C,C, C,_,Z,Z,X], #  2
                [_,_,_,A,A, A,A,A,A,A, C,C,_,_,_, _,_,Y,X,X], #  1
                [_,_,A,A,A, _,_,_,_,_, _,_,_,_,_, _,_,X,X,X], # 10

                [_,A,A,A,_, _,_,H,_,_, _,_,_,_,_, _,_,X,X,X], #  9
                [A,A,A,_,_, _,_,_,_,N, N,J,J,_,_, _,X,X,V,_], #  8
                [_,A,_,_,E, _,O,O,O,J, J,J,M,M,_, _,U,U,V,_], #  7
                [_,_,_,_,E, _,P,J,J,J, K,L,M,R,R, _,W,U,V,_], #  6
                [_,_,_,D,E, _,I,I,_,G, K,L,Q,Q,_, _,W,T,T,_], #  5

                [_,_,_,D,D, _,_,F,G,G, Q,Q,Q,_,_, _,T,T,T,_], #  4
                [_,_,D,D,D, _,_,F,Q,Q, Q,_,_,_,_, S,S,T,_,_], #  3
                [_,_,D,D,_, _,_,Q,Q,Q, _,_,_,_,S, S,S,_,_,_], #  2
                [_,_,D,D,_, _,_,_,_,_, _,_,S,S,S, S,_,_,_,_], #  1
                [_,_,D,_,_, _,_,_,_,_, _,_,S,S,_, _,_,_,_,_], #  0
            ]  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        self.nx, self.ny = self.obj_ids_in_8c.shape
        self.fld = np.zeros([self.nx, self.ny])

        # Count feature sizes and define thresholds
        # fmt: off
        lens = [
            23,  3,  8, 11,  3,  2,  3,  1,  2,  8,  2,  2,  3, # A-M (13)
             2,  3,  1, 11,  2, 11,  6,  3,  3,  2, 11,  1,  3, # N-Z (13)
        ]
        # fmt: on
        fld = self.obj_ids_in_8c
        lens_check = [len(fld[fld == i]) for i in range(26)]
        assert(lens == lens_check) # double-check manual counts
        # +----+------------------------+---------------+
        # |  N | FEATURES               | RANGE         |
        # +----+------------------------+---------------|
        # |  1 | H, P, Y                |               |
        # |  2 | F, I, K, L, N, R, W    |       n <  4  |
        # |  3 | B, E, G, M, O, U, V, Z |               |
        # +----+------------------------+---------------+
        # |  6 | T                      |  4 <= n < 10  |
        # |  8 | C, J                   |               |
        # +----+------------------------+---------------+
        # | 11 | D, Q, S, X             | 10 <= n       |
        # | 23 | A                      |               |
        # +----+------------------------+---------------+
        self.thresholds = [4, 10]

        # Define input features
        self.inds_features_in_8c = [[i] for i in range(26)]
        ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        f = lambda abc: [ABC.index(x) for x in abc]  # noqa
        # fmt: off
        self.inds_neighbors_in_8c = [
            f("C"), f("C"), f("AB"), f("E"), f("D"), f("GIQ"),  # A-F (6)
            f("FIJKQ"), f(""), f("FGJP"), f("GIKLMNOP"),        # F-J (4)
            f("GJLQ"), f("JKMQ"), f("JLQR"), f("JO"), f("JNP"), # K-O (5)
            f("IJO"), f("FGKLMR"), f("MQ"), f("T"), f("SUVQ"),  # P-T (5)
            f("TVWX"), f("TUX"), f("TU"), f("UVYZ"), f("XZ"),   # U-Y (5)
            f("XY"),                                            # Z   (1)
        ]
        # fmt: on

        # Define output features (sorted into categories below)
        #
        # - categorize features: [see table above]
        #
        # - identify_obvious_clutter:
        #    -> clutter0 : H(1)
        #    -> clutter1 : B(3), I(2), N(2), O(3), P(1), W(2)
        #    -> small    : E(3), F(2), G(3), K(2), L(2), M(3), R(2),
        #                  U(3), V(3), Y(1), Z(3)
        #    -> medium   : T(6), C(8), J(8)
        #    -> large    : A(23), D(11), Q(11), S(11), X(11)
        #
        # - merge_features_in_categories:
        #    -> clutter0 : H(1)
        #    -> clutter1 : B(3), INOP(8), W(2)
        #    -> small    : E(3), FGKLMR(14), UV(6), YZ(4)
        #    -> medium   : T(6), C(8), J(8)
        #    -> large    : A(23), D(11), Q(11), S(11), X(11)
        #
        # - merge_categories:
        #    -> clutter0 : H(1)
        #    -> clutter1 : B(3), INOP(8), W(2)
        #    -> medium   : E(3), FGKLMR(14), UV(6), YZ(4), T(6), C(8), J(8)
        #    -> large    : A(23), D(11), Q(11), S(11), X(11)
        #
        # - identify_clutter_neighbors:
        #    -> clutter0 : H(1)
        #    -> clutter1 : B(3), INOP(8), W(2), FGKLMR(14), UV(6), T(6),
        #                  C(8), J(8)
        #    -> medium   : E(3), YZ(4)
        #    -> large    : A(23), D(11), Q(11), S(11), X(11)
        #
        # - merge_categories:
        #    -> clutter0 : H(1)
        #    -> clutter1 : B(3), C(8), J(8), INOP(8), FGKLMR(14),
        #                  UV(6), W(2), T(6),
        #    -> large    : A(23), D(11), Q(11), S(11), X(11), E(3), YZ(4)
        #
        # - merge_features_in_categories:
        #    -> clutter0 : H(1)
        #    -> clutter1 : BC(11), FGIJKLMNOPR(30), UVWT(14)
        #    -> large    : A(23), DE(14), Q(11), S(11), XYZ(15)
        #
        # - separate_big_clutter_clusters:
        #    -> clutter0 : H(1), BC(11), FGIJKLMNOPR(30), UVWT(14)
        #    -> large    : A(23), DE(14), Q(11), S(11), XYZ(15)
        #
        # fmt: off
        self.inds_features_out_8c = [
            [B, C], [H], [F, G, I, J, K, L, M, N, O, P, R], [T, U, V, W], [A], [D, E],
            [Q], [S], [X, Y, Z],
        ]
        # fmt: on
        BC, H_, FGIJKLMNOPR, TUVW, A_, DE, Q_, S_, XYZ = range(9)
        # fmt: off
        self.inds_neighbors_out_8c = [
            [A_],           # [0] BC
            [],             # [1] H
            [Q_],           # [2] FGIJKLMNOPR
            [S_, XYZ],      # [3] TUVW
            [BC],           # [4] A
            [],             # [5] DE
            [FGIJKLMNOPR],  # [6] Q
            [TUVW],         # [7] S
            [TUVW],         # [8] XYZ
        ]
        # fmt: on

        self.setUpFeatures("in_8c", "out_8c")

        # Categorize output features
        self.features_out_cat = {
                "clutter0": [self.features_out_8c[i] for i in (0, 1, 2, 3)],
                "clutter1":  [],
                "small":     [],
                "medium":    [],
                "large":     [self.features_out_8c[i] for i in (4, 5, 6, 7, 8)],
            }

    def test_8c(self):
        features_cat = front_surgery_isolated(
                self.features_in_8c,
                self.thresholds,
                self.nx, self.ny,
                connectivity=8,
            )
        for category, sol in sorted(self.features_out_cat.items()):
            res = features_cat[category]
            # SR_TMP < sort by size because order of IDs is not equivalent...
            sol = sorted(sol, key=lambda f: f.n)
            res = sorted(res, key=lambda f: f.n)
            # SR_TMP >
            self.assertFeaturesEqual(res, sol,
                    check_boundaries=False,
                    check_shared_pixels=False,
                )


class Temporal_Base(TestFeatures_Base):
    """Run the second major front surgery step over two to three timesteps.

    Input:
    Two to three fields at different timesteps (NOW mandatorily, plus one or
    both of OLD and NEW) after they have all gone through the "isolated" step.
    "Fields" here refers to categorized lists of features. Note that at this
    point, there should be both "clutter1" and "clutter0" features, as well as
    "large" features, but no more "small" and "medium" features.

    Output:
    Adapted lists of features. Recovered features have been moved to "large",

    Algorithm:
     - Recover all <src> features at NOW with all large features at <ts>:
        - If the feature exceeds the max. size, skip it (optimization step)
        - Compute total overlap with all large features at <ts>
        - Compute the relative overlap wrt. the size of the feature
        - Recover the front if the relative overlap is sufficient
     - <src> refers to "clutter1" and "clutter0" (in that order)
     - <ts> refers to OLD and/or NEW (depending on what'self available)
     - Finally, merge all adjacent features in the same categories

    """

    def setUp(self):

        _ = np.nan
        # fmt: off
        self.obj_ids_in_old_8c = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,_, _,3,3,3,3, _,_,_,_,_, _,_,_,_,_], #  4
                [_,1,_,_,_, _,_,3,3,_, _,_,_,_,_, _,_,_,_,_], #  3
                [1,1,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  2
                [1,1,_,_,_, _,_,_,_,_, _,0,0,_,_, _,_,_,_,_], #  1
                [1,1,_,_,_, _,_,_,_,0, 0,0,_,_,_, _,_,_,_,_], # 10

                [1,1,_,_,_, _,_,0,0,0, _,_,_,_,_, _,_,_,_,_], #  9
                [1,_,_,_,_, _,0,0,0,0, _,_,_,_,2, _,_,_,_,_], #  8
                [1,_,_,_,_, 0,0,0,0,_, _,_,_,_,2, _,_,_,_,_], #  7
                [1,_,_,_,0, 0,0,0,_,_, _,_,_,2,2, _,_,_,_,_], #  6
                [1,_,_,_,0, 0,0,_,_,_, _,_,_,2,2, _,_,_,_,_], #  5

                [_,_,_,0,0, 0,0,_,_,_, _,9,_,2,2, _,_,_,_,_], #  4
                [_,_,_,0,0, 0,_,_,_,_, _,9,9,2,9, _,_,_,_,_], #  3
                [_,_,_,0,0, _,_,_,_,_, _,9,9,9,9, _,_,_,_,_], #  2
                [_,_,0,0,_, _,_,_,_,_, _,_,9,9,9, 9,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,9,9, _,_,_,_,_], #  0
            ]  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        self.nx, self.ny = self.obj_ids_in_old_8c.shape
        self.fld = np.zeros([self.nx, self.ny])
        # fmt: off
        self.obj_ids_in_now_8c = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,2,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,2,2,_, _,6,6,6,6, _,_,_,_,_, _,_,_,_,_], #  3
                [_,_,2,2,_, _,_,6,6,_, _,_,_,_,1, _,_,_,_,_], #  2
                [_,2,2,2,_, _,_,_,_,_, _,_,1,1,1, _,_,_,_,_], #  1
                [_,2,2,_,_, _,_,_,_,_, 1,1,1,1,_, _,7,7,_,_], # 10

                [_,2,2,_,_, _,_,_,_,1, 1,1,_,_,_, _,_,7,7,_], #  9
                [2,2,2,_,_, _,_,_,9,9, 9,1,_,_,_, _,_,7,7,7], #  8
                [2,2,2,_,_, _,_,9,9,9, 9,_,_,_,_, _,_,_,7,7], #  7
                [2,2,_,_,_, _,0,0,0,0, _,_,_,_,_, _,_,_,_,7], #  6
                [2,2,_,_,_, 0,0,0,0,_, _,_,_,_,_, _,_,_,_,_], #  5

                [_,_,_,_,_, 0,0,0,_,_, _,_,8,8,8, _,_,_,_,_], #  4
                [_,_,_,_,0, 0,0,0,_,_, _,8,8,8,8, 8,_,_,_,_], #  3
                [_,_,_,_,0, 0,_,_,_,_, _,8,8,8,8, 8,_,_,_,_], #  2
                [_,_,_,_,_, _,_,_,_,_, _,_,_,8,8, _,_,_,_,_], #  1
                [_,_,_,_,_, _,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ]  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        assert self.obj_ids_in_now_8c.shape == (self.nx, self.ny)
        # fmt: off
        self.obj_ids_in_new_8c = np.array(
            [  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
                [_,_,_,_,2, 2,_,3,3,_, _,_,_,_,_, _,_,_,_,_], #  4
                [_,_,_,_,2, 2,_,3,3,3, _,_,_,_,1, _,_,_,_,_], #  3
                [_,_,_,2,2, _,_,3,3,3, _,_,_,1,1, _,_,_,_,_], #  2
                [_,_,_,2,2, _,_,_,_,_, _,_,1,1,1, _,_,_,_,_], #  1
                [_,_,2,2,2, _,_,_,_,_, _,_,1,1,_, _,_,_,_,_], # 10

                [_,_,2,2,_, _,_,_,_,_, 1,1,1,_,_, _,4,4,4,_], #  9
                [_,_,2,2,_, _,_,_,_,_, 1,1,_,_,_, _,_,4,4,4], #  8
                [_,_,2,2,_, _,_,_,_,9, 9,9,_,_,_, _,_,4,4,4], #  7
                [_,2,2,_,_, _,_,_,_,9, 9,_,_,_,_, _,_,_,4,4], #  6
                [_,2,2,_,_, _,_,_,0,0, _,_,_,_,_, _,_,_,4,4], #  5

                [_,2,_,_,_, _,_,0,0,_, _,_,5,5,5, 5,_,_,_,4], #  4
                [_,_,_,_,_, _,0,0,0,_, _,5,5,5,5, 5,_,_,_,4], #  3
                [_,_,_,_,_, 0,0,0,_,_, _,_,5,5,5, 5,_,_,_,_], #  2
                [_,_,_,_,_, 0,0,_,_,_, _,_,_,_,5, _,_,_,_,_], #  1
                [_,_,_,_,_, 0,_,_,_,_, _,_,_,_,_, _,_,_,_,_], #  0
            ]  # 0 1 2 3 4  5 6 7 8 9 10 1 2 3 4 15 6 7 8 9
        ).T[:, ::-1]
        # fmt: on
        assert self.obj_ids_in_new_8c.shape == (self.nx, self.ny)

        self.inds_features_in_old_8c = [[i] for i in (0, 1, 2, 3, 9)]
        self.inds_neighbors_in_old_8c = [[], [], [4], [], [2]]

        self.inds_features_in_now_8c = [[i] for i in (0, 1, 2, 6, 7, 8, 9)]
        self.inds_neighbors_in_now_8c = [[4], [4], [], [], [], [], [0, 1]]

        self.inds_features_in_new_8c = [[i] for i in (0, 1, 2, 3, 4, 5, 9)]
        self.inds_neighbors_in_new_8c = [[6], [6], [], [], [], [], [0, 1]]

        self.setUpFeatures("in_new_8c", "in_now_8c", "in_old_8c")

        #
        # Categories:
        #   -> clutter0 : 4:7(10), 5:8(15)
        #   -> clutter1 : 3:6(6), 6:9(7)
        #   -> large    : 0:0(17), 1:1(12), 2:2(22)
        #

        # Categorize features
        f = self.features_in_old_8c
        self.features_in_cat_old_8c = {
                "clutter0" : [f[4]],
                "clutter1" : [],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[0], f[1], f[2], f[3]],
            }
        f = self.features_in_now_8c
        self.features_in_cat_now_8c = {
                "clutter0" : [f[4], f[5]],
                "clutter1" : [f[3], f[6]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[0], f[1], f[2]],
            }
        f = self.features_in_new_8c
        self.features_in_cat_new_8c = {
                "clutter0" : [],
                "clutter1" : [f[6]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[0], f[1], f[2], f[3], f[4], f[5]],
            }

#SR_TODO Turn one of the big clutter clusters (OLD or NEW) into non-clutter
#SR_TODO and set max_size for this test such that the results don't change,
#SR_TODO and then add another test (with min_overlap==0.8 just to change
#SR_TODO things up on that front as well) with max_size=-1 where the big
#SR_TODO clutter feature is unclutterized!
class Temporal_MinOverlap05_MaxSize12(Temporal_Base):
    """ Setup: min_overlap == 0.5 and max_size == 12 """

    def setUp(self):
        super().setUp()

        self.obj_ids_in_now_old_8c = self.obj_ids_in_now_8c
        self.obj_ids_in_now_new_8c = self.obj_ids_in_now_8c
        self.obj_ids_in_now_all_8c = self.obj_ids_in_now_8c

        #
        # - If the feature exceeds the max. size, skip it (optimization step)
        #   -> NEW: - 5:8 overlap 13/15=0.86 w/ 5:5(14) SKIP 15>12
        #
        # - Recover the front if the relative overlap is sufficient
        #   -> OLD: - 6:9 overlap 4/7=0.57 w/ 0:0
        #   -> NEW: - 3:6 overlap 5/6=0.83 w/ 3:3
        #           - 4:7 overlap 8/10=0.8 w/ 4:4
        #
        # - Finally, merge all adjacent features in the same categories
        #   -> OLD:
        #       -> clutter0 : 4:7(10), 5:8(15)
        #       -> clutter1 : 3:6(6)
        #       -> large    : 016:019(36), 2:2(22)
        #   -> NEW:
        #       -> clutter0 : 5:8(15)
        #       -> clutter1 : 6:9(7)
        #       -> large    : 0:0(17), 1:1(12), 2:2(22), 3:6(6), 4:7(10)
        #   -> ALL:
        #       -> clutter0 : 5:8(15)
        #       -> large    : 016:019(36), 2:2(22), 3:6(6), 4:7(10)
        #

        self.inds_features_out_now_old_8c = [[7], [8], [6], [0, 1, 9], [2]]
        self.inds_neighbors_out_now_old_8c = [[], [], [], [], []]

        self.inds_features_out_now_new_8c = [[8], [9], [0], [1], [2], [6], [7]]
        self.inds_neighbors_out_now_new_8c = [[], [0, 1], [6], [6], [], [], []]

        self.inds_features_out_now_all_8c = [[8], [0, 1, 9], [2], [6], [7]]
        self.inds_neighbors_out_now_all_8c = [[], [], [], [], []]

        self.setUpFeatures("out_now_old_8c", "out_now_new_8c", "out_now_all_8c")

        # Categorize output features
        f = self.features_out_now_old_8c
        self.features_out_cat_now_old_8c = {
                "clutter0" : [f[0], f[1]],
                "clutter1" : [f[2]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[3], f[4]],
            }
        f = self.features_out_now_new_8c
        self.features_out_cat_now_new_8c = {
                "clutter0" : [f[0]],
                "clutter1" : [f[1]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[2], f[3], f[4], f[5], f[6]],
            }
        f = self.features_out_now_all_8c
        self.features_out_cat_now_all_8c = {
                "clutter0" : [f[0]],
                "clutter1" : [],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[1], f[2], f[3], f[4]],
            }

        # Keyword arguments for front_surgery_temporal
        self.kwargs = {
                "nx"               : self.nx,
                "ny"               : self.ny,
                "min_overlap"      : 0.5,
                "max_size"         : 12,
                "connectivity"     : 8,
                "merge_nonclutter" : True,
            }

    def test_both_8c(self):
        features_cat = front_surgery_temporal(
                self.features_in_cat_new_8c,
                self.features_in_cat_now_8c,
                self.features_in_cat_old_8c,
                **self.kwargs
            )
        features_sol = self.features_out_cat_now_all_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )

    def test_new_8c(self):
        features_cat = front_surgery_temporal(
                self.features_in_cat_new_8c,
                self.features_in_cat_now_8c,
                None,
                **self.kwargs
            )
        features_sol = self.features_out_cat_now_new_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )

    def test_old_8c(self):
        features_cat = front_surgery_temporal(
                None,
                self.features_in_cat_now_8c,
                self.features_in_cat_old_8c,
                **self.kwargs
                    )
        features_sol = self.features_out_cat_now_old_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )

class Temporal_MinOverlap08_MaxSizeM1(Temporal_Base):
    """ Setup: min_overlap == 0.8 and max_size == -1 """

    def setUp(self):
        super().setUp()

        self.obj_ids_in_now_old_8c = self.obj_ids_in_now_8c
        self.obj_ids_in_now_new_8c = self.obj_ids_in_now_8c
        self.obj_ids_in_now_all_8c = self.obj_ids_in_now_8c

        #
        # - If the feature exceeds the max. size, skip it (optimization step)
        #   -> skipped (max_size == -1)
        #
        # - Recover the front if the relative overlap is sufficient
        #   -> NEW: - 3:6 overlap 5/6=0.83 (w/ 3:3)
        #           - 4:7 overlap 8/10=0.8 (w/ 4:4)
        #
        # - Finally, merge all adjacent features in the same categories
        #   -> OLD:
        #       -> clutter0 : 4:7(10), 5:8(15)
        #       -> clutter1 : 3:6(6), 6:9(7)
        #       -> large    : 0:0(17), 1:1(12), 2:2(22)
        #   -> NEW:
        #       -> clutter1 : 6:9(7)
        #       -> large    : 0:0(17), 1:1(12), 2:2(22), 3:6(6), 4:7(10)
        #                     5:8(15)
        #   -> ALL:
        #       -> clutter1 : 6:9(7)
        #       -> large    : 0:0(17), 1:1(12), 2:2(22), 3:6(6), 4:7(10),
        #                     5:8(15)
        #

        self.inds_features_out_now_old_8c = [[7], [8], [6], [9], [0], [1], [2]]
        self.inds_neighbors_out_now_old_8c = [[], [], [], [], [], [], []]

        self.inds_features_out_now_new_8c = [[9], [0], [1], [2], [6], [7], [8]]
        self.inds_neighbors_out_now_new_8c = [[], [0, 1], [6], [6], [], [], []]

        self.inds_features_out_now_all_8c = [[9], [0], [1], [2], [6], [7], [8]]
        self.inds_neighbors_out_now_all_8c = [[], [], [], [], [], [], []]

        self.setUpFeatures("out_now_old_8c", "out_now_new_8c", "out_now_all_8c")

        # Categorize output features
        f = self.features_out_now_old_8c
        self.features_out_cat_now_old_8c = {
                "clutter0" : [f[0], f[1]],
                "clutter1" : [f[2], f[3]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[4], f[5], f[6]],
            }
        f = self.features_out_now_new_8c
        self.features_out_cat_now_new_8c = {
                "clutter1" : [f[0]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[1], f[2], f[3], f[4], f[5], f[6]],
            }
        f = self.features_out_now_all_8c
        self.features_out_cat_now_all_8c = {
                "clutter0" : [],
                "clutter1" : [f[0]],
                "small"    : [],
                "medium"   : [],
                "large"    : [f[1], f[2], f[3], f[4], f[5], f[6]],
            }

        # Keyword arguments for front_surgery_temporal
        # Regarding merge_nonclutter: ...
        self.kwargs = {
                "nx"               : self.nx,
                "ny"               : self.ny,
                "min_overlap"      : 0.8,
                "max_size"         : -1,
                "connectivity"     : 8,
                "merge_nonclutter" : True,
            }

    def test_both_8c(self):
        features_cat = front_surgery_temporal(
                self.features_in_cat_new_8c,
                self.features_in_cat_now_8c,
                self.features_in_cat_old_8c,
                **self.kwargs
            )
        features_sol = self.features_out_cat_now_all_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )

    def test_new_8c(self):
        features_cat = front_surgery_temporal(
                self.features_in_cat_new_8c,
                self.features_in_cat_now_8c,
                None,
                **self.kwargs
            )
        features_sol = self.features_out_cat_now_new_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )

    def test_old_8c(self):
        features_cat = front_surgery_temporal(
                None,
                self.features_in_cat_now_8c,
                self.features_in_cat_old_8c,
                **self.kwargs
            )
        features_sol = self.features_out_cat_now_old_8c
        for category, sol in sorted(features_sol.items()):
            self.assertFeaturesEqual(features_cat[category], sol,
                    check_boundaries=False,
                    check_neighbors=False,
                    check_shared_pixels=False,
                )


if __name__ == "__main__":
    import logging as log

    log.getLogger().addHandler(log.StreamHandler(sys.stdout))
    log.getLogger().setLevel(log.DEBUG)
    unittest.main()

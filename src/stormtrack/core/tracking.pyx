#!/usr/bin/env python3

from __future__ import print_function

cimport cython
cimport numpy as np
from cpython.object cimport Py_EQ
from cpython.object cimport Py_GE
from cpython.object cimport Py_GT
from cpython.object cimport Py_LE
from cpython.object cimport Py_LT
from cpython.object cimport Py_NE
from libc.math cimport pow
from libc.math cimport sqrt
from libc.stdlib cimport exit
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libcpp cimport bool

#------------------------------------------------------------------------------

import itertools
import logging as log
import os
import random
from collections import OrderedDict as odict
from copy import copy
from copy import deepcopy
from datetime import datetime
from pprint import pformat
from pprint import pprint

import cython
import igraph as ig
import numpy as np
import shapely.geometry as geo

from .typedefs import Constants
from .identification import merge_adjacent_features
from .identification import features_grow

try:
    from ..utils.various import ipython
except ImportError:
    pass

#==============================================================================

#TS_FMT_DEFAULT = "%Y%m%d%H"
TS_FMT_DEFAULT = None

#==============================================================================

cdef class FeatureTracker:

    def __cinit__(self, *,
            float       f_overlap,
            float       f_size,
            float       min_p_tot,
            float       min_p_overlap,
            float       min_p_size,
            int         minsize                     = 1,
            int         maxsize                     = -1,
            int         max_children                = 6,
            int         split_tracks_n              = -1,
            int         grow_features_n             = 0,
            bint        reshrink_tracked_features   = False,
            bint        merge_features              = False,
            int         connectivity                = 8,
            bint        size_lonlat                 = False,
            str         area_lonlat_method          = "grid",
            int         nx                          = -1,
            int         ny                          = -1,
            np.ndarray  lon2d                       = None,
            np.ndarray  lat2d                       = None,
            np.ndarray  lon1d                       = None,
            np.ndarray  lat1d                       = None,
            str         ts_fmt                      = None,
            bint        debug                       = False,
            dict        _cache                      = None,
        ):
        if max_children < 1:
            raise ValueError("max_children < 1")

        if grow_features_n < 0:
            raise ValueError("grow_features_n < 0")
        elif grow_features_n > 0:
            reshrink_tracked_features = True

        self.f_overlap                  = f_overlap
        self.f_size                     = f_size
        self.min_p_tot                  = min_p_tot
        self.min_p_overlap              = min_p_overlap
        self.min_p_size                 = min_p_size
        self.minsize                    = minsize
        self.maxsize                    = maxsize
        self.size_lonlat                = size_lonlat
        self.area_lonlat_method         = area_lonlat_method
        self.max_children               = max_children
        self.split_tracks_n             = split_tracks_n
        self.grow_features_n            = grow_features_n
        self.reshrink_tracked_features  = reshrink_tracked_features
        self.merge_features             = merge_features
        self.ts_fmt                     = ts_fmt

        # Check/set grid size and lon/lat arrays
        if lon2d is not None and lat2d is not None:
            lon2d_shape = np.asarray(lon2d).shape
            lat2d_shape = np.asarray(lat2d).shape

            if lon2d_shape != lat2d_shape:
                err = ("lon2d and lat2d differ in shape: {} != {}"
                        ).format(lon2d_shape, lat2d_shape)
                raise ValueError(err)

            if len(lon2d_shape) != 2:
                err = "lon2d/lat2d not 2D: {}".format(lon2d_shape)
                raise ValueError(err)

            if nx > 0 and ny > 0:
                if (nx, ny) != lon2d_shape:
                    err = ("inconsistent grid sizes (nx, ny) v. lon2d/lat2d: "
                            "({}, {}) != {}").format(nx, ny, lon2d_shape)
                    raise ValueError(err)
            else:
                nx, ny = lon2d_shape

        elif lon1d is not None and lat1d is not None:
            lon1d_shape = np.asarray(lon1d).shape
            lat1d_shape = np.asarray(lat1d).shape
            lon1d_size = np.asarray(lon1d).size
            lat1d_size = np.asarray(lat1d).size

            if len(lon1d_shape) != 1 or len(lat1d_shape) != 1:
                err = ("lon1d/lat1d not 1D: {}, {}"
                        ).format(lon1d_shape, lat1d_shape)
                raise ValueError(err)

            if nx > 0:
                if nx != lon1d_size:
                    err = ("inconsistent grid sizes nx v. lon1d: {} != {}"
                            ).format(nx, lon1d_size)
                    raise ValueError(err)
            else:
                nx = lon1d_size

            if ny > 0:
                if ny != lat1d_size:
                    err = ("inconsistent grid sizes ny v. lat1d: {} != {}"
                            ).format(ny, lat1d_size)
                    raise ValueError(err)
            else:
                nx = lat1d_size

        elif nx <= 0 or ny <= 0:
            err = "must pass one of (nx, ny), (lon2d, lat2d), (lon1d, lat1d)!"
            raise ValueError(err)

        self.nx                         = nx
        self.ny                         = ny
        self.lon2d                      = lon2d
        self.lat2d                      = lat2d
        self.lon1d                      = lon1d
        self.lat1d                      = lat1d

        self.debug                      = debug

        if debug:
            log.debug("initialize FeatureTracker with debug output")

        self.used_ids = set()
        self._ts_nan = np.iinfo(np.uint64).max

        self.previous_timestep = self._ts_nan
        self.current_timestep = self._ts_nan

        self.active_tracks = []
        self.finished_tracks = []

        self._feature_dict_new = {}

        # Initialize constants
        self.constants = Constants(
                nx = nx,
                ny = ny,
                connectivity = connectivity,
                n_neighbors_max = 8,
            )

        # Initialize grids
        self._grid_new = Grid(self.constants, alloc_tables=True, n_slots=1)
        self._grid_now = Grid(self.constants, alloc_tables=True, n_slots=1)
        if grow_features_n == 0:
            self._grid_grow = None
        else:
            #SR_TODO figure out appropriate n_slots (no. neighbors?)
            #self._grid_grow = Grid(self.constants, alloc_tables=True, n_slots=1)
            pass #SR_TODO solve memory issue

    def __dealloc__(self):
        pass

    #def __repr__(self):
    #    return "{}".format(self.__class__.__name__)

    #--------------------------------------------------------------------------

    def check_setup(self, ref, action="raise"):
        if action not in ["raise", "warn"]:
            raise ValueError("invalid action: {}".format(action))

        setup = {
                "f_overlap"      : self.f_overlap,
                "f_size"         : self.f_size,
                "min_p_tot"      : self.min_p_tot,
                "min_p_overlap"  : self.min_p_overlap,
                "min_p_size"     : self.min_p_size,
                "minsize"        : self.minsize,
                "maxsize"        : self.maxsize,
                "max_children"   : self.max_children,
                "nx"             : self.nx,
                "ny"             : self.ny,
                "size_lonlat"    : self.size_lonlat,
                "split_tracks_n" : self.split_tracks_n,
                "merge_features" : self.merge_features,
                "ts_fmt"         : self.ts_fmt,
            }

        unchecked = sorted(setup.keys())
        leftovers = []
        for key, val in sorted(ref.items()):
            try:
                ok = (val == setup[key])
            except KeyError:
                leftovers.append(key)
            else:
                if not ok:
                    err = "inconsistent option {}: {} != {}".format(
                            key, setup[key], val)
                    if action == "raise":
                        raise Exception(err)
                    elif action == "warn":
                        log.warning(err)
                unchecked.remove(key)

        if unchecked:
            log.info("{} unchecked setup parameters: {}".format(len(unchecked),
                    ", ".join(unchecked)))

        if leftovers:
            log.info("{} left-over reference setup parameters: {}".format(
                    len(leftovers), ", ".join(leftovers)))

    #--------------------------------------------------------------------------

    cpdef void reset(self):

        self.previous_timestep = self._ts_nan
        self.current_timestep = self._ts_nan

        self.active_tracks = []
        self.finished_tracks = []

        self.used_ids = set()

        self._grid_new.reset()
        self._grid_now.reset()

    #cpdef bint id_in_use(self, np.uint64_t id_) except *:
    def id_in_use(self, id_):
        return (id_ in self.used_ids)

    #cpdef void register_id(self, np.uint64_t id_) except *:
    def register_id(self, id_):
        if self.id_in_use(id):
            err = "track id already in use: {}".format(id_)
            raise ValueError(id_)
        self.used_ids.add(id_)

    def min_ts_start(self):
        """Return timestep of earliest active track start."""
        if len(self.active_tracks) == 0:
            return None
        return min([t.ts_start() for t in self.active_tracks])

    #--------------------------------------------------------------------------

    def restart(self, tracks, timestep):
        """Restart tracker from track file.

        All unfinished tracks are inserted as active tracks.
        """

        self.previous_timestep = self._ts_nan
        self.current_timestep = timestep

        # Collect active and finished tracks, along with used track ids
        self.used_ids.clear()
        self.active_tracks.clear()
        self.finished_tracks.clear()
        for track in tracks:
            self.used_ids.add(track.id)
            if not track.is_complete() and timestep in track.timesteps():
                self.active_tracks.append(track)
                vs_end = track.graph.vs.select(ts=timestep)
                if "stop" in vs_end["type"]:
                    raise NotImplementedError("restart from complete period "
                            "(stop event found in types")
            else:
                self.finished_tracks.append(track)

    #==========================================================================

    cpdef void extend_tracks(self, list features, np.uint64_t timestep,
        ) except *:
        """Extend all active track branches with a new set of features.

        Branches for which no successor can be found (or, to be precise and
        consider tracks with gaps, for which still no successor could be
        found after a certain number of timsteps) are terminated.

        Tracks with no active branches left are terminated.
        """
        cdef bint debug = self.debug
        if debug: log.debug("{}.extend_tracks: timestep {}".format(self.__class__.__name__, timestep))
        cdef int i, j, n
        self.previous_timestep = self.current_timestep
        self.current_timestep = timestep

        # Reset NEW grid
        self._grid_new.reset()

        # Select features in size range
        cdef int n_features_old=len(features)
        cdef Feature feature, neighbor
        cdef np.float32_t area
        cdef list features_old
        if self.maxsize > 0:
            if self.size_lonlat:
                features_old = features
                features = []
                for feature in features_old:
                    if self.area_lonlat_method == "grid":
                        area = feature.area_lonlat(self.lon1d, self.lat1d,
                                method=self.area_lonlat_method)
                    else:
                        area = feature.area_lonlat(self.lon2d, self.lat2d,
                                method=self.area_lonlat_method)
                    if self.minsize <= area <= self.maxsize:
                        features.append(feature)
            else:
                features_old = features
                features = []
                for feature in features_old:
                    if self.minsize <= feature.n <= self.maxsize:
                        features.append(feature)
            if debug: log.debug("select {}/{} features in size range ({}..{})".format(len(features), n_features_old, self.minsize, self.maxsize))
        elif self.minsize > 0:
            if self.size_lonlat:
                features_old = features
                features = []
                for feature in features_old:
                    if self.area_lonlat_method == "grid":
                        area = feature.area_lonlat(self.lon1d, self.lat1d,
                                method=self.area_lonlat_method)
                    else:
                        area = feature.area_lonlat(self.lon2d, self.lat2d,
                                method=self.area_lonlat_method)
                    if self.minsize <= area:
                        features.append(feature)
            else:
                features = [feature for feature in features
                        if self.minsize <= feature.n]
            if debug: log.debug("select {}/{} features in size range ({}..)".format(len(features), n_features_old, self.minsize))

        if self.grow_features_n > 0:
            # Grow features
            #self._grid_grow.reset()
            features_grow(self.grow_features_n, features, self.constants,
                    inplace=True, retain_orig=True)#, grid=self._grid_grow)

        # Remove some neighbors TODO appropriate comment
        cdef int n_features=len(features)
        if n_features < n_features_old:
            for feature in features:
                for neighbor in feature.neighbors.copy():
                    if neighbor not in features:
                        if debug: log.debug("feature {}: remove neighbor {}".format(feature.id, neighbor.id))
                        feature.neighbors.remove(neighbor)

        # Convert features to cfeatures
        cdef cRegions cfeatures = cregions_create(n_features)
        cdef bint ignore_missing_neighbors = False
        dbg_check_features_cregion_pixels(features)
        features_to_cregions(
                features,
                n_features,
                &cfeatures,
                cregion_conf_default(),
                ignore_missing_neighbors,
                self._grid_new.to_c(),
                self.constants.to_c(),
            )
        dbg_check_features_cregion_pixels(features)
        self._feature_dict_new = {}
        for i in range(n_features):
            self._feature_dict_new[cfeatures.regions[i].id] = features[i]

        # Initialize pixel region table
        pixel_region_table_init_regions(
                self._grid_new.to_c().pixel_region_table,
                &cfeatures,
                &cfeatures,
                n_slots_max=1,
            )

        #=============================================
        if debug: log.debug("\n" + "<"*40 + "\n")
        #---------------------------------------------
        self._extend_tracks_core(features, &cfeatures)
        #---------------------------------------------
        if debug: log.debug("\n" + ">"*40 + "\n")
        #=============================================

        # Swap grids
        self._swap_grids()

    cdef void _swap_grids(self) except *:

        # Unlink cregions from features
        cdef FeatureTrack track
        cdef Feature feature
        for track in self.active_tracks:
            for feature in track.features_ts(self.previous_timestep):
                feature.reset_cregion(warn=False)

        # Swap grids
        cdef Grid grid_tmp
        grid_tmp = self._grid_now
        self._grid_now = self._grid_new
        self._grid_new = grid_tmp

    #SR_TODO move to separate class TrackExtender
    cdef void _extend_tracks_core(self, list features, cRegions* cfeatures
            ) except *:
        cdef bint debug = self.debug
        cdef int i, j, k, n
        cdef Feature feature
        cdef cRegion cfeature
        cdef FeatureTrack track

        #SR_DBG<
        # Check that features and their cregions are consistent pixel-wise
        # This will be obsolete once Feature uses cRegion's pixels directly
        dbg_check_features_cregion_pixels(features)
        for track in self.active_tracks:
            dbg_check_features_cregion_pixels(track.features())
        #SR_DBG>

        #DBG_BLOCK<
        if debug:
            log.debug("\nSTATUS: finished/active/features {:3} {:3} {:3}\n".format(len(self.finished_tracks), len(self.active_tracks), len(features)))
            log.debug("{} heads:".format(len(self.head_features())))
            for vertex in self.head_vertices():
                log.debug(" - [{}/{}]({}/{}) {}".format(vertex["feature"].id, (<Feature>vertex["feature"]).cregion.id, vertex["feature"].n, (<Feature>vertex["feature"]).cregion.pixels_n, vertex["type"]))
            log.debug("")
            log.debug("{} features:".format(len(features)))
            for i in range(cfeatures.n):
                log.debug(" - [{}/{}]({}/{})".format(features[i].id, cfeatures.regions[i].id, features[i].n, cfeatures.regions[i].pixels_n))
            log.debug("")
        #DBG_BLOCK>

        # Prepare array for successor candidates and probabilities
        cdef SuccessorCandidates combinations
        combinations.n = 0
        combinations.max = 20
        combinations.candidates = <SuccessorCandidate*>malloc(
                combinations.max*sizeof(SuccessorCandidate))
        for i in range(combinations.max):
            combinations.candidates[i].n = 0
            combinations.candidates[i].max = self.max_children
            combinations.candidates[i].p_tot = 0.0
            combinations.candidates[i].p_size = 0.0
            combinations.candidates[i].p_overlap = 0.0
            combinations.candidates[i].children = <cRegion**>malloc(
                    self.max_children*sizeof(cRegion*))
            combinations.candidates[i].p_shares = <np.float32_t*>malloc(
                    self.max_children*sizeof(np.float32_t))
            combinations.candidates[i].n_overlaps = <np.int32_t*>malloc(
                    self.max_children*sizeof(np.int32_t))
            for j in range(self.max_children):
                combinations.candidates[i].children[j] = NULL
                combinations.candidates[i].p_shares[j] = 0.0
                combinations.candidates[i].n_overlaps[j] = 0

        # Find all successor candidate combinations in both directions
        cdef int n_heads = len(self.head_features())
        cdef cGrid* dummy_grid = NULL
        cdef cRegions heads = cregions_create(n_heads)
        cdef bint cleanup=False, unlink=False
        cdef int forward=1, backward=-1
        for feature in self.head_features():
            cregions_link_region(&heads, feature.cregion, cleanup, unlink)
        self._find_successor_candidate_combinations(&heads, cfeatures,
                &combinations, direction=forward)
        #SR_TODO restrict backward search to combinations w/ multiple children
        #SR_TODO (one-to-one combinations all already found forward)
        self._find_successor_candidate_combinations(cfeatures, &heads,
                &combinations, direction=backward)
        if debug: log.debug("found {} successor candidate combinations".format(combinations.n))

        # Compute successor probabilities for all combinations
        self._compute_successor_probabilities(&combinations)

        # Sort combinations
        self._sort_successor_candidates(&combinations)

        #DBG_BLOCK<
        if debug:
            log.debug("\n{} successor candidate combinations (tot/size/overlap):".format(combinations.n))
            for i in range(combinations.n):
                child_ids = []
                for j in range(combinations.candidates[i].n):
                    child_ids.append(str(
                            combinations.candidates[i].children[j].id))
                log.debug(" {:4.2f} {:4.2f} {:4.2f} [{}]{}[{}] ({})".format(combinations.candidates[i].p_tot, combinations.candidates[i].p_size, combinations.candidates[i].p_overlap, combinations.candidates[i].parent.id, "->" if combinations.candidates[i].direction > 0 else "<-", "/".join(child_ids), combinations.candidates[i].direction))
            log.debug("")
        #DBG_BLOCK>

        # Assign successors
        cdef list features_todo = self._assign_successors(features, cfeatures,
                &combinations)

        # Start tracks (left-over features)
        n = 0
        for feature in features_todo:
            n += 1
            new_track = self._start_track(feature.cregion)
            if debug: log.debug("started new track {} with feature {}".format(new_track.id, feature.cregion.id))
        if debug: log.debug("started {} new tracks".format(n))

        # Finish branches
        n = 0
        for vertex in self.head_vertices():
            if vertex["ts"] < self.current_timestep:
                n += 1
                if debug: log.debug("finish branch of track {} at feature [{}/{}]".format(vertex["feature"].track().id, vertex["feature"].id, (<Feature>vertex["feature"]).cregion.id))
                self._finish_branch(vertex)
        if debug: log.debug("finished {} branches".format(n))

        #SR_TMP<
        # Count tracks to be finished to get an idea whether parallelization
        # might be worth-while and above which threshold
        n = sum([1 for t in self.active_tracks if not any(t.graph.vs["active_head"])])
        if debug: log.debug("[{}] {} tracks about to be finished ++++".format(self.current_timestep, n))
        #SR_TMP>

        # Finish tracks (no heads left)
        n = 0
        for track in self.active_tracks:
            if track.is_finished():
                n += 1
                self._finish_track(track, force=False)
        if n > 0:
            if debug: log.debug("finished {}/{} active tracks".format(n, len(self.active_tracks) + n))
        if debug: log.debug("finished {} tracks".format(n))

        #DBG_BLOCK<
        if debug:
            log.debug("\n{} heads:".format(len(self.head_features())))
            for feature in self.head_features():
                log.debug(" - track {} feature {}/{} ({}/{})".format(feature.track().id, feature.id, feature.cregion.id, feature.n, feature.cregion.pixels_n))
        #DBG_BLOCK>

        # Clean up arrays
        for i in range(combinations.max):
            free(combinations.candidates[i].children)
            free(combinations.candidates[i].p_shares)
            free(combinations.candidates[i].n_overlaps)
        free(combinations.candidates)

    #--------------------------------------------------------------------------

    cdef void _find_successor_candidates(self, cRegion* parent,
            cRegions* cfeatures, cRegions *candidates, int direction):
        cdef bint debug = self.debug
        cdef int i
        cdef cRegion* cfeature
        cdef bint cleanup=False, unlink=False
        cdef PixelRegionTable table_parent
        cdef PixelRegionTable table_child
        if direction > 0:
            table_parent = self._grid_now.to_c().pixel_region_table
            table_child = self._grid_new.to_c().pixel_region_table
        else:
            table_parent = self._grid_new.to_c().pixel_region_table
            table_child = self._grid_now.to_c().pixel_region_table
        candidates.n = 0
        if debug: log.debug("find successor candidates for [/{}] (up to {})".format(parent.id, cfeatures.n))
        for i in range(cfeatures.n):
            cfeature = cfeatures.regions[i]
            #SR_TMP<
            if debug: log.debug("[/{}](/{}) <-> [/{}](/{})".format(parent.id, parent.pixels_n, cfeature.id, cfeature.pixels_n))
            if cregion_overlaps_tables(parent, cfeature,
                    table_parent, table_child):
            #if cregion_overlaps(parent, cfeature):
            #SR_TMP>
                if debug: log.debug(" - [/{}](/{}) overlaps [/{}](/{})".format(parent.id, parent.pixels_n, cfeature.id, cfeature.pixels_n))
                cregions_link_region(candidates, cfeature, cleanup, unlink)
            else:
                if debug: log.debug(" - [/{}](/{}) doesn't overlap [/{}](/{})".format(parent.id, parent.pixels_n, cfeature.id, cfeature.pixels_n))

    #--------------------------------------------------------------------------

    cdef void _find_successor_candidate_combinations(self, cRegions* parents,
            cRegions* cfeatures, SuccessorCandidates *combinations,
            int direction) except *:
        cdef bint debug = self.debug
        cdef str _name_ = "{}._find_successor_candidate_combinations".format(self.__class__.__name__)
        if debug: log.debug("< {}: {} parents, {} candidates, {} existing combinations, direction {}".format(_name_, parents.n, cfeatures.n, combinations.n, direction))
        cdef int i, j
        cdef cRegion* parent
        cdef cRegion* child
        cdef cGrid* dummy_grid = NULL
        cdef cRegions candidates = cregions_create(20)
        cdef int n_new_combinations
        for i in range(parents.n):
            parent = parents.regions[i]
            if debug: log.debug("parent [{}]({})".format(parent.id, parent.pixels_n))

            # Find all successor candidates (non-zero overlap)
            self._find_successor_candidates(parent, cfeatures, &candidates,
                    direction)
            if debug: log.debug(" -> found {} candidates".format(candidates.n))

            # Find all combinations of candidates
            n_new_combinations = self._combine_candidates(
                    parent, &candidates, combinations, direction)

            #DBG_BLOCK<
            if debug:
                if debug: log.debug(" -> found {} new candidate combinations:".format(n_new_combinations))
                for i in range(combinations.n - n_new_combinations, combinations.n):
                    indices = []
                    for j in range(combinations.candidates[i].n):
                        child = combinations.candidates[i].children[j]
                        if child is not NULL:
                            indices.append(child.id)
                    if debug: log.debug("     - {}".format(", ".join([str(x) for x in indices])))
            #DBG_BLOCK>

    #--------------------------------------------------------------------------

    cdef int _combine_candidates(self, cRegion* parent,
            cRegions* candidates, SuccessorCandidates *combinations,
            int direction) except -1:
        cdef bint debug = self.debug
        cdef str _name_
        if debug: _name_ = "{}._combine_candidates".format(self.__class__.__name__)
        if debug: log.debug("< {}: {} candidates, direction {}".format(_name_, candidates.n, direction))
        cdef int i, j, k, n

        # Find all combinations of indices
        #SR_TMP< Python implementation (might be slow)
        cdef list indices = []
        for i in range(candidates.n):
            indices.append(candidates.regions[i].id)
        cdef int nmin = 1
        if direction < 0:
            nmin = 2
        indices = all_combinations(indices, nmin, self.max_children)
        cdef int n_combinations = len(indices)
        #if debug: log.debug("found {} combinations: {}".format(n_combinations, ", ".join(["/".join([str(i) for i in ii]) for ii in indices])))
        if debug: log.debug("found {} combinations: {}".format(n_combinations, pformat(indices)))
        #SR_TMP>

        # Collect cfeatures of all combinations
        cdef cRegion* candidate
        cdef np.uint64_t candidate_id
        cdef int n_new = 0
        for i in range(n_combinations):
            if combinations.n == combinations.max:
                successor_combinations_extend(combinations, self.max_children)
            n = len(indices[i])
            for j in range(n):
                candidate_id = indices[i][j]
                for k in range(candidates.n):
                    candidate = candidates.regions[k]
                    if candidate.id == candidate_id:
                        break
                else:
                    log.error("{}: candidate not found with id {}".format("_combine_candidates", candidate_id))
                combinations.candidates[combinations.n].children[j] = candidate
            combinations.candidates[combinations.n].n = n
            combinations.candidates[combinations.n].parent = parent
            combinations.candidates[combinations.n].direction = direction
            combinations.n += 1
            n_new += 1
        return n_new

    #--------------------------------------------------------------------------

    cdef void _compute_successor_probabilities(self,
            SuccessorCandidates* combinations) except *:
        """Compute overlap, area, and total successor probabilities,"""
        cdef bint debug = self.debug
        if debug: log.debug("compute successor probabilities of {} combinations".format(combinations.n))

        # Compute probabilities
        cdef int i, j
        cdef cRegion* child
        cdef int tot_size, tot_overlap, n_overlap
        cdef float p_size, p_overlap, p_tot
        cdef cRegion* parent
        cdef PixelRegionTable pixel_region_table_parent = NULL
        cdef PixelRegionTable pixel_region_table_child = NULL
        for i in range(combinations.n):
            tot_overlap = 0
            tot_size = 0
            if debug: child_ids = []
            parent = combinations.candidates[i].parent
            if combinations.candidates[i].direction > 0:
                pixel_region_table_parent = self._grid_now.to_c().pixel_region_table
                pixel_region_table_child = self._grid_new.to_c().pixel_region_table
            else:
                pixel_region_table_parent = self._grid_new.to_c().pixel_region_table
                pixel_region_table_child = self._grid_now.to_c().pixel_region_table
            for j in range(combinations.candidates[i].n):
                child = combinations.candidates[i].children[j]
                if debug: child_ids.append(child.id)
                tot_size += child.pixels_n
                n_overlap = cregion_overlap_n_tables(parent, child,
                        pixel_region_table_parent, pixel_region_table_child)
                combinations.candidates[i].n_overlaps[j] = n_overlap
                tot_overlap += n_overlap

                # Branchings: compute inididual successor probabilities
                compute_tracking_probabilities(&p_tot, &p_size, &p_overlap,
                        parent.pixels_n, child.pixels_n, n_overlap,
                        self.f_size, self.f_overlap,
                        min_p_size=0, min_p_overlap=0,
                    )
                combinations.candidates[i].p_shares[j] = p_tot

            # Compute probability shares (only relevant for branchings)
            # This is the individual successor probability of each child,
            # normalized by the sum over all children, and is a measure
            # for the successor quality of each child ("who's the favorite?")
            if combinations.candidates[i].n == 1:
                combinations.candidates[i].p_shares[j] = 1.0
            else:
                p_tot = 0
                for j in range(combinations.candidates[i].n):
                    p_tot += combinations.candidates[i].p_shares[j]
                if p_tot > 0:
                    for j in range(combinations.candidates[i].n):
                        with cython.cdivision(True):
                            combinations.candidates[i].p_shares[j] /= p_tot

            # Compute successor probabilities
            compute_tracking_probabilities(&p_tot, &p_size, &p_overlap,
                    parent.pixels_n, tot_size, tot_overlap,
                    self.f_size, self.f_overlap,
                    self.min_p_size, self.min_p_overlap)

            if debug: log.debug(" - [{}]{}[{}] tot/size/overlap {:4.2f} {:4.2f} {:4.2f}".format(parent.id, "->" if combinations.candidates[i].direction > 0 else "<-", "/".join([str(x) for x in child_ids]), p_tot, p_size, p_overlap))

            combinations.candidates[i].p_tot = p_tot
            combinations.candidates[i].p_size = p_size
            combinations.candidates[i].p_overlap = p_overlap

    cdef list _assign_successors(self, list features, cRegions* cfeatures,
            SuccessorCandidates* combinations):
        cdef str _name_ = "{}._assign_successors".format(
                self.__class__.__name__)
        cdef bint debug = self.debug
        cdef int i, j, k
        cdef list features_todo = features.copy()
        cdef int i_feature
        cdef FeatureTrack track
        cdef int n_children
        cdef Feature parent_feature, child_feature
        cdef list child_features, child_features_p_shares
        cdef list vertices_now, vertices_new
        cdef list tracks
        cdef Feature head #SR_TMP
        cdef list already_assigned = []
        cdef bint skip
        cdef str type
        cdef int direction
        cdef dict attrs

        for i in range(combinations.n):
            if combinations.candidates[i].p_tot < self.min_p_tot:
                continue
            direction = combinations.candidates[i].direction

            #DBG_BLOCK<
            if debug:
                child_ids = []
                for j in range(combinations.candidates[i].n):
                    child_ids.append(str(combinations.candidates[i].children[j].id))
                log.debug("\nCOMBINATION: {} {} {}".format(combinations.candidates[i].parent.id, "->" if combinations.candidates[i].direction > 0 else "<-", "/".join(child_ids)))
            #DBG_BLOCK<

            #SR_TMP< TODO implement with less Python overhead
            # Collect child features
            skip = False
            child_features = []
            child_features_p_shares = []
            child_features_n_overlaps = []
            n_children = combinations.candidates[i].n
            if direction > 0:
                for j in range(n_children):
                    child = combinations.candidates[i].children[j]
                    child_features_p_shares.append(
                            combinations.candidates[i].p_shares[j])
                    child_features_n_overlaps.append(
                            combinations.candidates[i].n_overlaps[j])
                    if child.id in already_assigned:
                        if debug: log.debug("[{}] already assigned -> skip combination".format(child.id))
                        skip = True
                        break
                    for k in range(cfeatures.n):
                        if child.id == cfeatures.regions[k].id:
                            child_features.append(features[k])
                            break
                    else:
                        log.error("cregion [{}] not found".format(child.id))
                        exit(4)
            else:
                for j in range(n_children):
                    child = combinations.candidates[i].children[j]
                    child_features_p_shares.append(
                            combinations.candidates[i].p_shares[j])
                    child_features_n_overlaps.append(
                            combinations.candidates[i].n_overlaps[j])
                    if child.id in already_assigned:
                        if debug: log.debug("[{}] already assigned -> skip combination".format(child.id))
                        skip = True
                        break
                    for child_feature in self.head_features():
                        if child.id == child_feature.cregion.id:
                            child_features.append(child_feature)
                            break
                    else:
                        log.error("cregion [{}] not found".format(child.id))
                        exit(4)
            if skip:
                continue
            #SR_TMP>

            # Get parent feature
            parent = combinations.candidates[i].parent
            if parent.id in already_assigned: #SR_TMP
                if debug: log.debug("[{}] already assigned -> skip combination".format(parent_feature.id))
                continue
            if direction > 0:
                for head in self.head_features():
                    if head.cregion.id == parent.id:
                        parent_feature = head
                        break
                else:
                    log.error("feature corresponding to [{}]({}) not found among heads".format(parent.id, parent.pixels_n))
                    exit(4)
                track = parent_feature.track()
                if debug: log.debug("select track [{}]({}) of parent feature {}".format(track.id, track.n, parent_feature.id))
            else:
                for j in range(cfeatures.n):
                    if cfeatures.regions[j].id == parent.id:
                        parent_feature = features[j]
                        break
                else:
                    log.error("feature corresponding to [{}]({}) not found".format(parent.id, parent.pixels_n))
                    exit(4)

            # In case of a merging, merge the incoming tracks
            if direction < 0:
                tracks = []
                for child_feature in child_features:
                    if child_feature.track() not in tracks:
                        tracks.append(child_feature.track())
                track = self._merge_tracks(tracks)
                if debug: log.debug("merged tracks {} into {}".format(", ".join([str(t.id) for t in tracks]), track.id))

            # Get/add parent vertex
            if direction > 0:
                vertex_parent = parent_feature.vertex()
            else:
                parent_feature.timestep = self.current_timestep
                vertex_parent = track_graph_add_feature(track.graph, parent_feature)
                parent_feature.set_track(track)

            # Adapt parent type in case of splitting
            if direction > 0 and n_children > 1:
                type = vertex_parent["type"]
                if not "splitting" in type:
                    if type in ("start", "genesis", "merging"):
                        vertex_parent["type"] = "{}/{}".format(
                                type, "splitting")
                    else:
                        vertex_parent["type"] = "splitting"

            # Add and link children
            for j in range(n_children):
                child_feature = child_features[j]
                child = child_feature.cregion
                #DBG_BLOCK<
                if debug:
                    if direction > 0:
                        log.debug("continue track {} after [{}]({}/{}) with [{}]({}/{}) (->)".format(track.id, parent_feature.id, parent_feature.n, parent_feature.cregion.pixels_n, child_feature.id, child_feature.n, child_feature.cregion.pixels_n))
                    else:
                        log.debug("continue track {} after [{}]({}/{}) with [{}]({}/{}) (<-)".format(track.id, child_feature.id, child_feature.n, child_feature.cregion.pixels_n, parent_feature.id, parent_feature.n, parent_feature.cregion.pixels_n))
                #DBG_BLOCK>

                #SR_TMP<
                already_assigned.append(parent.id)
                already_assigned.append(child.id)
                #SR_TMP>

                # Add/get child vertex
                if direction > 0:
                    child_feature.timestep = self.current_timestep
                    vertex_child = track_graph_add_feature(track.graph, child_feature)
                    child_feature.set_track(track)
                else:
                    vertex_child = child_feature.vertex()

                # Update "active_head" property
                if direction > 0:
                    vertex_parent["active_head"] = False
                    vertex_child["active_head"] = True
                else:
                    vertex_child["active_head"] = False
                    vertex_parent["active_head"] = True

                # Set "type" property
                if direction > 0:
                    vertex_child["type"] = "continuation"
                else:
                    if n_children == 1:
                        vertex_parent["type"] = "continuation"
                    else:
                        vertex_parent["type"] = "merging"

                # Add edge between parent and child
                attrs = dict(
                        p_tot     = combinations.candidates[i].p_tot,
                        p_size    = combinations.candidates[i].p_size,
                        p_overlap = combinations.candidates[i].p_overlap,
                        p_share   = child_features_p_shares[j],
                        n_overlap = child_features_n_overlaps[j],
                    )
                #SR_TMP< SR_DIRECTED
                if vertex_parent["ts"] > vertex_child["ts"]:
                    edge = track_graph_add_edge(track.graph,
                            vertex_child, vertex_parent, attrs)
                else:
                #SR_TMP>
                    edge = track_graph_add_edge(track.graph,
                            vertex_parent, vertex_child, attrs)

                #SR_TMP<
                if direction > 0:
                    features_todo.remove(child_feature)
            if direction < 0:
                features_todo.remove(parent_feature)
            #SR_TMP>

        return features_todo

    #==========================================================================

    cpdef int split_active_tracks(self) except *:
        cdef bint debug = self.debug
        if debug: log.debug("try to split {} active tracks".format(len(self.active_tracks)))
        if self.split_tracks_n < 0:
            return 0
        nnew = 0
        for track in self.active_tracks.copy():
            self.active_tracks.remove(track)
            subtracks = track.split(self.split_tracks_n)
            if debug: log.debug(" -> [{}] {}".format(track.id, "not split" if len(subtracks) == 1 else "split into {} subtracks ({} active)".format(len(subtracks), len([t for t in subtracks if t.is_active()]))))
            del track
            for subtrack in subtracks:
                if subtrack.is_active():
                    self.active_tracks.append(subtrack)
                else:
                    del subtrack.graph.vs["active_head"]
                    subtrack.finish(force=False, split_n=-1,
                            merge_features=self.merge_features)
                    self.finished_tracks.append(subtrack)
                    nnew += 1
        return nnew

    #--------------------------------------------------------------------------

    cdef object _merge_tracks(self, list tracks):
        """Merge tracks and return the merged track."""
        cdef bint debug = self.debug
        if debug: log.debug("{}._merge_tracks: {}".format(self.__class__.__name__, ", ".join([str(t.id) for t in tracks])))
        return merge_tracks(tracks, self.active_tracks)

    #--------------------------------------------------------------------------

    cpdef object head_vertices(self):
        return [h for t in self.active_tracks for h in t.head_vertices()]

    cpdef list head_features(self):
        return [h for t in self.active_tracks for h in t.head_features()]

    #--------------------------------------------------------------------------

    cdef FeatureTrack _start_track(self, cRegion* cfeature):
        cdef bint debug = self.debug
        if debug: log.debug("{}._start_track([{}])".format(self.__class__.__name__, cfeature.id))
        cdef Feature feature = self._feature_dict_new[cfeature.id]
        #SR_TMP<
        feature.timestep = self.current_timestep
        #SR_TMP>
        cdef str type
        if self.previous_timestep == self._ts_nan:
            type = "start"
        else:
            type = "genesis"
        #SR_TMP<
        cdef dict config = dict(
                f_overlap       = self.f_overlap,
                f_size          = self.f_size,
                min_p_tot       = self.min_p_tot,
                min_p_overlap   = self.min_p_overlap,
                min_p_size      = self.min_p_size,
                minsize         = self.minsize,
                maxsize         = self.maxsize,
                size_lonlat     = self.size_lonlat,
                max_children    = self.max_children,
                merge_features  = self.merge_features,
                ts_fmt          = self.ts_fmt,
            )
        #SR_TMP>
        fid = self.new_track_id(self.current_timestep)
        cdef FeatureTrack new_track = FeatureTrack(
                id_     = fid,
                feature = feature,
                type    = type,
                config  = config,
                tracker = self,
            )
        new_track.graph.vs["active_head"] = True
        self.active_tracks.append(new_track)
        return new_track

    def new_track_id(self, np.uint64_t ts):
        return new_track_id(ts, self.used_ids)

    cpdef void finish_tracks(self) except *:
        """Force-finish all still active branches/tracks."""
        cdef bint debug = self.debug
        if debug: log.debug("force-finish {} active tracks".format(len(self.active_tracks)))
        cdef FeatureTrack track
        for track in self.active_tracks.copy():
            self._finish_track(track, force=True)

    cdef void _finish_branch(self, object vertex):
        cdef str type = vertex["type"]
        if type in ("start", "genesis", "merging"):
            vertex["type"] = "{}/{}".format(type, "lysis")
        else:
            vertex["type"] = "lysis"
        vertex["active_head"] = False

    cdef void _finish_track(self, FeatureTrack track, bint force) except *:

        #SR_DBG<
        dbg_check_features_cregion_pixels(track.features())
        #SR_DBG>

        self.active_tracks.remove(track)
        cdef list subtracks
        subtracks = track.finish(force=force, split_n=self.split_tracks_n,
                merge_features=self.merge_features)
        del track
        self.finished_tracks.extend(subtracks)

        cdef FeatureTrack subtrack
        cdef Feature feature
        cdef dict data_orig
        if self.reshrink_tracked_features:
            for subtrack in subtracks:
                for feature in subtrack.features():
                    if "feature_orig" in feature.properties:
                        data_orig = feature.properties.pop("feature_orig")
                        feature.set_values(data_orig["values"])
                        feature.set_pixels(data_orig["pixels"])
                        #SR_TMP< Account for old files (only one shell)
                        try:
                            feature.set_shells(data_orig["shells"])
                        except KeyError:
                            feature.set_shells([data_orig["shell"]])
                        #SR_TMP>
                        feature.set_holes(data_orig["holes"])

    cpdef list pop_finished_tracks(self):
        """Return all finished tracks and remove them fron the tracker."""
        cdef list finished_tracks = self.finished_tracks.copy()
        self.finished_tracks = []
        return finished_tracks

    cdef void _sort_successor_candidates(self,
            SuccessorCandidates* combinations) except *:
        """Sort successor candidates by total successor probability."""
        cdef list combinations_list = []
        cdef int i, j
        for i in range(combinations.n):
            combinations_list.append((combinations.candidates[i].p_tot, i))
        combinations_list.sort(reverse=True)
        cdef SuccessorCandidate* tmp = <SuccessorCandidate*>malloc(
                combinations.n*sizeof(SuccessorCandidate))
        for i in range(combinations.n):
            tmp[i] = combinations.candidates[i]
        for i in range(combinations.n):
            j = combinations_list[i][1]
            combinations.candidates[i] = tmp[j]
        free(tmp)

#------------------------------------------------------------------------------

cdef void compute_tracking_probabilities(
        float* p_tot,
        float* p_size,
        float* p_overlap,
        int n_parent,
        int n_child,
        int n_overlap,
        float f_size,
        float f_overlap,
        float min_p_size,
        float min_p_overlap,
    ) except *:

    # Size probability
    p_size[0] = min(<float>n_parent, n_child)/max(<float>n_parent, n_child)

    # Overlap probability
    p_overlap[0] = 2*n_overlap/(<float>n_parent + n_child)

    # Total probability
    p_tot[0] = 0.0
    if p_size[0] >= min_p_size:
        p_tot[0] += f_size*p_size[0]
    if p_overlap[0] >= min_p_overlap:
        p_tot[0] += f_overlap*p_overlap[0]

def merge_tracks(tracks, active_tracks=None):
    cdef bint debug = False
    if debug: log.debug("merge_tracks: {} tracks".format(len(tracks)))

    target_track = tracks.pop(0)
    new_fids = []
    for track in tracks:

        # Add vertices
        for vertex in track.graph.vs:
            feature = vertex["feature"]
            if feature is not None:
                new_fids.append(feature.id)
                if debug: log.debug("move feature {} to track {}".format(feature.id, target_track.id))

            target_track.graph.add_vertex(**vertex.attributes())
            new_vertex = target_track.graph.vs[-1]

            if feature is not None:
                feature.set_track(target_track)
                feature.set_vertex(new_vertex)

        # Add existing edges
        for edge in track.graph.es:

            source_vertex_old = track.graph.vs.find(edge.source)
            target_vertex_old = track.graph.vs.find(edge.target)

            source_feature = source_vertex_old["feature"]
            target_feature = target_vertex_old["feature"]

            if debug: log.debug("add edge between features {} and {} to track {}".format(source_feature.id, target_feature.id, target_track.id))

            source_vertex = target_track.graph.vs.find(feature=source_feature)
            target_vertex = target_track.graph.vs.find(feature=target_feature)

            #SR_TMP< SR_DIRECTED
            if source_vertex["ts"] > target_vertex["ts"]:
                target_track.graph.add_edge(target_vertex.index,
                        source_vertex.index, **edge.attributes())
            else:
            #SR_TMP>
                target_track.graph.add_edge(source_vertex.index,
                        target_vertex.index, **edge.attributes())

            new_edge = target_track.graph.es.find(_between=(
                    (source_vertex.index,), (target_vertex.index,)))

        if active_tracks is not None:
            active_tracks.remove(track)

    if target_track.missing_features_stats is not None:
        # Remove obsolete missing features stats
        inds = [i for i, fid in enumerate(
                target_track.missing_features_stats["id"]) if fid not in new_fids]
        for key, vals in target_track.missing_features_stats.items():
            target_track.missing_features_stats[key] = [vals[i] for i in inds]

    return target_track

cpdef void dbg_check_features_cregion_pixels(list features) except *:
    cdef Feature feature
    for feature in features:
        if feature.cregion is not NULL:
            if feature.n != feature.cregion.pixels_n:
                err = "feature [{}/{}]: inconsistent pixels: ({}/{})".format(
                        feature.id, feature.cregion.id,
                        feature.n, feature.cregion.pixels_n)
                print("warning: "+err)
                #raise Exception(err)
                outfile = "cyclone_tracking_debug_feature_inconsistent_pixels.txt"
                print("debug: write feature info to {}".format(outfile))
                with open(outfile, "a") as fo:
                    fo.write(err+"\n")

#------------------------------------------------------------------------------

def new_track_id(np.uint64_t ts, set used_ids):
    cdef np.uint64_t new_id
    new_id = ts*10000
    while new_id in used_ids:
        new_id += 1
        if new_id == (ts+1)*1000:
            err = "all 9999 track ids for timestep {} used up!".format(ts)
            raise Exception(err)
    return new_id

#------------------------------------------------------------------------------

cdef class FeatureTrackSplitter:

    def __cinit__(self, used_ids=None):
        self.debug = False
        self.used_ids = used_ids

    def split(self, track, n, preserve=False):
        """Split a track into subtracks by eliminating branching.

        At each branching, only the dominant track is continued (highest
        individual successor probability). The others are turned into new
        tracks.

        Optionally, a check is conducted whether two branches merge during
        the previous/next n timesteps before/after a merging/splitting, in
        which case the branching is retained (for those branches).
        To omit this, pass n < 0.

        The original track is destroyed unless explicitly stated otherwise.

        Returns all new tracks.
        """
        debug = self.debug

        if n < 0 or track.n < 3:
            return [track]
        self.track_config = track.config

        if preserve:
            graph = copy(track.graph)
        else:
            graph = track.graph

        if debug: log.debug("")
        if debug: log.debug("split track {} ({} vertices, {} edges)".format(track.id, len(graph.vs), len(graph.es)))
        if debug: log.debug("{} vertices:\n{}".format(len(graph.vs), "\n  ".join([vertex2str(v) for v in graph.vs])))
        if debug: log.debug("{} edges:\n{}".format(len(graph.vs), "\n  ".join([edge2str(e) for e in graph.es])))
        if debug: log.debug("")

        retained_edges = set()

        # Process all splittings
        splittings = graph.vs.select(lambda v: "splitting" in v["type"])
        if debug: log.debug("process {} splittings".format(len(splittings)))
        for vertex in splittings:
            direction=1
            self._process_branching_vertex(graph, vertex, n, direction,
                    retained_edges)

        # Process all mergings
        # Note: mergings must be here, determined after the splittings have
        # been processed, because some mergings have potentially already
        # been removed above (cases where a merging with only two children
        # immediately follows a splitting along a detached branch)
        mergings = graph.vs.select(lambda v: "merging" in v["type"])
        if debug: log.debug("process {} mergings".format(len(mergings)))
        for vertex in mergings:
            direction = -1
            self._process_branching_vertex(graph, vertex, n, direction,
                    retained_edges)

        # Recompute tracking probabilities and remove edges
        # with too low probabilities (iterative approach)
        iter_max = 100000
        for iter_i in range(iter_max):
            self.recompute_tracking_probabilities(graph, copy(retained_edges))
            n_rm = self.remove_low_probability_edges(graph, retained_edges)
            if n_rm == 0:
                break
        else:
            err = ("timeout while recomputing tracking probabilities "
                    "(aborted after {} iterations)").format(iter_i)
            raise Exception(err)

        # Split into subgraphs
        subgraphs = self._split_graph(graph)
        if debug: log.debug("split into {} subgraphs".format(len(subgraphs)))

        # Remove all vertices from the old graph, now
        # that all features are linked to new graphs
        # (probably totally unnecessary, but cannot hurt)
        for _ in range(track.n):
            graph.delete_vertices(graph.vs[0])

        # Create new tracks from the subgraphs
        subtracks = []
        for subgraph in subgraphs:
            ts_start = min(subgraph.vs["ts"])
            tracker = track.tracker
            kwas = dict(graph=subgraph, config=track.config)
            if tracker is None:
                if self.used_ids is None:
                    err = ("if feature tracks are not linked to a tracker, "
                            "{} must be initialized with used_ids").format(
                            self.__class__.__name__)
                    raise Exception(err)
                id_ = new_track_id(ts_start, self.used_ids)
                self.used_ids.add(id_)
                kwas["id_"] = id_
            else:
                kwas["tracker"] = tracker
                kwas["timestep"] = ts_start
            subtrack = FeatureTrack(**kwas)
            subtracks.append(subtrack)
            if debug: log.debug(" -> create new track {} (n={})".format(subtrack.id, subtrack.n))

        return subtracks

    def recompute_tracking_probabilities(self, graph, retained_edges):
        cdef float p_tot, p_size, p_overlap
        while retained_edges:
            name_source, name_target = retained_edges.pop()
            vxi_source = graph.vs.find(name_source).index
            vxi_target = graph.vs.find(name_target).index
            edge = graph.es.find(_between=((vxi_source,), (vxi_target,)))

            vs_old = graph.vs[edge.target].predecessors()
            vs_new = graph.vs[edge.source].successors()
            vis_old = [v.index for v in vs_old]
            vis_new = [v.index for v in vs_new]
            edges = graph.es.select(_between=(vis_old, vis_new))

            # Determine direction
            if len(vis_old) == 1 and len(vis_new) >= 1:
                direction = 1
                parent = vs_old[0]
            elif len(vis_old) > 1 and len(vis_new) == 1:
                direction = -1
                parent = vs_new[0]
            else:
                err = "invalid vertices: {}x OLD ({}), {}x NEW ({})".format(
                        len(vis_old), [v["feature"].id for v in vs_old],
                        len(vis_new), [v["feature"].id for v in vs_new])
                raise Exception(err)

            # Collect sizes and overlaps of children
            size_parent = parent["feature"].n
            overlap_children = 0
            size_children = 0
            p_shares_tot = 0
            for edge in edges:
                vx_source = graph.vs[edge.source]
                vx_target = graph.vs[edge.target]
                key = (vx_source["name"], vx_target["name"])
                if key in retained_edges:
                    retained_edges.remove(key)

                if direction > 0:
                    child = vx_target
                else:
                    child = vx_source

                size_child = child["feature"].n
                size_children += size_child
                overlap_children += edge["n_overlap"]

                # Compute individual p_tot to later compute p_shares
                compute_tracking_probabilities(
                        &p_tot, &p_size, &p_overlap,
                        size_parent, size_child, edge["n_overlap"],
                        self.track_config["f_size"],
                        self.track_config["f_overlap"],
                        self.track_config["min_p_size"],
                        self.track_config["min_p_overlap"],
                    )
                edge["p_share"] = p_tot
                p_shares_tot += p_tot

            # Compute overall tracking probabilities
            compute_tracking_probabilities(
                    &p_tot, &p_size, &p_overlap,
                    size_parent, size_children, overlap_children,
                    self.track_config["f_size"],
                    self.track_config["f_overlap"],
                    self.track_config["min_p_size"],
                    self.track_config["min_p_overlap"],
                )
            edge["p_tot"] = p_tot
            edge["p_size"] = p_size
            edge["p_overlap"] = p_overlap

            # Finalize p_shares
            for edge in edges:
                if p_shares_tot > 0:
                    edge["p_share"] /= p_shares_tot

    def remove_low_probability_edges(self, graph, retained_edges):

        n_rm = 0
        vs_source = set()
        vs_target = set()
        es_del = set()

        # Determine edges to remove based on p_tot
        for name_source, name_target in copy(retained_edges):
            vx_source = graph.vs.find(name_source)
            vx_target = graph.vs.find(name_target)
            edge = graph.es.find(_between=(
                    (vx_source.index,), (vx_target.index,)))
            if edge["p_tot"] < self.track_config["min_p_tot"]:
                vs_source.add(vx_source)
                vs_target.add(vx_target)
                es_del.add(edge.index)
                retained_edges.remove((name_source, name_target))
                n_rm += 1

        if n_rm > 0:
            # Remove edges
            graph.delete_edges(es_del)

            # Adapt vertex types
            for vx_source in vs_source:
                if len(vx_source.successors()) == 1:
                    self._adapt_vertex_type_parent(vx_source, direction=1)
            for vx_target in vs_target:
                if len(vx_target.predecessors()) == 1:
                    self._adapt_vertex_type_parent(vx_target, direction=-1)

        return n_rm

    def _split_graph(self, graph):

        # Unfortunately, decomposing a directed graph currently doesn't work..
        # WEAK mode should work (STRONG is not implemented), yet doesn't!
        # Use workaround below instead (likely much slower)
        #subgraphs = graph.decompose(mode="weak")

        #WORKAROUND<

        # Store edge attributes
        es_attrs = {}
        for edge in graph.es:
            source_name = graph.vs[edge.source]["name"]
            target_name = graph.vs[edge.target]["name"]
            es_attrs[(source_name, target_name)] = edge.attributes()

        # Convert graph to undirected and split it
        # WARNING: edge attributes are lost during conversion
        graph.to_undirected()
        subgraphs = graph.decompose(mode="weak")

        # Convert the subgraphs back to directed
        for subgraph in subgraphs:

            # Add all edges twice (both backwards and forwards in time)
            # and remove the half that point backwards in time
            subgraph.to_directed(mutual=True)
            es_del = subgraph.es.select(lambda e: (
                    e.graph.vs[e.source]["ts"] > e.graph.vs[e.target]["ts"]))
            subgraph.delete_edges(es_del)

            # Insert edge attributes
            for edge in subgraph.es:
                source_name = subgraph.vs[edge.source]["name"]
                target_name = subgraph.vs[edge.target]["name"]
                key = (source_name, target_name)
                edge.update_attributes(es_attrs[key])

            # Add attributes even in absence of edges
            # (otherwise edge argument checks fail)
            if len(subgraph.es) == 0:
                for key in FeatureTrack.es_attrs:
                    subgraph.es[key] = None

        #WORKAROUND>

        return subgraphs

    def _process_branching_vertex(self, graph, vertex, n, direction,
            retained_edges):
        debug = self.debug
        _name_ = "_process_branching_vertex"
        if debug: log.debug("> {}: vertex ({}@{}: {}), n {}, direction {}".format(_name_, vertex["feature"].id, vertex["ts"], vertex["type"], n, direction))

        # Collect all edges involved in the branching
        edges_del = self._collect_edges(vertex, direction)
        if debug: log.debug("collected {} branching edges".format(len(edges_del)))
        if len(edges_del) == 0:
            err = "{}: len(edges_del) == 0: {} {} {}".format(_name_,
                    vertex, vertex.neighbors(), direction)
            raise Exception(err)

        # Select winner by successor probability/-ies
        edges_del = self._sort_edges_unambiguously(vertex, edges_del,
                direction, nlimit=120)
        edge_winner = edges_del.pop(0)
        if debug: log.debug("select major edge {}".format(edge2str(edge_winner)))
        edges_keep = [edge_winner]
        vertex_winner = self._get_other_vertex(vertex, edge_winner)

        # Check whether any other branches should be retained
        # This is the case if, during a certain timestep window,
        # either the branches remerge, or one of them finishes
        # Otherwise, the branching is eliminated
        if n > 0:

            # Retain all edges whose branches finish in the timestep window
            if n > 1:
               self._retain_edges_short_branch(graph, edges_del, edges_keep,
                       direction, n-2)

            # Starting from the main edge, collect all reverse branchings
            # in the given timestep window defined by n
            branchings_winner = set()
            self._collect_reverse_branchings(
                    vertex_winner, direction, n-1, branchings_winner)

            # If there are any reverse branchings in reach of the main branch,
            # then check whether any of these is in reach of any of the other
            # edges, and retain those edges (because the branches remerge)
            if len(branchings_winner) > 0:
                self._find_edges_to_keep(vertex, edge_winner, branchings_winner,
                        edges_del, edges_keep, direction, n)

        if debug: log.debug("keep {}/{} edges".format(len(edges_keep), len(edges_keep) + len(edges_del)))

        # If only one branch is to be kept, eliminate the branching,
        # i.e., adapt the vertex type of the parent (rest done below)
        # (Note: if merging/splitting, only one of them is eliminated now)
        if len(edges_keep) == 1:
            self._adapt_vertex_type_parent(vertex, direction)

        # Adapt vertex types of children and remove edges to be eliminated
        edges_del_named = []
        for edge in edges_del:
            key = (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"])
            if key in retained_edges:
                retained_edges.remove(key)
            assert edge not in edges_keep #SR_DBG
            other_vertex = self._get_other_vertex(vertex, edge)
            self._adapt_vertex_type_child(other_vertex, direction)
            edges_del_named.append((
                    graph.vs[edge.source]["name"],
                    graph.vs[edge.target]["name"]))

        # Collect names of edges to be kept; must be done before deleting
        # edges because deleting from the graph changes edge indices
        for edge in edges_keep:
            key = (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"])
            retained_edges.add(key)

        # Delete edges from graph
        if debug: log.debug("delete {} edges from graph".format(len(edges_del_named)))
        graph.delete_edges(edges_del_named)

    def _get_other_vertex(self, vertex, edge):
        if edge.source == vertex.index:
            return vertex.graph.vs[edge.target]
        elif edge.target == vertex.index:
            return vertex.graph.vs[edge.source]
        else:
            raise Exception("other vertex not found (should not happen)")

    def _collect_edges(self, vertex, direction):
        if direction > 0:
            return [vertex.graph.es.find(_between=((vertex.index,), (s.index,))
                    ) for s in vertex.successors()]
        return [vertex.graph.es.find(_between=((vertex.index,), (p.index,))
                ) for p in vertex.predecessors()]

    def _retain_edges_short_branch(self, graph, edges_del, edges_keep,
            direction, n):
        """Retain all edges whose branches finish in the given window."""

        # Select all ends vertices of the track in the respective direction
        if direction > 0:
            vs_ends = list(graph.vs.select(type_in=("lysis", "stop")))

            # Add active heads (if track unfinished)
            if "active_head" in graph.vs.attributes():
                vs_heads = list(graph.vs.select(active_head=True))
                for vx in vs_heads:
                    if vx not in vs_ends:
                        vs_ends.append(vx)

        elif direction < 0:
            vs_ends = list(graph.vs.select(type_in=("genesis", "start")))

        # For each edge, get the max. directed distance to an end
        # Retain edge if max. directed distance to end in window
        for edge in edges_del:
            vid = edge.target if direction > 0 else edge.source
            vertex = graph.vs[vid]
            mode = "OUT" if direction > 0 else "IN"
            pathlens = vertex.shortest_paths(vs_ends, mode=mode)
            pathlens = [l for l in pathlens[0] if not np.isinf(l)]
            if len(pathlens) > 0 and max(pathlens) <= n:
                edges_del.remove(edge)
                edges_keep.append(edge)

    def _collect_reverse_branchings(self, vertex, direction, n, branchings):

        # Add to list if reverse branching
        if (direction > 0 and "merging" in vertex["type"] or
                direction < 0 and "splitting" in vertex["type"]):
            branchings.add(vertex)

        # Check if we're done
        if n == 0:
            return

        # Continue along the next edges
        edges = self._collect_edges(vertex, direction)
        for edge in edges:
            other_vertex = self._get_other_vertex(vertex, edge)
            self._collect_reverse_branchings(
                    other_vertex, direction, n-1, branchings)

    def _find_edges_to_keep(self, vertex, edge_main, branchings_main,
            edges_del, edges_keep, direction, n):
        edges_keep_new = []
        branchings_edges = {}
        for edge in edges_del.copy():
            other_vertex = self._get_other_vertex(vertex, edge)
            branchings_edge = set()

            # Collect all branchings in the opposite direction
            # in the given timestep window defined by n
            self._collect_reverse_branchings(
                    other_vertex, direction, n-1, branchings_edge)

            # If any of these branchings in reach of this edge is
            # also in reach of the main edge, then retain this edge
            # (such a shared edge means the two branches remerge)
            if len(branchings_main.intersection(branchings_edge)) > 0:
                branchings_edges[edge] = branchings_edge
                edges_del.remove(edge)
                edges_keep.append(edge)
                edges_keep_new.append(edge)

        for edge in edges_keep_new:
            self._find_edges_to_keep(vertex, edge, branchings_edges[edge],
                    edges_del, edges_keep, direction, n)

    #SR_TODO better name that expresses edge elimination
    def _adapt_vertex_type_parent(self, vertex, direction):
        debug = self.debug
        _name_ = "_adapt_vertex_type_parent"
        if debug: log.debug("< {}: {} (direction {})".format(_name_, vertex2str(vertex), direction))
        type = vertex["type"]
        if direction > 0:
            if type == "splitting":
                vertex["type"] = "continuation"
            elif type == "start/splitting":
                vertex["type"] = "start"
            elif type == "genesis/splitting":
                vertex["type"] = "genesis"
            elif type == "merging/splitting":
                vertex["type"] = "merging"
            else:
                err = "{}.{}: merging: child type {}".format(
                        self.__class__.__name__, _name_, type)
                raise NotImplementedError(err)
        elif direction < 0:
            if type == "merging":
                vertex["type"] = "continuation"
            elif type == "merging/stop":
                vertex["type"] = "genesis/stop"
            elif type == "merging/lysis":
                vertex["type"] = "genesis/lysis"
            elif type == "merging/splitting":
                vertex["type"] = "splitting"
            else:
                err = "{}.{}: merging: child type {}".format(
                        self.__class__.__name__, _name_, type)
                raise NotImplementedError(err)
        else:
            raise ValueError("invalid direction: {}".format(direction))
        if debug: log.debug(" -> {}".format(vertex2str(vertex)))

    #SR_TODO better name that expresses edge elimination
    def _adapt_vertex_type_child(self, vertex, direction):
        debug = self.debug
        _name_ = "_adapt_vertex_type_child"
        if debug: log.debug("< {}: {} (direction {})".format(_name_, vertex2str(vertex), direction))
        type = vertex["type"]

        # Forward in time
        if direction > 0:

            if type == "continuation":
                vertex["type"] = "genesis"

            elif type in ["stop", "lysis", "splitting"]:
                vertex["type"] = "genesis/{}".format(type)

            elif "merging" in type:
                if len(vertex.predecessors()) == 1:
                    if type == "merging":
                        vertex["type"] = "continuation"
                    elif type.startswith("merging/"):
                        vertex["type"] = type.replace("merging/", "")
                    else:
                        raise NotImplementedError("child/merging")

            else:
                err = "{}.{}: merging: child type {}".format(
                        self.__class__.__name__, _name_, type)
                raise NotImplementedError(err)

        # Backward in time
        elif direction < 0:

            if type == "continuation":
                vertex["type"] = "lysis"

            elif type in ["start", "genesis", "merging"]:
                vertex["type"] = "{}/lysis".format(type)

            elif "splitting" in type:
                if len(vertex.successors()) == 1:
                    if type == "splitting":
                        vertex["type"] = "continuation"
                    elif type.startswith("splitting/"):
                        vertex["type"] = type.replace("splitting/", "")
                    else:
                        raise NotImplementedError("child/splitting")

            else:
                err = "{}.{}: merging: child type {}".format(
                        self.__class__.__name__, _name_, type)
                raise NotImplementedError(err)

        else:
            raise ValueError("invalid direction: {}".format(direction))
        if debug: log.debug(" -> {}".format(vertex2str(vertex)))

    def _sort_edges_unambiguously(self, vertex, edges, direction, nlimit):
        _name_ = "_sort_edges_unambiguously"

        if len(edges) < 2:
            return edges

        # Bigger p_share wins
        edges.sort(key=lambda e: e["p_share"], reverse=True)
        if edges[0]["p_share"] > edges[1]["p_share"]:
            return edges

        # Bigger p_tot wins
        tmp = [e for e in edges if e["p_share"] == edges[0]["p_share"]]
        tmp.sort(key=lambda e: e["p_tot"], reverse=True)
        edges = tmp + edges[len(tmp):]
        if edges[0]["p_tot"] > edges[1]["p_tot"]:
            return edges

        # Function to fetch feature of 'other vertex'
        if direction > 0:
            vx_other_feature = lambda e: e.graph.vs[e.target]["feature"]
        elif direction < 0:
            vx_other_feature = lambda e: e.graph.vs[e.source]["feature"]

        # Bigger 'other feature' wins
        edges.sort(key=lambda e: vx_other_feature(e).n, reverse=True)
        if vx_other_feature(edges[0]).n > vx_other_feature(edges[1]).n:
            return edges

        # Stronger 'other feature' wins
        edges.sort(key=lambda e: vx_other_feature(e).sum(), reverse=True)
        if vx_other_feature(edges[0]).sum() > vx_other_feature(edges[1]).sum():
            return edges
        edges.sort(key=lambda e: vx_other_feature(e).max(), reverse=True)
        if vx_other_feature(edges[0]).max() > vx_other_feature(edges[1]).max():
            return edges

        # Ambiguity could not be resolved
        # Abort if 'other feature' is above size threshold
        n = vx_other_feature(edges[0]).n
        log.warning(("{}: ambiguous 'winning edge': multiple edges have same "
                " biggest value for p_share/p_tot: {}/{}").format(_name_,
                edges[0]["p_share"], edges[0]["p_tot"]))
        if n > nlimit:
            err = ("{}: feature adjacent to 'winning edge' too big to ignore "
                    "ambiguity ({} > {}); if {} is still acceptably small, "
                    "consider increasing nlimit={}").format(_name_, n, nlimit,
                    n, nlimit)
            raise Exception(err)
        log.warning(("{}: feature adjacent to 'winning edge' small enough "
                "to ignore ambiguity ({} <= {}); if {} is unacceptably big, "
                "consider decreasing nlimit={}").format(_name_, n, nlimit,
                nlimit, nlimit))
        return edges

#------------------------------------------------------------------------------

#DBG_PERMANENT<
def vertex2str(vertex):
    return "[{}@{}: {}]".format(
            vertex["feature"].id,
            vertex["ts"],
            vertex["type"],
        )
def edge2str(edge):
    graph = edge.graph
    return "([{}]<({:5.3f}/{:5.3f}/{:5.3f}/{:5.3f}/{:5.3f})>[{}])".format(
            graph.vs[edge.source]["feature"].id,
            edge["n_overlap"]   if edge["n_overlap"]    else -1,
            edge["p_tot"]       if edge["p_tot"]        else -1,
            edge["p_share"]     if edge["p_share"]      else -1,
            edge["p_size"]      if edge["p_size"]       else -1,
            edge["p_overlap"]   if edge["p_overlap"]    else -1,
            graph.vs[edge.target]["feature"].id,
        )
#DBG_PERMANENT>

#------------------------------------------------------------------------------

cdef list all_combinations(list elements, int nmin, int nmax):
    return list(itertools.chain.from_iterable(itertools.combinations(
            elements, r) for r in range(nmin, nmax + 1)))

cdef void successor_combinations_extend(SuccessorCandidates* combinations,
        int max_children):
    cdef int i, nmax_old = combinations.max, nmax_new = 5*nmax_old
    cdef SuccessorCandidate* tmp = <SuccessorCandidate*>malloc(
            nmax_old*sizeof(SuccessorCandidate))
    for i in range(nmax_old):
        tmp[i].n = combinations.candidates[i].n
        tmp[i].max = combinations.candidates[i].max
        tmp[i].children = combinations.candidates[i].children
        tmp[i].p_shares = combinations.candidates[i].p_shares
        tmp[i].n_overlaps = combinations.candidates[i].n_overlaps
        tmp[i].parent = combinations.candidates[i].parent
        tmp[i].direction = combinations.candidates[i].direction
        tmp[i].p_tot = combinations.candidates[i].p_tot
        tmp[i].p_size = combinations.candidates[i].p_size
        tmp[i].p_overlap = combinations.candidates[i].p_overlap
    free(combinations.candidates)
    combinations.candidates = <SuccessorCandidate*>malloc(
            nmax_new*sizeof(SuccessorCandidate))
    for i in range(nmax_old):
        combinations.candidates[i].n = tmp[i].n
        combinations.candidates[i].max = tmp[i].max
        combinations.candidates[i].children = tmp[i].children
        combinations.candidates[i].p_shares = tmp[i].p_shares
        combinations.candidates[i].n_overlaps = tmp[i].n_overlaps
        combinations.candidates[i].parent = tmp[i].parent
        combinations.candidates[i].direction = tmp[i].direction
        combinations.candidates[i].p_tot = tmp[i].p_tot
        combinations.candidates[i].p_size = tmp[i].p_size
        combinations.candidates[i].p_overlap = tmp[i].p_overlap
    free(tmp)
    for i in range(nmax_old, nmax_new):
        combinations.candidates[i].n = 0
        combinations.candidates[i].max = max_children
        combinations.candidates[i].children = <cRegion**>malloc(
                max_children*sizeof(cRegion*))
        combinations.candidates[i].p_shares = <np.float32_t*>malloc(
                max_children*sizeof(np.float32_t))
        combinations.candidates[i].n_overlaps = <np.int32_t*>malloc(
                max_children*sizeof(np.int32_t))
        for j in range(max_children):
            combinations.candidates[i].children[j] = NULL
            combinations.candidates[i].p_shares[j] = 0.0
            combinations.candidates[i].n_overlaps[j] = -1
        combinations.candidates[i].parent = NULL
        combinations.candidates[i].direction = 0
        combinations.candidates[i].p_tot = 0.0
        combinations.candidates[i].p_size = 0.0
        combinations.candidates[i].p_overlap = 0.0
    combinations.max = nmax_new

#==============================================================================

cpdef FeatureTrack FeatureTrack_rebuild(np.uint64_t id_, object graph,
        dict config
    ):
    track = FeatureTrack(id_=id_, graph=graph, config=config)
    return track

#SR_TODO turn into proper extension class (i.e. cythonize methods etc.)
cdef class FeatureTrack:

    gr_attrs = set()
    vs_attrs = set(["feature", "name", "ts", "type",
            "missing_successors", "missing_predecessors"])
    vs_attrs_opt = set(["active_head", "feature_id"])
    es_attrs = set(["n_overlap", "p_tot", "p_size", "p_overlap", "p_share"])

    def __cinit__(self, *,
            id_         = None,
            feature     = None,
            type        = "genesis",
            features    = None,
            graph       = None,
            config      = None,
            tracker     = None,
            timestep    = None,
            _cache      = None,
        ):
        if id_ is None:
            if tracker is None or timestep is None:
                err = "either id_ must be pass, or tracker along with timestep"
                raise ValueError(err)
            id_ = tracker.new_track_id(timestep)
        if tracker is not None:
            tracker.register_id(id_)
        self.id = id_

        self.tracker = tracker

        self.debug = False

        self.config = {} if config is None else dict(config)

        # Initialize graph: either empty, create from features, or passed in
        if graph is None:
            if features is None:
                self._init_empty_graph()
            else:
                raise NotImplementedError("init graph from features")
        else:
            self.graph = graph
            if features is None:
                features = graph.vs["feature"]
            for feature_ in features:
                feature_.set_track(self)

            #SR_TMP< TODO remove once no more old track files in use
            if "start" in graph.vs.attributes():
                del graph.vs["start"]
            if "head" in graph.vs.attributes():
                del graph.vs["head"]
            #SR_TMP>

            #SR_TMP< TODO not sure whether this should actually be retained..?
            if "missing_successors" not in graph.vs.attributes():
                graph.vs["missing_successors"] = None
            if "missing_predecessors" not in graph.vs.attributes():
                graph.vs["missing_predecessors"] = None
            if len(graph.es) == 0:
                if "n_overlap" not in graph.es.attributes():
                    graph.es["n_overlap"] = None
            #SR_TMP>

        self._graph_check_attrs()

        # Add first feature
        if feature is not None:
            vertex = track_graph_add_feature(
                    self.graph,
                    feature,
                    attrs=dict(head=True, type=type),
                )
            feature.set_track(self)

        # If the track is incomplete, total track stats and individual
        # missing feature stats are stored, the latter as a dict of lists
        self.set_total_track_stats(None)
        self.set_missing_features_stats(None)

        # Init names and functions for total track and missing features stats
        # as odicts (keys: names; values: stats extraction functions)
        # These are used to ensure consistent ordering for comparable files
        self._init_total_track_stats_keys_fcts()
        self._init_missing_features_stats_keys_fcts()

        # Initialize cache
        self._cache = {} if _cache is None else _cache

    def __copy__(self):
        return deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__(
                id_         = self.id,
                graph       = deepcopy(self.graph),
                config      = deepcopy(self.config),
                tracker     = self.tracker,
                _cache      = deepcopy(self._cache)
            )
        other.set_total_track_stats(deepcopy(self.total_track_stats))
        other.set_missing_features_stats(deepcopy(self.missing_features_stats))
        assert self == other
        return other

    def _init_empty_graph(self):
        self.graph = ig.Graph(directed=True)
        for key in self.gr_attrs:
            self.graph[key] = None
        for key in self.vs_attrs:
            self.graph.vs[key] = None
        for key in self.es_attrs:
            self.graph.es[key] = None

    def _graph_check_attrs(self, graph=None):
        if graph is None:
            graph = self.graph

        # Check graph attributes
        if not set(graph.attributes()) == self.gr_attrs:
            err = ("track {}: graph attributes:\nexpected:\n    {}\n"
                    "found:\n    {}").format(self.id,
                    "\n    ".join([str(a) for a in sorted(self.gr_attrs)]),
                    "\n    ".join([str(a) for a in sorted(graph.attributes())]))
            raise Exception(err)

        # Check vertex attributes
        graph_vs_attrs = graph.vs.attributes()
        if (any(a not in self.vs_attrs.union(self.vs_attrs_opt)
                    for a in graph_vs_attrs) or
                not all(a in graph_vs_attrs for a in self.vs_attrs)):
            err = ("track {}: vertex attributes:\nexpected:\n    {}\n"
                    "found:\n    {}").format(self.id,
                    "\n    ".join([str(a) for a in sorted(self.vs_attrs)]),
                    "\n    ".join([str(a) for a in sorted(graph.vs.attributes())]))
            raise Exception(err)

        # Check edge attributes
        #if not set(graph.es.attributes()) == self.es_attrs:
        #    err = "track {}: edge attributes: expected {}, found {}".format(
        #            self.id, sorted(self.es_attrs),
        #            sorted(graph.es.attributes()))
        #    raise Exception(err)

    def __repr__(self):
        incomplete = "" if self.is_complete() else " !INCOMPLETE!"
        return "<FeatureTrack[{id0}]: id={id1}, n={n}, d={d}{ic}>".format(
                id0=id(self), id1=self.id, n=self.n, d=self.duration(),
                ic=incomplete)

    def __reduce__(self):
        return FeatureTrack_rebuild, (self.id, self.graph, self.config)

    def __hash__(self):
        """Hash based on select attributes."""
        attrs = (
                self.id,
                self.n,
                self.ts_start(total=False),
                self.ts_end(total=False),
            )
        return hash(attrs)

    def __richcmp__(self, FeatureTrack other, int op):
        # <   0
        # <=  1
        # ==  2
        # !=  3
        # >   4
        # >=  5
        if op == Py_EQ:
            if not isinstance(other, self.__class__):
                return False
            elif (  self.id == other.id and
                    self.n == other.n and
                    set(self.feature_ids()) == set(other.feature_ids())
                ):
                return True
            return False
        elif op == Py_NE:
            return not self == other
        elif op == Py_LT:
            if not isinstance(other, self.__class__):
                err = "invalid types for comparison: {} < {}".format(
                        self.__class__.__name__, other.__class__.__name__)
                raise ValueError(err)
            return self.id < other.id
        elif op == Py_GT:
            if not isinstance(other, self.__class__):
                err = "invalid types for comparison: {} < {}".format(
                        self.__class__.__name__, other.__class__.__name__)
                raise ValueError(err)
            return self.id > other.id
        elif op == Py_LE:
            if self == other:
                return True
            return self < other
        elif op == Py_GE:
            if self == other:
                return True
            return self > other
        raise NotImplementedError("comparison operation {}".format(op))

    def is_finished(self):
        if "active_head" not in self.graph.vs.attributes():
            return True
        return not any(self.graph.vs["active_head"])

    def is_active(self):
        return not self.is_finished()

    def is_complete(self):
        """Whether the track is complete or only a partial subtrack."""
        data_pred = self.graph.vs["missing_predecessors"]
        data_succ = self.graph.vs["missing_successors"]
        return not any(data_pred) and not any(data_succ)

    #SR_TODO merge with FeatureTrack.n
    def size(self, total=True):
        if total and not self.is_complete():
            return self.total_track_stats["n"]
        n = len(self.graph.vs)
        return n

    def timesteps(self, *, total=True, ts_start=None, ts_end=None):
        """Return timesteps.

        Argument 'total' is only relevant for partial tracks; if true, missing
        timesteps (i.e. those that belong to other partial tracks of the
        same track) are considered, too.
        """
        timesteps = sorted(set(self.graph.vs["ts"]))
        if (total and not self.is_complete() and
                self.missing_features_stats is not None):
            missing_timesteps = self.missing_features_stats["timestep"]
            timesteps = sorted(set(timesteps + missing_timesteps))
        if ts_start is not None:
            timesteps = [ts for ts in timesteps if ts >= ts_start]
        if ts_end is not None:
            timesteps = [ts for ts in timesteps if ts <= ts_end]
        return timesteps

    def clear_cache(self, key=None):
        if key is not None:
            self._cache = None
        elif key in self._cache:
            del self._cache[key]

    #--------------------------------------------------------------------------

    @classmethod
    def from_features_linear(cls, features_ts, id_, config=None,
            edge_attrs=None):
        """Create linear track from features at equidistant timesteps.

        This is particularly useful to create simple tracks for testing.
        """
        if edge_attrs is not None:
            raise NotImplementedError("edge_attrs")

        # Check timesteps
        if len(features_ts) > 2:
            timesteps = sorted(features_ts.keys())
            dts = timesteps[1] - timesteps[0]
            #SR_TMP<
            if dts > 1:
                raise NotImplementedError("dts > 1")
            #SR_TMP>
            for i, ts_i in enumerate(timesteps[:-1]):
                dts_i = timesteps[i+1] - ts_i
                if dts_i != dts:
                    err = "timesteps not equidistant: {} != {}".format(
                            dts_i, dts)
                    raise ValueError(err)

        # Set timesteps on features
        for timestep, feature in features_ts.items():
            feature.timestep = timestep

        # Create empty track
        track = cls(id_=id_, config=config)
        for feature in features_ts.values():
            feature.set_track(track)

        # Add features
        for timestep, feature in sorted(features_ts.items()):
            vx_attrs = None #SR_TMP
            track_graph_add_feature(track.graph, feature, attrs=vx_attrs)

        # Add edges
        for ts0, ts1 in zip(timesteps[:-1], timesteps[1:]):
            feature0 = features_ts[ts0]
            feature1 = features_ts[ts1]
            vx0, vx1 = feature0.vertex(), feature1.vertex()
            eg_attrs = None #SR_TMP
            track_graph_add_edge(track.graph, vx0, vx1, attrs=eg_attrs)

        return track

    #--------------------------------------------------------------------------

    def _init_total_track_stats_keys_fcts(self, cache=False):
        self._total_track_stats_keys_fcts_lst = [
                ("n"          , lambda t: int(t.n)),
                ("duration"   , lambda t: int(t.duration())),
                ("footprint_n", lambda t: int(t.footprint_n(cache=cache))),
            ]

    def _init_missing_features_stats_keys_fcts(self):
        self._missing_features_stats_keys_fcts_lst = [
                ("id"           , lambda f: int(f.id)),
                ("timestep"     , lambda f: int(f.timestep)),
                ("center"       , lambda f: [int(i) for i in f.center]),
                ("n_holes"      , lambda f: 0 if f.holes is None else len(f.holes)),
                ("associates_n" , lambda f: f.associates_n()),
                ("n"            , lambda f: int(f.n)),
                ("sum"          , lambda f: float(np.nansum   (f.values))         if f.values.size > 0 else -1.0),
                ("min"          , lambda f: float(np.nanmin   (f.values))         if f.values.size > 0 else -1.0),
                ("max"          , lambda f: float(np.nanmax   (f.values))         if f.values.size > 0 else -1.0),
                ("mean"         , lambda f: float(np.nanmean  (f.values))         if f.values.size > 0 else -1.0),
                ("median"       , lambda f: float(np.nanmedian(f.values))         if f.values.size > 0 else -1.0),
                ("abssum"       , lambda f: float(np.nansum   (np.abs(f.values))) if f.values.size > 0 else -1.0),
                ("absmin"       , lambda f: float(np.nanmin   (np.abs(f.values))) if f.values.size > 0 else -1.0),
                ("absmax"       , lambda f: float(np.nanmax   (np.abs(f.values))) if f.values.size > 0 else -1.0),
                ("absmean"      , lambda f: float(np.nanmean  (np.abs(f.values))) if f.values.size > 0 else -1.0),
                ("absmedian"    , lambda f: float(np.nanmedian(np.abs(f.values))) if f.values.size > 0 else -1.0),
            ]

    #SR_TMP< TODO turn into proper getter, or guard against assignment
    @property
    def total_track_stats(self):
        if self._total_track_stats_lst is None:
            return None
        return odict(self._total_track_stats_lst)
    #SR_TMP>

    def reset_total_track_stats(self):
        self.set_total_track_stats(None)

    def set_total_track_stats(self, stats, *, ignore_missing=False, skip=None):

        if stats is None:
            self._total_track_stats_lst = None
            return

        keys = [k for k, f in self._total_track_stats_keys_fcts_lst]

        # Check argument stats
        if (set(keys) != set(stats.keys()) and not ignore_missing):
            expected = set(keys)
            found = set(stats.keys())
            missing = expected.difference(found)
            extra = found.difference(expected)
            err = "invalid total track stats: {} expected: {}".format(
                    len(expected), ", ".join(sorted(expected)))
            if len(missing) > 0:
                err += "; {} missing: {}".format(len(missing), ", ".join(
                        sorted(missing)))
            if len(extra) > 0:
                err += "; {} extra: {}".format(len(extra), ", ".join(
                        sorted(extra)))
            raise ValueError(err)

        # Transfer argument stats in correct order
        self._total_track_stats_lst = []
        for key in keys:
            if skip is not None and key in skip:
                self._total_track_stats_lst.append((key, None))
            else:
                self._total_track_stats_lst.append((key, stats[key]))

    def n_missing_features(self):

        if (self.missing_features_stats is None or
                len(self.missing_features_stats["id"]) == 0):
            if not self.is_complete():
                err = ("track {}: incomplete yet lacks missing features stats"
                        ).format(self.id)
                raise Exception(err)
            return 0

        fids = self.missing_features_stats["id"]

        if len(fids) > len(set(fids)):
            err = ("track {}: double entries in missing features stats: {}"
                    ).format(sorted(set([fid for fid in fids
                            if fids.count(fid) > 1])))
            raise Exception(err)

        return len(fids)

    #SR_TMP< TODO turn into proper getter, or guard against assignment
    @property
    def missing_features_stats(self):
        if self._missing_features_stats_lst is None:
            return None
        return odict(self._missing_features_stats_lst)
    #SR_TMP>

    def set_missing_features_stats(self, stats, ignore_missing=False):

        if stats is None:
            self._missing_features_stats_lst = None
            return

        keys = [k for k, f in self._missing_features_stats_keys_fcts_lst]

        # Check argument stats
        if (not set(keys) == set(stats.keys()) and
                not ignore_missing):
            # Error!

            expected = set(keys)
            found = set(stats.keys())
            missing = expected.difference(found)
            extra = found.difference(expected)

            err = "invalid missing features stats: {} expected: {}".format(
                    len(expected), ", ".join(sorted(expected)))
            if len(missing) > 0:
                err += "; {} missing: {}".format(len(missing), ", ".join(
                        sorted(missing)))
            if len(extra) > 0:
                err += "; {} extra: {}".format(len(extra), ", ".join(
                        sorted(extra)))
            raise ValueError(err)

        # Transfer argument stats in correct order
        self._missing_features_stats_lst = []
        for key in keys:
            self._missing_features_stats_lst.append((key, stats[key]))

    def update_missing_features_stats(self):

        if self.is_complete():
            self.set_missing_features_stats(None)
            return

        if self.missing_features_stats is None:
            err = ("incomplete track ({}) without missing feature stats"
                    ).format(self.id)
            raise Exception(err)

        # IDs of features currently present
        fids_present = [f.id for f in self.features()]

        # IDs of features previously missing
        fids_old = self.missing_features_stats["id"]

        # Indices of missing feature stats entries to be kept
        inds_keep = [i for i, fid in enumerate(fids_old)
                if fid not in fids_present]

        # Remove entries of features not missing anymore
        lst_old = copy(self._missing_features_stats_lst)
        lst_new = []
        for key, vals_old in lst_old:
            vals_new = [vals_old[i] for i in inds_keep]
            lst_new.append((key, vals_new))
        self._missing_features_stats_lst = lst_new

        # Check total number of features
        n_present = len(self.features())
        n_missing = len(inds_keep)
        n_tot = self.total_track_stats["n"]
        if n_present + n_missing != n_tot:
            err = ("inconsistent number of features: "
                    "{} present + {} missing != {} total"
                    ).format(n_present, n_missing, n_tot)
            raise Exception(err)

    def compute_total_track_stats(self, skip=None):

        if not self.is_complete():
            err = ("total track stats must be stored as long as a track is "
                    "still complete, i.e., before breaking it up into "
                    "partial tracks")
            raise Exception(err)

        keys = [k for k, f in self._total_track_stats_keys_fcts_lst]
        self._total_track_stats_lst = []
        for key, fct in self._total_track_stats_keys_fcts_lst:
            if skip is not None and key in skip:
                self._total_track_stats_lst.append((key, None))
                continue
            self._total_track_stats_lst.append((key, fct(self)))

    def store_missing_features_stats(self, other):

        keys = [k for k, f in self._missing_features_stats_keys_fcts_lst]

        stats_self = self.missing_features_stats
        stats_other = other.missing_features_stats

        if stats_self is None:
            stats_self = odict([(k, []) for k in keys])
        else:
            stats_self = odict([(k, stats_self[k]) for k in keys])

        # Copy missing features from other that are not part of self
        fids = [f.id for f in self.features()]
        if stats_other is not None:
            inds = [i for i, fid in enumerate(
                    stats_other["id"]) if fid not in fids and
                            fid not in stats_self["id"]]
            for key, vals in stats_other.items():
                stats_self[key].extend([vals[i] for i in inds])

        for feature in other.features():
            for key, fct in self._missing_features_stats_keys_fcts_lst:
                stats_self[key].append(fct(feature))

        self._missing_features_stats_lst = [
                (k, v) for k, v in stats_self.items()]

    #--------------------------------------------------------------------------

    def unlink_features(self):
        features = self.graph.vs["feature"]
        self.graph.vs["feature_id"] = [f.id for f in features]
        self.graph.vs["feature"] = None
        for feature in features:
            feature.unset_track()

    def features(self):
        return self.graph.vs["feature"]

    def feature_ids(self):
        return [f.id for f in self.features()]

    def features_ts(self, ts=None, format="list"):

        if self.n == 0:
            if format == "list":
                return []
            elif format == "dict":
                return {}

        if isinstance(ts, datetime):
            ts = self._format_ts(ts, "int")

        if format == "list":
            if ts is None:
                timesteps = self.get_timesteps(total=False)
                return [[vertex["feature"] for vertex in self.vertices_ts(ts)]
                        for ts in timesteps]
            return [vertex["feature"] for vertex in self.vertices_ts(ts)]

        elif format == "dict":
            if ts is None:
                timesteps = self.get_timesteps(total=False)
                return {ts: [vertex["feature"] for vertex
                        in self.vertices_ts(ts)] for ts in timesteps}
            return {ts: vertex["feature"] for vertex in self.vertices_ts(ts)}

        else:
            err = "invalid format: {}".format(format)
            raise ValueError(format)

    def features_n(self, total=True):
        """Get a list of all feature sizes."""
        sizes = [f.n for f in self.features()]
        if not self.is_complete():
            missing_sizes = self.missing_features_stats["n"]
            sizes += missing_sizes
        return sorted(sizes)

    def features_ts_n(self, total=True):
        """Sum of feature sizes for each timestep."""
        return [sum(ns) for ns in self.features_ts_ns(total)]

    def features_ts_ns(self, total=True, cache=False):
        """Feature sizes for each timestep."""
        __name__ = "features_ts_ns"

        if cache and __name__ in self._cache:
            return self._cache[__name__]

        #SR_TMP<
        ## Collect sizes at available timesteps
        #ns_ts =  [[f.n for f in fs] for fs in self.features_ts()]

        #if total and not self.is_complete():

        #    #-- Collect sizes at missing timesteps

        #    missing_tss = self.missing_features_stats["timestep"]
        #    missing_ns = self.missing_features_stats["n"]

        #    # Collect sizes at each timestep
        #    missing_ns_by_ts = {}
        #    for ts, ns in zip(missing_tss, missing_ns):
        #        if ts not in missing_ns_by_ts:
        #            missing_ns_by_ts[ts] = []
        #        missing_ns_by_ts[ts].append(ns)

        #    # Distinguish missing timesteps at the beginning and end
        #    ts_start_nontot = self.ts_start(total=False)
        #    ts_end_nontot = self.ts_end(total=False)
        #    missing_ns_ts_pre, missing_ns_ts_post = [], []
        #    for ts, ns in sorted(missing_ns_by_ts.items()):
        #        if ts < ts_start_nontot:
        #            missing_ns_ts_pre.append(ns)
        #        elif ts > ts_end_nontot:
        #            missing_ns_ts_pre.append(ns)
        #        else:
        #            err = ("track {}: timestep {} should be missing and thus "
        #                    "outside range [{}..{}] but apparently is not"
        #                    ).format(self.id, ts, ts_start_nontot, ts_end_nontot)
        #            raise Exception(err)

        #    # Merge all sizes
        #    ns_ts = missing_ns_ts_pre + ns_ts + missing_ns_ts_post
        #SR_TMP-
        timesteps = self.timesteps(total=total)
        ns_ts = [[] for _ in timesteps]
        for feature in self.features():
            ind = timesteps.index(feature.timestep)
            ns_ts[ind].append(feature.n)
        if total and not self.is_complete():
            for ts, n in zip(
                    self.missing_features_stats["timestep"],
                    self.missing_features_stats["n"]):
                ind = timesteps.index(ts)
                ns_ts[ind].append(n)
        #SR_TMP>

        if cache:
            self._cache[__name__] = ns_ts

        return ns_ts

    def _mean_centers_ts_hull(self, weighted=True, total=True, cache=False):
        """Convex hull around centers."""

        # Determine the mean center at each timestep (weighted by feature size)
        mean_centers = self.mean_centers_ts(weighted=weighted, total=total,
                cache=cache)

        # Determine the enclosing hullgon of the centers (as a convex hull)
        hull = geo.MultiPoint(mean_centers).convex_hull

        return hull

    def mean_centers_ts_hull_len(self, weighted=True, total=True, cache=False):
        """Length of hull around centers."""
        hull = self._mean_centers_ts_hull(weighted=weighted, total=total,
                cache=cache)
        return hull.length

    def mean_centers_ts_hull_area(self, weighted=True, total=True, cache=False):
        """Area of hull around centers."""
        hull = self._mean_centers_ts_hull(weighted=weighted, total=total,
                cache=cache)
        return hull.area

    #SR_TMP<
    def mean_centers_ts_hull_len__per__distance_mean_centers(self):
        kwas = dict(total=True, weighted=True, cache=True)
        hull = self.mean_centers_ts_hull_len(**kwas)
        dist = self.distance_mean_centers(**kwas)
        if dist == 0:
            # Largest hull/dist value: 2.0
            # (ex. 2 points: hull is twice the point distance; there and back)
            return 2.0
        return hull/dist
    def mean_centers_ts_hull__area_per_len(self):
        kwas = dict(total=True, weighted=True, cache=True)
        area = self.mean_centers_ts_hull_area(**kwas)
        len_ = self.mean_centers_ts_hull_len(**kwas)
        if len_ == 0:
            return np.nan
        return area/len_
    def features_n__mean(self):
        return np.mean(self.features_n())
    def features_n__median(self):
        return np.median(self.features_n())
    def features_ts_n__mean(self):
        return np.mean(self.features_ts_n())
    def features_ts_n__median(self):
        return np.median(self.features_ts_n())
    def features_ts_n__p80(self):
        return np.percentile(self.features_ts_n(), 80)
    def features_ts_n__p90(self):
        return np.percentile(self.features_ts_n(), 90)
    def features_ts_n__p90_minus_median(self):
        return self.features_ts_n__p90() - self.features_ts_n__median()
    #SR_TMP>

    def get_edge(self, feature1, feature2):
        return self.graph.es.find(_between=
                ((feature1.vertex().index,), (feature2.vertex().index,)))

    def head_vertices(self):
        """Return all active track head vertices."""
        return self.graph.vs.select(lambda v: v["active_head"])

    def head_features(self):
        """Return all active track head vertices."""
        return self.graph.vs.select(lambda v: v["active_head"])["feature"]

    def duration(self, total=True):
        """Track duration in number of timesteps.

        Argument 'total' is only relevant for partial subtracks; if True,
        missing timesteps (i.e. those that belong to other partial subtracks
        of the same track) are considered, too.
        """
        return len(self.timesteps(total=total)) - 1

    def age(self, ts, total=True):
        """Current age of track in number of timesteps."""
        ts_dt = self._format_ts(ts, format="datetime")
        ts_start_dt = self.ts_start(total=total, format="datetime")
        ts_end_dt = self.ts_end(total=total, format="datetime")
        if ts_dt <= ts_start_dt:
            return 0
        elif ts_dt >= ts_end_dt:
            return self.duration(total=total)
        else:
            timesteps = self.get_timesteps(total=total)
            ts_int = self._format_ts(ts, format="int")
            return timesteps.index(ts_int)

    def _format_ts(self, ts, format):
        ts_fmt = self.config.get("ts_fmt", TS_FMT_DEFAULT)
        if format == "int":
            if isinstance(ts, datetime):
                ts = int(ts.strftime(ts_fmt))
        elif format == "datetime":
            if not isinstance(ts, datetime):
                ts = datetime.strptime(str(ts), ts_fmt)
        else:
            raise ValueError("unknown format '{}'".format(format))
        return ts

    def ts_start(self, total=True, format="int"):
        timesteps = self.get_timesteps(total=total)
        if len(timesteps) > 0:
            return self._format_ts(timesteps[0], format)
        return None

    def ts_end(self, total=True, format="int"):
        timesteps = self.get_timesteps(total=total)
        if len(timesteps) > 0:
            return self._format_ts(timesteps[-1], format)
        return None

    cpdef list finish(self, bint force=False, int split_n=-1,
            bint merge_features=True):
        """Finish track; split into subtracks if requested."""

        # Force-finish track (end of tracking)
        if force:
            for vertex in self.head_vertices():
                type = vertex["type"]
                if type in ("start", "genesis", "merging"):
                    vertex["type"] = "{}/{}".format(type, "stop")
                else:
                    vertex["type"] = "stop"
                vertex["active_head"] = False

        if split_n < 0:
            subtracks = [self]
        else:
            # Split into subtracks
            subtracks = self.split(split_n)

        for subtrack in subtracks:
            # Merge features
            if merge_features and subtrack.n > 1:
                subtrack.merge_features(self.tracker.constants)

        # Store total track stats (for complete tracks)
        for subtrack in subtracks:
            if subtrack.is_complete():
                subtrack.compute_total_track_stats()
            else:
                raise Exception("incomplete track")
                subtrack.store_missing_features_stats(self)
                self.store_missing_features_stats(subtrack)
                subtrack.reset_total_track_stats()

        return subtracks

    cpdef list split(self, int n=0, used_ids=None, bint preserve=False):
        """Split a track into subtracks by eliminating branching.

        At each branching, only the dominant track is continued (highest
        individual successor probability). The others are turned into new
        tracks.

        For n > 0, a check is conducted whether two branches merge during
        the previous/next n timesteps before/after a merging/splitting, in
        which case the branching is retained (for those branches).

        The original track is destroyed unless explicitly stated otherwise.

        Returns all new tracks.
        """
        splitter = FeatureTrackSplitter(used_ids)
        return splitter.split(self, n, preserve=preserve)

    def graph_vs_es_ts(self, start=None, end=None):
        """Return subgraph covering only a certain time window."""

        # Check and convert arguments
        if start is None and end is None:
            return self.graph
        if start is None:
            start = self.ts_start()
        else:
            start = self._format_ts(start, "int")
        if end is None:
            end = self.ts_end()
        else:
            end = self._format_ts(end, "int")

        # Select vertices
        vs = self.graph.vs.select(ts_ge=start, ts_le=end)

        # Select edges
        es = self.graph.es([e.index for e in self.graph.es
                if e.source in vs.indices and e.target in vs.indices])

        return vs, es

    #==========================================================================
    #SR_TMP<

    def values_med_abs_med(self):
        """Compute median of absolute medians of feature values."""
        meds = [np.median(f.values) for f in self.features()]
        if self.missing_features_stats is not None:
            meds += self.missing_features_stats["median"]
        return np.median(np.abs(meds))

    def values_mean_abs_mean(self):
        """Compute mean of absolute means of feature values."""
        means = [np.mean(f.values) for f in self.features()]
        if self.missing_features_stats is not None:
            means += self.missing_features_stats["mean"]
        return np.mean(np.abs(means))

    def values_med_abs_mean(self):
        """Compute median of absolute means of feature values."""
        means = [np.mean(f.values) for f in self.features()]
        if self.missing_features_stats is not None:
            means += self.missing_features_stats["mean"]
        return np.median(np.abs(means))

    def values_med_mean_abs(self):
        """Compute median of means of absolute feature values."""
        means = [np.abs(f.values).mean() for f in self.features()]
        if self.missing_features_stats is not None:
            means += self.missing_features_stats["absmean"]
        return np.median(means)

    def values_med_med_abs(self):
        """Compute median of medians of absolute feature values."""
        means = [np.median(np.abs(f.values)) for f in self.features()]
        if self.missing_features_stats is not None:
            means += self.missing_features_stats["absmed"]
        return np.median(means)

    #--------------------------------------------------------------------------

    def is_associated(self, nmin, *, **kwas):
        na = self.n_associated(**kwas)
        return na >= min([nmin, self.n])

    def is_associated_ts(self, nmin, *, **kwas):
        na = self.n_associated_ts(**kwas)
        return na >= min([nmin, self.n])

    def n_associated(self, *, all_=None, any_=None, total=True, cache=False):
        """Count features with certain associates."""
        return self._n_associated_core(all_, any_, total=total, ts=False,
                cache=cache)

    def n_associated_ts(self, *, all_=None, any_=None, total=True, cache=False):
        """Count timesteps with features with certain associates."""
        return self._n_associated_core(all_, any_, total=total, ts=True,
                cache=cache)

    def _n_associated_core(self, all_, any_, total, ts, cache):

        key = ("n_associated__ts_{ts}__all_{all}__any_{any}__total_{tot}"
                ).format(ts=str(ts), tot=str(total),
                any=("" if all_ is None else "_".join(sorted(all_))),
                all=("" if any_ is None else "_".join(sorted(any_))))
        if cache and key in self._cache:
            return self._cache[key]

        # Check arguments
        if all_ is None and any_ is None:
            err = "must pass list of allociate names to 'all_' or 'any_'"
            raise ValueError(err)
        elif all_ is not None and any_ is not None:
            err = "cannot pass both 'all_' and 'any_'"
            raise ValueError(err)
        elif (any_ is not None and isinstance(any_, str) or
                all_ is not None and isinstance(all_, str)):
            err = "must pass associate names in list (or the like)"
            raise ValueError(err)

        # Count associates
        n_tot = 0
        for features_ts in self.features_ts():
            n_ts = 0
            for feature in features_ts:
                if all_ is not None and all(
                        len(feature.associates.get(i, [])) > 0 for i in all_):
                    n_ts += 1
                if any_ is not None and any(
                        len(feature.associates.get(i, [])) > 0 for i in any_):
                    n_ts += 1
            if ts:
                n_tot += (1 if n_ts > 0 else 0)
            else:
                n_tot += n_ts

        # Add info from missing features (for partial tracks)
        if total and self.missing_features_stats is not None:
            missing_info_ts = {}
            for timestep, associates_n in zip(
                    self.missing_features_stats["timestep"],
                    self.missing_features_stats["associates_n"]):
                if timestep not in missing_info_ts:
                    missing_info_ts[timestep] = []
                missing_info_ts[timestep].append(
                        {name: n for name, n in associates_n})
            for timestep, features_info in missing_info_ts.items():
                n_ts = 0
                for asscs in features_info:
                    if all_ is not None and all(
                            asscs.get(i, 0) > 0 for i in all_):
                        n_ts += 1
                    if any_ is not None and any(
                            asscs.get(i, 0) > 0 for i in any_):
                        n_ts += 1
                if ts:
                    n_tot += (1 if n_ts > 0 else 0)
                else:
                    n_tot += n_ts

        if cache:
            self._cache[key] = n_tot

        return n_tot

    #==========================================================================

    def centers_ts(self, total=True, cache=False):
        """Return all centers, grouped by timestep."""
        __name__ = "centers_ts"

        if cache and __name__ in self._cache:
            return self._cache[__name__]

        #SR_TMP<
        #centers_by_ts = {}
        #for feature in self.features():
        #    ts = feature.timestep
        #    if ts not in centers_by_ts:
        #        centers_by_ts[ts] = []
        #    centers_by_ts[ts].append(feature.center)

        #if total and not self.is_complete():
        #    for ts in set(self.missing_features_stats["timestep"]):
        #        if ts in centers_by_ts:
        #            err = "timestep both present and missing: {}".format(ts)
        #            raise Exception(err)
        #        centers_by_ts[ts] = []
        #    for ts, center in zip(
        #            self.missing_features_stats["timestep"],
        #            self.missing_features_stats["center"]):
        #        centers_by_ts[ts].append(feature.center)

        #    centers = list(odict(sorted(centers_by_ts.items())).values())
        #SR_TMP-
        timesteps = self.timesteps(total=total)
        centers = [[] for _ in timesteps]
        for feature in self.features():
            ind = timesteps.index(feature.timestep)
            centers[ind].append(feature.center)
        if total and not self.is_complete():
            for ts, center in zip(
                    self.missing_features_stats["timestep"],
                    self.missing_features_stats["center"]):
                ind = timesteps.index(ts)
                centers[ind].append(center)
        #SR_TMP>

        if cache:
            self._cache[__name__] = centers

        return centers

    def mean_centers_ts(self, weighted=True, total=True, cache=False):
        """Return the mean center at each timestep.

        Optionally, the averaging of multiple centers at a given timestep
        is weighted by the sizes of the respective features.
        """
        __name__ = "mean_centers_ts"

        if cache and __name__ in self._cache:
            return self._cache[__name__]

        mean_centers = []
        for centers, sizes in zip(
                self.centers_ts(total=total),
                self.features_ts_ns(total=total),
            ):

            if len(centers) != len(sizes):
                err = ("different number of centers ({}) and sizes ({})"
                        ).format(len(centers), len(sizes))
                err += "\n\n{}\n{}\n\n{}\n{}".format(
                        self.centers_ts(total=total),
                        len(self.centers_ts(total=total)),
                        self.features_ts_ns(total=total),
                        len(self.features_ts_ns(total=total)),
                    )
                err += "\n\n{}\n{}".format(
                        sorted([len(i) for i in self.centers_ts(total=total)]),
                        sorted([len(i) for i in self.features_ts_ns(total=total)]),
                    )
                raise Exception(err)

            # Compute mean center
            kwas = dict(axis=0)
            if weighted:
                # Weigh center positions by feature sizes
                # The mean center is thus closer to bigger features
                kwas["weights"] = sizes
            mean_center = np.average(centers, **kwas).astype(np.float32)

            mean_centers.append(mean_center)

        if cache:
            self._cache[__name__] = mean_centers

        return mean_centers

    def distance_mean_centers(self, N=1, weighted=True, total=True, cache=False):
        """Total distance along mean center at every timestep."""

        # Collect path along mean centers
        path = self.mean_centers_ts(total=total, weighted=weighted, cache=cache)

        if N > 1:
            # Smooth the path
            path = self._smooth_path(path, N)

        # Compute path length
        if len(path) == 1:
            length = 0.0
        else:
            length = geo.LineString(path).length

        return length

    def _smooth_path(self, path, N):
        if not N > 0 and N%2 == 1:
            raise ValueError("N not a positive uneven integer: ".format(N))

        npts = path.shape[1]
        if npts <= 3 or N == 1:
            return path

        #SR_TMP< TODO convert format (see comment below)
        assert path.shape[0] == 2
        #SR_TMP>

        # Adjust window size for small paths
        while N > npts:
            N -= 2

        # Compute running means for axes of sufficient length
        win = np.ones(N)/float(N) # window
        path = np.array([
                np.convolve(path[0], win, mode="valid"),
                np.convolve(path[1], win, mode="valid")
            ]).round().astype(np.int32)

        return path

    #==========================================================================

    def velocity__mean_centers(self):
        """Velocity based on mean center distances.

        unit: grid points per timestep
        """
        #SR_TMP<
        weighted=True
        total=True
        cache = True
        #SR_TMP>

        dist = self.distance_mean_centers(weighted=weighted, total=total,
                cache=cache)
        duration = float(self.duration(total=total))

        if duration == 0:
            return 0
        return dist/duration

    #==========================================================================

    def footprint(self, nx=None, ny=None, *,
            ts_start=None, ts_end=None, timesteps=None, flatten=True,
            cache=False, cache_feature_masks=False, __warned=[False],
        ):
        """Compute total footprint of track as 2D mask."""
        __name__ = "footprint"
        _tss_str = ("None" if timesteps is None else
                        "-".join(str(ts) for ts in timesteps))
        __key__ = ("{}"+"__{}"*5).format(
                nx, ny, ts_start, ts_end, _tss_str, flatten)

        if nx is None or ny is None:
            if self.tracker is not None:
                nx = self.tracker.constants.nx
                ny = self.tracker.constants.ny
                #SR_DBG<
                if __warned[0] != 1:
                    method_name = "Feature.footprint"
                    log.warning(("{}: dimensions from tracker: nx={}, ny={} !!"
                            ).format(method_name, nx, ny))
                    __warned[0] = 1 # warn only once
                #SR_DBG>
            else:
                nx, ny = [1542]*2
                if __warned[0] != 2:
                    log.warning("{}: hard-coded: nx={}, ny={} !!".format(
                            "Feature.footprint", nx, ny))
                    __warned[0] = 2 # warn only once

        if timesteps is None:
            timesteps = self.timesteps(ts_start=ts_start, ts_end=ts_end)
        elif ts_start is not None or ts_end is not None:
            raise ValueError("mutually exclusive: timesteps, ts_start/ts_end")

        if not self.is_complete():
            timesteps_notot = self.timesteps(total=False)
            if not all(ts in timesteps_notot for ts in timesteps):
                err = ("partial track {} missing timesteps: not all of {} in {}"
                        ).format(self.id, timesteps, timesteps_notot)
                raise Exception(err)

        if cache and __name__ in self._cache:
            return self._cache[__key__]

        fld = np.zeros([nx, ny], np.int32)
        for feature in self.features():
            if feature.timestep in timesteps:
                mask = feature.to_mask(nx, ny, cache=cache_feature_masks)
                fld[mask] += 1

        if flatten:
            fld[fld > 0] = 1

        if cache:
            self._cache[__key__] = fld

        return fld

    def to_mask(self, nx=None, ny=None, **kwas):
        """Create track mask over whole duration or select timesteps.

        See method 'footprint' for details on options.
        """
        footprint = self.footprint(nx=nx, ny=ny, flatten=True, **kwas)
        return footprint.astype(np.bool)

    #SR_TMP<
    def max_feature_count(self):
        return self.footprint(flatten=False).max()
    #SR_TMP>

    def footprint_n(self, **kwas):
        """Compute size of total footprint of track."""

        if self.is_complete():
            # Case 1: Complete track: compute footprint
            fld_tot = self.footprint(**kwas)
            fp_tot = fld_tot.sum()
        else:
            # Incomplete track: Read precomputed value from total track stats
            if self.total_track_stats is None:
                err = ("track {}: "
                        "incomplete, yet lacking missing track stats "
                        "(self.total_track_stats is None); "
                        "cannot compute footprint size!").format(self.id)
                raise Exception(err)
            fp_tot = self.total_track_stats["footprint_n"]
            if fp_tot is None:
                err = ("track {}: footprint_n of total track stats is None"
                        ).format(self.id)
                raise Exception(err)

        #SR_TODO
        #
        # Account for case where only a subset of timesteps is requested
        # and all are part of the complete part of the track, in which case
        # the footprint can be computed even though a track is incomplete!
        #

        return fp_tot

    #SR_TMP<
    def footprint_n__log10(self):
        n = self.footprint_n(cache=True)
        if n <= 0:
            err = "track {}: invalid footprint_n <= 0: {}".format(self.id, n)
            raise Exception(err)
        return np.log10(n)
    #SR_TMP>

    def footprint_n_rel(self, op="median", total=True, cache=False):
        """Compute track footprint relative to median/... feature size."""
        __name__ = "footprint_n_rel__"+op

        if cache and __name__ in self._cache:
            return self._cache[__name__]

        # Compute the reference value (mean or median)
        if op.endswith("_ts"):
            sizes = self.features_ts_n(total)
        else:
            sizes = self.features_n(total)
        if op.startswith("median"):
            fp_ref = np.median(sizes)
        elif op.startswith("mean"):
            fp_ref = np.mean(sizes)
        elif op.startswith("max"):
            fp_ref = np.max(sizes)
        elif op.startswith("percentile_"):
            pctl = float(op.split("_")[1])
            fp_ref = np.percentile(sizes, pctl)
        else:
            raise ValueError("invalid operator: {}".format(op))

        # Compute relative footprint
        fp_tot = self.footprint_n(cache=cache)
        fp_rel = float(fp_tot)/fp_ref

        if cache:
            self._cache[__name__] = fp_rel

        return fp_rel

    #SR_TMP<
    def footprint_n_rel__median(self):
        return self.footprint_n_rel(op="median", cache=True)
    def footprint_n_rel__p80(self):
        return self.footprint_n_rel(op="percentile_80", cache=True)
    def footprint_n_rel__mean(self):
        return self.footprint_n_rel(op="mean", cache=True)
    def footprint_n_rel__max(self):
        return self.footprint_n_rel(op="max", cache=True)
    def footprint_n_ts_rel__median(self):
        return self.footprint_n_rel(op="median_ts", cache=True)
    def footprint_n_ts_rel__p80(self):
        return self.footprint_n_rel(op="percentile_80_ts", cache=True)
    def footprint_n_ts_rel__mean(self):
        return self.footprint_n_rel(op="mean_ts", cache=True)
    def footprint_n_ts_rel__max(self):
        return self.footprint_n_rel(op="max_ts", cache=True)
    #SR_TMP>

    #==========================================================================

    cpdef void merge_features(self, Constants constants) except *:
        """Merge all adjacent features at each timestep."""
        #log.info("track {}: merge neighbors among {} features".format(
        #        self.id, self.n))
        nold = self.n
        TrackFeatureMerger(self, constants.nx, constants.ny,
                constants.connectivity, timestep=self.ts_end()).run()
        if self.n < nold:
            log.info("track {}: merged features: {} -> {}".format(
                    self.id, nold, self.n))

    @classmethod
    def from_old_track(cls, old_track, new_features):
        """Create a track from an old-style track.

        Note that the features must already have been converted to new-style
        and passed as a dictionary with the feature IDs as keys.
        """
        graph = ig.Graph(directed=True)

        new_features_table = {f.id: f for f in new_features}

        # Idea: Add a class for vertices (basically like Events) which
        # contains variables for both a feature and a corresponding cfeature.
        # When working in Python/Cython, link features accordingly and leave
        # the other NULL/None. When crossing the Python/Cython interface,
        # check for NULL/None and convert accordingly.
        #
        # Eventually, though, just turn Feature into an extension type
        # and internalize the whole cfeature thing (if that's feasible
        # performance-wise).
        #

        # Add edges for start features
        ts_start = old_track.ts_start()
        events = old_track.events_ts(ts_start)
        for event in events:
            fid = event.feature().id()
            feature = new_features_table[fid]
            vertex = track_graph_add_feature(graph, feature)

        # Follow track and build graph with features
        iter_max = 1000
        for iter_i in range(iter_max):
            next_events = []
            for event in events:
                fid = event.feature().id()
                feature = new_features_table[fid]
                vertex = feature.vertex()
                for next_event in event.next():
                    next_fid = next_event.feature().id()
                    if next_event in next_events:
                        next_vertex = next_event.feature().vertex()
                    else:
                        next_feature = new_features_table[next_fid]
                        next_events.append(next_event)
                        next_vertex = track_graph_add_feature(graph, next_feature)
                    #SR_TMP< SR_DIRECTED
                    if vertex["ts"] > next_vertex["ts"]:
                        edge = track_graph_add_edge(graph, next_vertex, vertex)
                    else:
                    #SR_TMP>
                        edge = track_graph_add_edge(graph, vertex, next_vertex)

            if not next_events:
                break
            events = next_events
        else:
            log.error("loop timeout ({} iterations)".format(iter_i))
            exit(2)

        return cls(
                id           = old_track.id(),
                graph        = graph,
                features     = new_features,
            )

    def json_dict(self, format=True):
        """Return track information as a json-compatible dict."""
        timesteps = self.get_timesteps(total=False)
        feature_ids = [f.id for f in self.features()]
        jdat = odict([
                ("id", self.id),
                ("timesteps", timesteps),
                ("features", feature_ids),
                ("is_complete", self.is_complete()),
            ])
        jdat["stats"] = odict([
                ("n_features", self.n),
                ("max_length", self.max_length()),
                ("duration", self.duration()),
            ])
        if not self.is_complete():
            if self.total_track_stats is None:
                err = ("track {} is incomplete but lacking total track stats "
                        "(self.total_track_stats is None)").format(self.id)
                raise Exception(err)
            if self.missing_features_stats is None:
                err = ("track {} is incomplete but lacking missing features "
                        "stats (self.missing_features_stats is None)").format(
                        self.id)
                raise Exception(err)
            jdat["total_track_stats"] = self.total_track_stats
            jdat["missing_features_stats"] = self.missing_features_stats
        return jdat

    @property
    def n(self):
        return len(self.features())

    def n_holes(self, op="sum", total=True):
        if op == "sum":
            reduce = np.sum
        elif op == "mean":
            reduce = np.mean
        elif op == "median":
            reduce = np.median
        else:
            raise ValueError("invalid operator: {}".format(op))
        n_holes_features = [len(f.holes) for f in self.features()]
        if not self.is_complete():
            try:
                n_holes_features.extend(
                        self.missing_features_stats["n_holes"])
            except KeyError:
                err = ("track {}: holes missing from missing features stats"
                        ).format(self.id)
                raise Exception(err)
        return reduce(n_holes_features)

    #SR_TMP<
    def n_holes_sum(self, total=True):
        return self.n_holes("sum", total)
    def n_holes_mean(self, total=True):
        return self.n_holes("mean", total)
    def n_holes_median(self, total=True):
        return self.n_holes("median", total)
    #SR_TMP>

    def max_length(self):
        return -1

    def get_timesteps(self, total=True, format="int"):
        """Return timesteps.

        Argument total is only relevant for partial subtracks; if True,
        missing timesteps (i.e. those that belong to other subtracks of the
        same track) are considered, too.
        """
        timesteps = set(self.graph.vs["ts"])
        if not self.is_complete() and total:
            if self.missing_features_stats is None:
                err = ("track {}: incomplete but no missing_features_stats"
                        ).format(self.id)
                raise Exception(err)
            timesteps |= set(self.missing_features_stats["timestep"])
        timesteps = sorted(timesteps)
        if (format == "datetime" and not isinstance(timesteps[0], datetime) or
                format == "int" and not isinstance(timesteps[0], int)):
            timesteps = [self._format_ts(ts, format) for ts in timesteps]
        return timesteps

    def vertices_ts(self, ts=None):
        """Return all vertices at a given timestep."""
        #SR_TMP<
        if isinstance(ts, datetime):
            ts = self._format_ts(ts, "int")
        #SR_TMP>
        if ts is None:
            return [list(self.vertices_ts(ts))
                    for ts in self.get_timesteps(total=False)]
        return self.graph.vs.select(lambda v: v["ts"] == ts)

    #==========================================================================

    def cut_off(self, *, until, compute_footprint=True):
        """Cut off part of the track in order to save it to disk.

        The respective vertices are removed from the original track.
        Information about the lost edges is retained in both partial tracks.

        Be aware that the returned subtrack still has the same id,
        i.e. it is not save to work with the resulting partial tracks!

        """
        if not self.is_finished():
            err = "track {}: cannot cut part off active track".format(self.id)
            raise Exception(err)

        if not self.ts_start() <= until < self.ts_end():
            err = ("invalid argument until (until must be in track range {}.."
                    "{}): {}").format(self.ts_start(), self.ts_end(), until)
            raise ValueError(err)

        if self.is_complete():
            _skip = ["footprint_n"] if not compute_footprint else None
            self.compute_total_track_stats(skip=_skip)
        elif self.total_track_stats is None:
            err = "track {}: incomplete, yet lacking total track stats"
            raise Exception(err)

        graph_retain = self.graph
        ts_cutoff_edge = until
        _tss_retain = [ts for ts in self.graph.vs["ts"] if ts > until]
        if not _tss_retain:
            err = ("track {}: full range ({}..{}); non-total range ({}..{}); "
                    "no timesteps > {} found in graph").format(self.id,
                    self.ts_start(total=True), self.ts_end(total=True),
                    self.ts_start(total=False), self.ts_end(total=False))
            raise Exception(err)
        ts_retain_edge = min(_tss_retain)

        # Select vertices and edges involved in the splitting
        vs_cutoff = graph_retain.vs.select(ts_le=ts_cutoff_edge)
        vs_retain = graph_retain.vs.select(ts_gt=ts_cutoff_edge)
        vs_cutoff_edge = vs_cutoff.select(ts_eq=ts_cutoff_edge)
        vs_retain_edge = vs_cutoff.select(ts_eq=ts_retain_edge)

        # Collect attributes of edges about to be eliminated
        # and store it on both adjacent vertices (source and target)
        es_remove = graph_retain.es.select(_source_in=vs_cutoff_edge)
        attr_names = es_remove.attribute_names()
        attrs = {key: es_remove[key] for key in attr_names}
        for edge in es_remove:
            vx_source = graph_retain.vs[edge.source]
            vx_target = graph_retain.vs[edge.target]
            fid_source = vx_source["feature"].id
            fid_target = vx_target["feature"].id

            if vx_source["missing_successors"] is None:
                vx_source["missing_successors"] = {}
            vx_source["missing_successors"][fid_target] = edge.attributes()

            if vx_target["missing_predecessors"] is None:
                vx_target["missing_predecessors"] = {}
            vx_target["missing_predecessors"][fid_source] = edge.attributes()

        # Extract partial track into separate track
        graph_cutoff = graph_retain.subgraph(vs_cutoff)
        graph_retain.delete_vertices(vs_cutoff.indices)
        partial_track = FeatureTrack(id_=self.id, graph=graph_cutoff,
                config=self.config)

        # Store missing track and features stats
        _skip = ["footprint_n"] if not compute_footprint else None
        partial_track.set_total_track_stats(
                deepcopy(self.total_track_stats), skip=_skip)
        self.store_missing_features_stats(partial_track)
        partial_track.store_missing_features_stats(self)

        return partial_track

    @classmethod
    def merge_partial_tracks(cls, subtracks, *,
            incomplete_period_start=None, incomplete_period_end=None):
        """Merge partial tracks back into a single track."""

        if len(subtracks) == 0:
            raise ValueError("no subtracks to merge")

        elif len(subtracks) == 1:
            _subtrack = subtracks[0]
            _ts_start = _subtrack.ts_start(total=False)
            _ts_end = _subtrack.ts_end(total=False)

            # Only one subtrack: nothing to merge
            if not (_subtrack.is_complete() or
                    _ts_start == incomplete_period_start or
                    _ts_end == incomplete_period_end
                ):
                # Single incomplete track that is not at the start or end
                # of an incomplete period: we've got ourselves an error!
                err = ("only single, incomplete track passed: {} "
                        "({}..{})").format(_subtrack.id, _ts_start, _ts_end)
                raise ValueError(err)

            return _subtrack

        # Check that all subtracks share the same track id
        subtracks_ids_str = sorted([str(st.id) for st in subtracks])
        if len(set(subtracks_ids_str)) != 1:
            err = "subtracks expected to have the same track id: {}".format(
                    ", ".join(subtracks_ids_str))

        # Check that all total track stats are identical
        for subtrack in subtracks[1:]:
            if subtrack.total_track_stats != subtracks[0].total_track_stats:
                err = ("tracks {} v. {}: total track stats differ:\n{}\n !=\n{}"
                        ).format(subtrack.id, subtracks[0].id,
                                pformat(subtrack.total_track_stats),
                                pformat(subtracks[0].total_track_stats),
                            )
                raise Exception(err)

        # Merge subtracks
        subtracks.sort(key=lambda t: t.ts_start())
        track = merge_tracks(subtracks)

        # Restore missing edges between former subtracks
        vs_source = list(track.graph.vs.select(missing_successors_ne=None))
        vs_target = list(track.graph.vs.select(missing_predecessors_ne=None))
        key_succ, key_pred = "missing_successors", "missing_predecessors"
        for vx_source in [v for v in vs_source]:
            if vx_source["feature"] is not None:
                fid_source_vx = vx_source["feature"].id
            else:
                fid_source_vx = vx_source["feature_id"]

            for vx_target in [v for v in vs_target]:
                if vx_target["feature"] is not None:
                    fid_target_vx = vx_target["feature"].id
                else:
                    fid_target_vx = vx_target["feature_id"]
                if fid_target_vx not in vx_source[key_succ]:
                    continue

                edge_attrs = vx_source[key_succ][fid_target_vx]
                track.graph.add_edge(vx_source, vx_target, **edge_attrs)

                del vx_target[key_pred][fid_source_vx]
                if len(vx_target[key_pred]) == 0:
                    vx_target[key_pred] = None
                    vs_target.remove(vx_target)

                del vx_source[key_succ][fid_target_vx]
                if len(vx_source[key_succ]) == 0:
                    vx_source[key_succ] = None
                    vs_source.remove(vx_source)
                    break

        # Remove any leftover vertices at start and end of incomplete period
        if incomplete_period_start:
            vs_target = [v for v in vs_target
                    if v["ts"] != incomplete_period_start]
        if incomplete_period_end:
            vs_source = [v for v in vs_source
                    if v["ts"] != incomplete_period_end]

        if vs_source or vs_target:
            # Ooop, leftover source/target vertices... That's an error!
            fids_source = [vx["feature"].id for vx in vs_source]
            fids_target = [vx["feature"].id for vx in vs_target]
            ntot = len(fids_source) + len(fids_target)
            err = " {} left-over {}:".format(ntot,
                    ("vertices" if ntot > 1 else "vertex"))
            if len(fids_target) > 0:
                err += "\n {} target {} (beginning): {} [{}]".format(
                        len(fids_target),
                        ("vertices" if ntot > 1 else "vertex"),
                        ", ".join([str(fid) for fid in fids_target]),
                        "/".join([str(vx["ts"]) for vx in vs_target]),
                    )
                raise Exception(err)
            if len(fids_source) > 0:
                err += "\n {} source {} (end): {} [{}]".format(
                        len(fids_source),
                        ("vertices" if ntot > 1 else "vertex"),
                        ", ".join([str(fid) for fid in fids_source]),
                        "/".join([str(vx["ts"]) for vx in vs_source]),
                    )
                raise Exception(err)

        # Check track start
        if (incomplete_period_start is not None and
                track.ts_start(total=False) != incomplete_period_start):
            if (min(track.timesteps(total=False)) !=
                    min(track.timesteps(total=True))):
                err = "start of track {} still incomplete: {} != {}".format(
                        track.id, track.ts_start(total=False),
                        incomplete_period_start)
                raise Exception(err)

        # Check track end
        if (incomplete_period_end is not None and
                track.ts_end(total=False) != incomplete_period_end):
            if (max(track.timesteps(total=False)) !=
                    max(track.timesteps(total=True))):
                err = ("end of track {} still incomplete: {} != {}").format(
                        track.id, track.ts_end(total=False),
                        incomplete_period_end)
                raise Exception(err)

        # Update missing features stats
        track.update_missing_features_stats()

        return track

#------------------------------------------------------------------------------

def track_graph_add_feature(graph, feature, attrs=None):
    if attrs is None:
        attrs = {}
    name = str(feature.id)
    #print("add_vertex {}: {}".format(name, feature.id))

    if name in graph.vs.select(name):
        err = "{}: feature {} already in graph".format("grap_add_vertex", name)
        log.error(err)
        exit(4)

    graph.add_vertex(name)
    vertex = next(iter(graph.vs(name=name)))
    vertex.update_attributes(
            feature = feature,
            ts = feature.timestep,
            **attrs
        )
    feature.set_vertex(vertex)

    return vertex

def track_graph_add_edge(graph, vertex1, vertex2, attrs=None):
    #print("add edge: {} -> {}".format(fid1, fid2))

    if vertex1["ts"] == vertex2["ts"]:
        err = "cannot add edge between vertices of same timestep ({})".format(
                vertex1["ts"])
        raise ValueError(err)

    if vertex1["ts"] > vertex2["ts"]:
        err = ("invalid vertex pair: must be in chronological order "
                "(i.e. timestep of first vertex smaller than of the second): "
                " {}, {}").format(vertex1["ts"], vertex2["ts"])
        raise ValueError(err)

    if attrs is None:
        attrs = {}

    graph.add_edge(vertex1, vertex2, **attrs)

    return graph.es[-1]

#==============================================================================

def remerge_partial_tracks(subtracks, counter=False, is_subperiod=False):
    """Reconstruct partial tracks.

    After reading tracks from consecutive track files (no matter how many),
    collect all tracks in a single list and pass it to his function.

    Tracks which share the same id will be merged.
    """
    if not subtracks:
        return []
        #raise ValueError("must pass subtracks")

    tracks_mended = []

    # Determine time period
    _tss = set([ts for track in subtracks
            for ts in track.timesteps(total=False)])
    ts_start, ts_end = min(_tss), max(_tss)

    # Collect subtracks by id
    subtracks_grouped = {}
    for subtrack in subtracks:
        if subtrack.id not in subtracks_grouped:
            subtracks_grouped[subtrack.id] = []
        subtracks_grouped[subtrack.id].append(subtrack)

    # Merge subtracks that share the same id
    n = len(subtracks_grouped)
    for i, (tid, subtracks_i) in enumerate(sorted(subtracks_grouped.items())):
        if counter:
            frac = i/n
            msg = " {: 4.1%} {} ({} subtracks)".format(i/n, subtracks_i[0].id,
                    len(subtracks_i))
            try:
                w, h = os.get_terminal_size()
            except OSError:
                pass
            else:
                print(msg[:w], end="\r", flush=True)

        #print("reconstruct track {} from {} subtracks".format(
        #        next(iter(subtracks_i)).id), len(subtracks_i))
        kwas = dict()
        if is_subperiod:
            kwas["incomplete_period_start"] = ts_start
            kwas["incomplete_period_end"] = ts_end
        subtrack = FeatureTrack.merge_partial_tracks(subtracks_i, **kwas)

        tracks_mended.append(subtrack)

    if counter:
        try:
            w, h = os.get_terminal_size()
        except OSError:
            pass
        else:
            print(" "*w, end="\r", flush=True)

    return tracks_mended

#==============================================================================

cdef class TrackFeatureMerger:

    def __cinit__(self, FeatureTrack track, int nx, int ny, int connectivity,
            np.uint64_t timestep=0):
        self.track = track
        self.nx = nx
        self.ny = ny
        self.connectivity = connectivity
        self.timestep = timestep

        self.vs_attrs = FeatureTrack.vs_attrs
        self.es_attrs = FeatureTrack.es_attrs

        self.debug = False
        if self.debug: log.debug("\nmerge features in track {} (n={})".format(track.id, track.n))

    cpdef void run(self) except *:
        cdef bint debug = self.debug
        cdef str _name_ = "TrackFeatureMerger.run"

        # Select all features with neighbors belonging to the same track
        cdef list all_features = self.track.features();
        cdef Feature feature, neighbor
        cdef list features_todo = []
        if debug: log.debug("track {}: find features to merge".format(self.track.id))
        for i, feature in enumerate(self.track.features()):
            for j, neighbor in enumerate(feature.neighbors):
                if feature.track().id == neighbor.track().id:
                    if debug: log.debug(" -> feature {} ({}/{}) due to neighbor {} ({}/{})".format(feature.id, i+1, self.track.n, neighbor.id, j, len(feature.neighbors)))
                    features_todo.append(feature)
                    break

        # Early exit if nothing to do
        if len(features_todo) == 0:
            return

        if debug: log.debug("\n ++++ MERGER ++++ TRACK {}\n".format(self.track.id))

        # Check that we're not missing any vertex or edge attributes
        if len(self.track.graph.vs) > 0:
            graph_vs_attrs = self.track.graph.vs.attributes()
            if not all([a in graph_vs_attrs for a in self.vs_attrs]):
                err = ("{}: track {}: graph vertex attributes don't match:\n"
                        "expected:\n    {}\nfound:\n    {}"
                        ).format(_name_, self.track.id,
                        "\n    ".join([str(a) for a in sorted(self.vs_attrs)]),
                        "\n    ".join([str(a) for a in sorted(graph_vs_attrs)]))
                log.error(err)
                exit(5)
        if (len(self.track.graph.es) > 0 and
                (set(self.track.graph.es.attributes()) != self.es_attrs)):
            es = self.track.graph.es
            err = ("{}: track {}: attributes of {} graph edges don't match:\n"
                    "expected:\n    {}\nfound:\n    {}").format(
                    _name_, self.track.id, len(es),
                    "\n    ".join([str(a) for a in sorted(self.es_attrs)]),
                    "\n    ".join([str(a) for a in sorted(es.attributes())]))
            raise Exception(err)

        # Process these features until none are left
        while len(features_todo) > 0:
            self.merge_feature(all_features, features_todo)

    #--------------------------------------------------------------------------

    cpdef void merge_feature(self, list features, list features_todo) except *:
        cdef bint debug = self.debug
        cdef str _name_ = "TrackFeatureMerger.merge_feature"
        if debug: log.debug("< {}: {} todo: {}".format(_name_, len(features_todo), [f.id for f in features_todo]))

        cdef Feature feature, neighbor, other_neighbor
        cdef list features_orig, external_neighbors

        # Collect all adjacent features to be merged, and collect all
        # external neighbors of these features
        feature = features_todo.pop()
        if debug: log.debug("feature: {}; {} neighbors: {}".format(feature.id, len(feature.neighbors), [f.id for f in feature.neighbors]))
        features_orig = [feature]
        external_neighbors = []
        self.collect_neighbors(feature, features_todo, features_orig,
                external_neighbors)
        cdef np.uint64_t new_fid = min([feature.id for feature in features_orig])
        feature = None

        # prevent "warning: features_to_cregions: cannot create cregion:
        #          feature.cregion not NULL"
        for feature in features_orig:
            feature.cleanup_cregion()

        # Merge the original features into one merged feature
        cdef Feature merged_feature
        cdef list merged_features
        merged_features = merge_adjacent_features(features_orig, new_fid,
                find_neighbors=False, ignore_missing_neighbors=True,
                nx=self.nx, ny=self.ny)
        if len(merged_features) > 1:
            log.error(("{}: merging adjacent features results in more than "
                    "one features; shouldn't happen!").format(_name_))
            exit(2)
        merged_feature = merged_features[0]
        assert(merged_feature.id == new_fid) #SR_TMP SR_DBG
        merged_feature.set_track(self.track)

        # Link the merged feature to all external neighbors
        for neighbor in external_neighbors:
            merged_feature.neighbors.append(neighbor)
            neighbor.neighbors.append(merged_feature)

        # Collect vertices and edges
        cdef list vs_orig=[], vs_prev=[], vs_next=[], es_prev=[], es_next=[]
        self.collect_vertices_edges(features_orig,
                vs_orig, vs_prev, vs_next, es_prev, es_next)

        # Collect and merge original vertex attributes
        cdef dict vx_attrs = {}
        self.collect_merge_vs_attrs(vx_attrs, vs_orig, merged_feature)

        # Collect and merge original edge attributes
        cdef dict es_attrs_prev={}, es_attrs_next={}
        self.collect_merge_es_attrs(es_attrs_prev, es_attrs_next, vs_orig,
                vs_prev, vs_next, es_prev, es_next, merged_feature)

        # Replace old vertices and edges
        self.replace_vertices_edges(merged_feature, features_orig, vs_orig,
                vs_prev, vs_next, es_prev, es_next, vx_attrs,es_attrs_prev,
                es_attrs_next)

    #--------------------------------------------------------------------------

    cpdef void collect_neighbors(self, Feature feature, list features_todo,
            list features_orig, list external_neighbors) except *:
        cdef str _name_ = "TrackFeatureMerger.collect_neighbors"
        cdef bint debug = self.debug

        cdef Feature neighbor, other_neighbor
        for neighbor in feature.neighbors.copy():
            if debug: log.debug(" -> neighbor: {}; {} neighbors: {}".format(neighbor.id, len(neighbor.neighbors), [f.id for f in neighbor.neighbors]))
            if feature in neighbor.neighbors:
                if debug: log.debug("feature {}: remove neighbor {}".format(feature.id, neighbor.id))
                neighbor.neighbors.remove(feature)
            else:
                err = "{}: feature {} missing in neighbors of it's neighbor {}".format(_name_, feature.id, neighbor.id)
                raise Exception(err)

            # Store neighbors that belong to other tracks separately
            if neighbor.track().id != feature.track().id:
                if debug: log.debug("feature {}: remove neighbor {}".format(feature.id, neighbor.id))
                feature.neighbors.remove(neighbor)
                external_neighbors.append(neighbor)
                continue

            if neighbor not in features_todo:
                log_timestep = "?" if self.timestep is None else self.timestep
                log.warning(("[{}] warning: track {}: neighbor {} not in "
                        "features_todo (track size: {}; todo: {})").format(
                        log_timestep, feature.track().id, neighbor.id,
                        self.track.n, len(features_todo)))
                continue

            features_todo.remove(neighbor)
            features_orig.append(neighbor)

            # !! Recursive call !!
            self.collect_neighbors(neighbor, features_todo, features_orig,
                    external_neighbors)

    #--------------------------------------------------------------------------

    cpdef void collect_vertices_edges(self, list features_orig, list vs_orig,
            list vs_prev, list vs_next, list es_prev, list es_next) except *:
        cdef str _name_ = "TrackFeatureMerger.collect_vertices_edges"
        cdef Feature feature
        cdef object vertex
        for feature in features_orig:
            vertex = feature.vertex()
            vs_orig.append(vertex)
            for other_vertex in vertex.predecessors():
                if other_vertex not in vs_prev:
                    vs_prev.append(other_vertex)
                    es_prev.extend(self.track.graph.es.select(_between=(
                            (other_vertex.index,), (vertex.index,))))
            for other_vertex in vertex.successors():
                if other_vertex not in vs_next:
                    vs_next.append(other_vertex)
                    es_next.extend(self.track.graph.es.select(_between=(
                            (vertex.index,), (other_vertex.index,))))

        if not self.debug:
            return

        if len(es_prev) != len(vs_prev) or len(es_next) != len(vs_next):
            err = ("{}: inconsistent number of vertices ({}+{}) and edges "
                    "({}+{})").format(_name_, len(vs_prev), len(vs_next),
                    len(es_prev), len(es_next))
            raise Exception(err)

        if (len(set(vs_prev)) != len(vs_prev) or
                len(set(vs_next)) != len(vs_next)):
            err = "{}: duplicate vertices: [{}], [{}]".format(_name_,
                    ", ".join([str(v["name"]) for v in vs_prev]),
                    ", ".join([str(v["name"]) for v in vs_next]))
            raise Exception(err)

    cpdef void collect_merge_vs_attrs(self, dict vx_attrs, list vs_orig,
            Feature merged_feature) except *:
        cdef str _name_ = "TrackFeatureMerger.collect_merge_vs_attrs"
        vx_attrs["feature"] = merged_feature
        if "active_head" in vs_orig[0].attributes():
            vx_attrs["active_head"] = all([vertex["active_head"]
                    for vertex in vs_orig])
        vx_attrs["name"] = str(merged_feature.id)
        vx_attrs["ts"] = merged_feature.timestep
        vx_attrs["type"] = None # set later
        vx_attrs["missing_predecessors"] = None
        vx_attrs["missing_successors"] = None
        if not all([a in vx_attrs.keys() for a in self.vs_attrs]):
            err = "forgotten vertex attributes: {}".format(
                    sorted(self.vs_attrs - vx_attrs.keys()))
            raise Exception(err)

    cpdef void collect_merge_es_attrs(self, dict es_attrs_prev,
            dict es_attrs_next, list vs_orig, list vs_prev, list vs_next,
            list es_prev, list es_next, Feature merged_feature) except *:
        """Collect attributes of vertices and edges involved in the merging.

        Collect vertices of all features parallel to the merged feature
        that are connected to features at the previour/next timestep.

        Ex: If feature A is split into B and C, and C is part of the
            merged feature, then we want to collect features like B.

        Recompute a single set of successor probabilities shared by all
        involved edges, even for features that are not merged.

        Example:

        B0 and B1 are merged, but A3 is also connected to B2

              A0  A1 A2    A3         A0 A1 A2 A3
                \ | /     /  \    =>    \ | | / \
                  B0 <-> B1   B2           BM   B2

        Now, we have a many-to-many successorship, which is otherwise not
        possible with our algorithm! Properly resolving this to get full
        consistency (i.d. no such many-to-many relationships) might
        result in track splitting etc. or complex computations to get
        globally consistent p_tot (I fear).

        This is why we just ignore connections such as A3-B2!

        So the concrete steps are (for "prev", i.e. backwards in time):

          - Identify all vertices at the previous timestep connected
            to any of the features to be merged

          - Also identify all any edged from these vertices to other
            current vertices, which are not part of the merging;
            set p_share of these edges to -1 and otherwise ignore them
            (issue a warning, though, to get an idea how frequent this is)

          - Now compute a single merging-event successor probability,
            which is the same for all directly involved edges (only
            differing in p_share):

              - p_size = min(A*.n, BM)/max(A*.n, BM)
              - p_ovlp = 2*ovlp(A*, BM)/(A*.n + BM)
              - p_tot = f_size*p_size + f_ovlp*p_ovlp
              - p_share(i) = p_tot(i)/SUM[i..n](p_tot(i))
        """
        cdef str _name_ = "TrackFeatureMerger.collect_merge_es_attrs"

        # Backwards
        self._collect_merge_es_attrs_core(es_attrs_prev, es_prev, vs_prev,
                vs_orig, merged_feature, direction=-1)

        # Forwards
        self._collect_merge_es_attrs_core(es_attrs_next, es_next, vs_next,
                vs_orig, merged_feature, direction=1)

    cpdef void _collect_merge_es_attrs_core(self, dict es_attrs, list es_dir,
            list vs_dir, list vs_orig, Feature merged_feature,
            int direction) except *:

        cdef bint debug = self.debug
        cdef str _name_ = "TrackFeatureMerger._collect_merge_es_attrs_core"

        # Find all edges connected to any unrelated vertices at the timestep
        # of the feature merging and 'disable' those edges
        cdef object vertex, other_vertex, edge, other_edge
        for vertex in vs_dir:
            for other_vertex in vertex.neighbors():
                if other_vertex["ts"] != merged_feature.timestep:
                    continue
                edge = self.track.graph.es.find(_between=(
                        (vertex.index,), (other_vertex.index,)))
                if edge in es_dir:
                    continue
                #log.warning(("[{}] warning: track {}: many-to-many "
                #        "relationships between vertices").format(self.timestep,
                #        self.track.id))

        # Select features of involved vertices (minus the merged feature)
        cdef Feature feature
        cdef list features_dir = [vertex["feature"] for vertex in vs_dir]
        #SR_DBG<
        if (len(features_dir) >
                len(set([feature.id for feature in features_dir]))):
            err = "{}: duplicate feature(s): {}".format(_name_, ", ".join(sorted([str(f.id) for f in features_dir])))
            raise Exception(err)
        #SR_DBG>
        #DBG_BLOCK<
        if debug:
            fids = [str(f.id) for f in features_dir]
            if direction < 0:
                dbg_features = "[{}]->[{}]".format("/".join(fids), merged_feature.id)
            else:
                dbg_features = "[{}]->[{}]".format(merged_feature.id, "/".join(fids))
        #DBG_BLOCK>

        # Collect total size and overlap for all feature,
        # and compute individual successor probabilities for p_share
        cdef list p_size_lst=[], p_ovlp_lst=[], p_tot_lst=[]
        cdef list n_lst=[], ovlp_lst=[]
        cdef int ovlp_i
        cdef float p_size_i, p_ovlp_i, p_tot_i
        for feature in features_dir:

            ovlp_i = merged_feature.overlap_n(feature)
            p_size_i = (float(min([feature.n, merged_feature.n]))/
                        float(max([feature.n, merged_feature.n])))
            p_ovlp_i = 2.0*ovlp_i/float(feature.n + merged_feature.n)
            p_tot_i = (self.track.config["f_size"]*p_size_i +
                       self.track.config["f_overlap"]*p_ovlp_i)

            n_lst.append(feature.n)
            ovlp_lst.append(ovlp_i)
            p_size_lst.append(p_size_i)
            p_ovlp_lst.append(p_ovlp_i)
            p_tot_lst.append(p_tot_i)

        if debug: log.debug(dbg_features+": individual probabilities (for p_share):")
        if debug: log.debug("  id        : {}".format("  ".join(["{:5}".format(f.id) for f in features_dir])))
        if debug: log.debug("  n         : {}".format("  ".join(["{:5}".format(n) for n in n_lst])))
        if debug: log.debug("  ovlp      : {}".format("  ".join(["{:5}".format(n) for n in ovlp_lst])))
        if debug: log.debug("  p_size    : {}".format("  ".join(["{:5.3f}".format(p) for p in p_size_lst])))
        if debug: log.debug("  p_overlap : {}".format("  ".join(["{:5.3f}".format(p) for p in p_ovlp_lst])))
        if debug: log.debug("  p_tot     : {}".format("  ".join(["{:5.3f}".format(p) for p in p_tot_lst])))

        # Compute successor total probabilities over the whole set of features
        cdef float p_tot_sum = sum(p_tot_lst)
        cdef int   tot_n     = sum(n_lst), tot_ovlp = sum(ovlp_lst)
        cdef list  p_shares  = [p/p_tot_sum for p in p_tot_lst]
        cdef float n_min     = min([tot_n, merged_feature.n])
        cdef float n_max     = max([tot_n, merged_feature.n])
        cdef float p_size    = n_min/n_max
        cdef float p_ovlp    = 2*float(tot_ovlp)/float(n_min + n_max)
        cdef float p_tot = (self.track.config["f_size"]*p_size +
                            self.track.config["f_overlap"]*p_ovlp)

        if debug: log.debug(dbg_features+" overall probabilities:")
        if debug: log.debug(" - p_size    : {:5.3f}".format(p_size))
        if debug: log.debug(" - p_overlap : {:5.3f}".format(p_ovlp))
        if debug: log.debug(" - p_tot     : {:5.3f}".format(p_tot))
        if debug: log.debug(" - p_shares  : [{}]".format(p_size, p_ovlp, p_tot, ", ".join(["{:5.3f}".format(p) for p in p_shares])))

        # Store edge properties, using the vertex name as key
        cdef int i
        for i, feature in enumerate(features_dir):
            es_attrs[feature.vertex()["name"]] = dict(
                    n_overlap=ovlp_lst[i], p_size=p_size, p_overlap=p_ovlp,
                    p_tot=p_tot, p_share=p_shares[i])

    cpdef void replace_vertices_edges(self, Feature merged_feature,
            list features_orig, list vs_orig, list vs_prev, list vs_next,
            list es_prev, list es_next, dict vx_attrs, dict es_attrs_prev,
            dict es_attrs_next) except *:

        cdef bint debug = self.debug
        cdef str _name_ = "TrackFeatureMerger.replace_vertices_edges"

        # Remove old edges
        cdef es_all = es_prev + es_next
        cdef object edge
        cdef list eid_del = [edge.index for edge in es_all]
        self.track.graph.delete_edges(eid_del)
        es_prev.clear(); es_next.clear(); es_all.clear() # invalid objects
        cdef Feature feature
        for feature in features_orig:
            feature.reset_vertex()

        # Store prev/next features because respective vertex lists will be
        # invalid after replace the old pre-merge vertices by the new one
        cdef object vertex
        cdef list features_prev=[], features_next=[]
        for vertex in vs_prev:
            features_prev.append(vertex["feature"])
        for vertex in vs_next:
            features_next.append(vertex["feature"])

        # Remove old vertices
        cdef list vs_types_old=[vertex["type"] for vertex in vs_orig]
        cdef list vid_del = [vertex.index for vertex in vs_orig]
        self.track.graph.delete_vertices(vid_del)
        vs_orig.clear(); vs_prev.clear(); vs_next.clear() # invalid objects

        # Add vertex for merged feature
        self.track.graph.add_vertex(**vx_attrs)
        cdef object new_vertex = self.track.graph.vs[-1]
        merged_feature.set_vertex(new_vertex)
        if (vx_attrs["feature"] != merged_feature or
                merged_feature.vertex()["feature"] != merged_feature):
            log.error("{}: error adding merged feature to graph".format(_name_))
            exit(4)

        # Add edges to predecessor/successor vertices
        cdef list eid_add=[]
        cdef str vx_name
        cdef dict attrs_dir, attrs
        cdef str key
        cdef dict es_attrs = {key: [] for key in self.es_attrs}
        for vx_name, attrs in es_attrs_prev.items():
            vertex = self.track.graph.vs.find(vx_name)
            eid_add.append((vertex.index, new_vertex.index))
            for key, val in attrs.items():
                es_attrs[key].append(val)
        for vx_name, attrs in es_attrs_next.items():
            vertex = self.track.graph.vs.find(vx_name)
            #SR_TMP< SR_DIRECTED
            if new_vertex["ts"] > vertex["ts"]:
                eid_add.append((vertex.index, new_vertex.index))
            else:
            #SR_TMP>
                eid_add.append((new_vertex.index, vertex.index))
            for key, val in attrs.items():
                es_attrs[key].append(val)
        self.track.graph.add_edges(eid_add)

        # Add attributes to new edges
        cdef object es_new
        es_new = self.track.graph.es.select(_between=list(zip(*eid_add)))
        cdef list vals
        for key, vals in es_attrs.items():
            es_new[key] = vals

        # Set vertex types
        self.set_feature_types(features_prev)
        self.set_feature_types(features_next)
        self.set_feature_types([merged_feature])

    cpdef void set_feature_types(self, list features) except *:
        cdef str _name_ = "TrackFeatureMerger.set_feature_types"

        cdef object vertex, other_vertex
        cdef list vs_prev, vs_next
        cdef np.uint64_t timestep, other_timestep
        cdef str type, old_type, new_type
        cdef int n_prev, n_next
        cdef list types
        for feature in features:
            vertex = feature.vertex()
            old_type = vertex["type"]
            timestep = feature.timestep

            #SR_TODO Make graph directed to make ts comparisons obsolete
            # Collect neighboring vertices
            vs_prev = []
            vs_next = []
            for other_vertex in vertex.neighbors():
                other_timestep = other_vertex["ts"]
                if other_timestep < timestep:
                    vs_prev.append(other_vertex)
                elif other_timestep > timestep:
                    vs_next.append(other_vertex)
                else:
                    err = ("{}: link between vertices of same timestep ({}): "
                            "features {}, {}").format(_name_, vertex["ts"],
                            vertex["feature"].id, other_vertex["feature"].id)
                    raise Exception(err)
            n_prev = len(vs_prev)
            n_next = len(vs_next)

            # Determine and set new type
            if n_prev == 0 and n_next == 0:
                err = "{}: vertex {} has no neighbors (feature {})".format(
                        _name_, vertex.index, vertex["feature"].id)
                raise Exception(err)
            elif n_prev == 1 and n_next == 1:
                new_type = "continuation"
            elif n_prev == 0:
                types = self.track.graph.vs.select(ts=timestep)["type"]
                if any([type and "start" in type for type in types]):
                    new_type = "start"
                else:
                    new_type = "genesis"
                if n_next > 1:
                    new_type = "{}/splitting".format(new_type)
            elif n_next == 0:
                types = self.track.graph.vs.select(ts=timestep)["type"]
                if any([type and "stop" in type for type in types]):
                    new_type = "stop"
                else:
                    new_type = "lysis"
                if n_prev > 1:
                    new_type = "merging/{}".format(new_type)
            elif 0 < n_prev <= 1 and n_next > 1:
                new_type = "splitting"
            elif n_prev > 1 and 0 < n_next <= 1:
                new_type = "merging"
            elif n_prev > 1 and n_next > 1:
                new_type = "merging/splitting"
            else:
                err = "{}: not implemented: {} next, {} prev".format(
                        _name_, n_prev, n_next)
                raise Exception(err)
            vertex["type"] = new_type

#==============================================================================
# Feature wrapper class for compatibility with old-style tracking
#==============================================================================

class TrackableFeature_Oldstyle(Feature):
    """Wrapper class for Feature which is compatible with the old-style tracking."""

    cls_combination = None # Set below

    def __init__(self, values, pixels, center, extrema, shells, holes, id):
        super().__init__(values, pixels, center, extrema, shells, holes, id)
        self._id = id
        self._center = center

        #SR_TMP<
        self._is_unassigned = True
        self._event = None
        self.attr = {}
        #SR_TMP>

    def __repr__(self):
        return ("<{cls}[{id0}]: id={id1}, area={a}, min={min:5.3f}, "
                "mean={mean:5.3f}, max={max:5.3f}>").format(
                cls=self.__class__.__name__, id0=id(self), id1=self.id_str(),
                a=self.area(), min=np.min(self.values),
                mean=np.mean(self.values), max=np.max(self.values))

    @classmethod
    def from_feature2d(cls, feature):
        """Make a 'new-style' Feature object trackable."""
        return cls(
                feature.values,
                feature.pixels,
                feature.center,
                feature.extrema,
                feature.shells,
                feature.holes,
                feature.id,
            )

    def id(self):
        return self._id

    def __eq__(self, other):
        return self.id() == other.id()

    def __lt__(self, other):
        return self.id() < other.id()

    def __hash__(self):
        return id(self)

    def get_info(self, path=True):
        jdat = odict()
        #jdat["class"] = self.__class__.__name__
        jdat["class"] = "GenericFeature"
        jdat["id"] = self.id()
        if self.event():
            jdat["event"] = self.event().id()
        jdat["area"] = self.area()
        if path:
            jdat["path"] = [[int(x), int(y)] for x, y in self.shells]
        jdat["center"] = [int(self.center()[0]), int(self.center()[1])]
        for key, val in self.attr.items():
            jdat["attr_"+key] = val
        return jdat

    def center(self):
        return self._center

    def area(self):
        return self.n

    def area_ratio(self, other):
        if self.area() > other.area():
            return self.area()/other.area()
        return other.area()/self.area()

    def event(self):
        return self._event

    def link_event(self, event):
        self._event = event

    def id_str(self):
        return str(self.id())

    def intersects(self, other):
        return self.overlaps(other)

    def radius(self):
        return np.sqrt(self.area()/np.pi)

    def center_distance(self, other):
        cdef int x0, x1, y0, y1
        x0, y0 = self.center()
        x1, y1 = other.center()
        return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    def overlap(self, other):
        if other.__class__.__name__ == "TrackableFeature_Oldstyle":
            return self.overlap_n(other)
        elif other.__class__.__name__ == "TrackableFeatureCombination_Oldstyle":
            return np.sum([self.overlap_n(o) for o in other.features()])
        err = "{}.overlap_fraction({})".format(
                self.__class__.__name__, other.__class__.__name__)
        raise NotImplementedError(err)

    def overlap_fraction(self, other):
        return 2*self.overlap(other)/(self.area() + other.area())

    def is_periodic(self):
        return False

    def is_unassigned(self):
        return self.event() is None

#------------------------------------------------------------------------------

class TrackableFeatureCombination_Oldstyle(TrackableFeature_Oldstyle):

    def __init__(self, features):
        self._features = [f for f in features]
        self._event = None

    def features(self):
        return self._features.copy()

    def id(self):
        raise NotImplementedError("{}.id".format(self.__class__.__name__))

    def __eq__(self, other):
        return self.id_str() == other.id_str()

    def __lt__(self, other):
        return (sum([int(i) for i in self.id_str().split("/")]) <
                sum([int(i) for i in self.id_str().split("/")]))

    def __hash__(self):
        return id(self)

    def get_info(self, path=True):
        raise NotImplementedError("{}.get_info".format(self.__class__.__name__))

    def center(self):
        return np.mean([f.center() for f in self.features()], axis=0)

    def area(self):
        return np.sum([f.area() for f in self.features()])

    def id_str(self):
        return "/".join([f.id_str() for f in self.features()])

    def overlaps(self, other):
        return any(Feature.overlaps(f, other) for f in self.features())

    def overlap(self, other):
        if other.__class__.__name__ == "TrackableFeature_Oldstyle":
            return np.sum([f.overlap_n(other) for f in self.features()])
        elif other.__class__.__name__ == "TrackableFeatureCombination_Oldstyle":
            return np.sum([f.overlap_n(o)
                    for f in self.features() for o in other.features()])
        err = "{}.overlap_fraction({})".format(
                self.__class__.__name__, other.__class__.__name__)
        raise NotImplementedError(err)

    def is_unassigned(self):
        if all(f.is_unassigned() for f in self.features()):
            return True
        if not any(f.is_unassigned() for f in self.features()):
            return False
        err = "{}[{}]: some but not all features are unassigned".format(
                self.__class__.__name__, self.id_str())
        raise Exception(err)

TrackableFeature_Oldstyle.cls_combination = TrackableFeatureCombination_Oldstyle

#==============================================================================


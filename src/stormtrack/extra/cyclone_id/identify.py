#!/usr/bin/env python

# Standard library
import os
import logging as log
import numpy as np

# Local
from .cyclone import Cyclone
from .cyclone import CycloneIOReaderBinary
from .cyclone import CycloneIOWriterBinary
from .cyclone import Depression
from .. import timing
from ..utilities_misc import Contour
from ..utilities_misc import DOMAIN_BOUNDARY_CONTOUR
from ...utils.spatial import path_along_domain_boundary


__all__ = []


def identify_features(slp, topo, conf):

    # Initialize sub-timer
    timings = conf["timings-identify"]
    if timings:
        timer = timing.get_timer("current_file").add_subtimer(
            "identification", "main", reset_existing=True
        )
        timer.add_timers(
            ["preparation", "depressions", "cyclones"], reset_existing=True
        )

    if timings:
        timer.start("preparation")

    # Smoothing, find minima/maxima, identify contours
    prepare_slp(slp, topo, conf)

    if timings:
        timer.end("preparation")

    # Make domain boundary contour globally available
    store_domain_boundary(slp)

    if timings:
        timer.start("depressions")

    if conf.get("ids-datetime", False):
        id0 = conf["datetime"] * 10 ** conf["ids-datetime-digits"]
    else:
        id0 = None

    # Identify depressions (contour clusters around minima)
    depressions = Depression.create(
        contours=slp.contours(),
        minima=slp.minima(),
        maxima=slp.maxima(),
        contours_valid=False,
        ncont_min=conf["depression-min-contours"],
        bcc_frac=conf["bcc-fraction"],
        len_min=conf["contour-length-min"],
        len_max=conf["contour-length-max"],
        id0=id0,
    )

    if timings:
        timer.end("depressions")

    if timings:
        timer.start("cyclones")

    # Identify cyclones (up to three centers: SCCs, DCCs, TCCs)
    cyclones = Cyclone.create(
        depression=depressions,
        thresh_dcc=0.5,
        thresh_tcc=0.7,
        min_depth=conf["min-cyclone-depth"],
        nmin_max=conf["max-minima-per-cyclone"],
        ncont_min=conf["depression-min-contours"],
        id0=id0,
    )

    if timings:
        timer.end("cyclones")

    # Return a dictionary for flexibility as to what to return
    features = {"depressions": depressions, "cyclones": cyclones}

    return features


def prepare_slp(slp, topo, conf):

    # Add subtimer
    timings = conf["timings-identify"]
    if timings:
        timer = (
            timing.get_timer("current_file")
            .get_subtimer("identification", "main")
            .add_subtimer("preparation", "main", reset_existing=True)
        )
        timer.add_timers(["smoothing", "extrema", "contours"], reset_existing=True)

    # Compute contour levels
    dp = conf["contour-interval"]
    lvl0 = conf["contour-level-start"]
    lvl1 = conf["contour-level-end"]
    levels = np.arange(lvl0, lvl1 + dp, dp)

    if timings:
        timer.start("smoothing")

    # Smooth the field
    slp.smooth(conf["smoothing-sigma"])

    if timings:
        timer.end("smoothing")

    # Apply the boundary hack for the contour computation
    slp.apply_boundary_hack(1, 1000.0)

    if timings:
        timer.start("extrema")

    # Identify SLP minima/maxima or read them from file
    if conf["read-slp-extrema"]:
        infile_extrema = "{}/{}".format(
            conf["slp-extrema-file-path"], conf["slp-extrema-file"]
        )
        slp.read_extrema(infile_extrema, cls_reader=CycloneIOReaderBinary)
    else:
        slp.identify_minima(conf)
        slp.identify_maxima(conf)
        if conf["save-slp-extrema"]:
            outfile_extrema = "{}/{}".format(
                conf["slp-extrema-file-path"], conf["slp-extrema-file"]
            )
            slp.save_extrema(outfile_extrema)
            if not conf["save-slp-contours"]:
                exit(0)

    # Remove unwanted minima/maxima
    slp.points_filter_boundary(conf["size-boundary-zone"])
    if topo is not None:
        slp.points_filter_topography(topo, conf["topo-cutoff-level"])

    if timings:
        timer.end("extrema")

    if timings:
        timer.start("contours")

    # Compute SLP contours or read them from file
    if conf["read-slp-contours"]:
        infile_contours = "{}/{}".format(
            conf["slp-contours-file-path"], conf["slp-contours-file"]
        )
        slp.read_contours(infile_contours, levels, CycloneIOReaderBinary)
    else:
        slp.compute_contours(levels, conf)
        if conf["save-slp-contours"]:
            outfile_contours = "{}/{}".format(
                conf["slp-contours-file-path"], conf["slp-contours-file"]
            )
            slp.save_contours(outfile_contours, CycloneIOWriterBinary)
            exit(0)

    if timings:
        timer.end("contours")


def store_domain_boundary(fld):
    path = list(
        zip(*path_along_domain_boundary(fld.lon, fld.lat, fld=fld, bnd_nan=True))
    )
    DOMAIN_BOUNDARY_CONTOUR = Contour(path, level=-1, id=-1)


if __name__ == "__main__":
    pass

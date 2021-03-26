#!/usr/bin/env python3

# Standard library
import datetime as dt
import functools
import json
import logging as log
import os
import re
import warnings
from multiprocessing import Pool

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import scipy as sp
import shapely.geometry as geo
from descartes.patch import PolygonPatch
from matplotlib import cm
from mpl_toolkits.basemap import Basemap

try:
    from numpy.ma.core import MaskedArrayFutureWarning
except ImportError:
    MaskedArrayFutureWarning = None  # type: ignore

# Local
from ..utils.netcdf import nc_prepare_file
from ..utils.spatial import path_along_domain_boundary
from .utilities_misc import Domain
from .utilities_misc import Field2D
from .utilities_misc import inds2lonlat
from .utilities_misc import order_dict


__all__ = []

# Plot precip

PRECIP_LEVELS_PSEUDO_LOG_ORIG = np.array(
    [
        0.1,
        0.2,
        1.0,
        2.0,
        4.0,
        6.0,
        10.0,
        20.0,
        40.0,
        60.0,
    ]
)
PRECIP_LEVELS_PSEUDO_LOG = np.array(
    [
        0.1,
        0.22,
        0.46,
        1,
        2.2,
        4.6,
        10,
        22,
        46,
        100,
    ]
)
PRECIP_LEVELS_PSEUDO_LOG_NARROW = np.array(
    [
        1,
        1.5,
        2.2,
        3.2,
        4.6,
        7,
        10,
        15,
        22,
        32,
    ]  # 46,
)

PRECIP_LEVELS_LOG = 10 ** np.arange(-1, 2.1, 0.2)
PRECIP_LEVELS_LOG_NARROW = 10 ** np.arange(0, 1.6, 0.1)
assert len(PRECIP_LEVELS_LOG) == 16

# Precip NCL colormap 'precip2_17lev'
# src: www.ncl.ucar.edu/Document/Graphics/ColorTables/precip2_17lev.shtml
PRECIP_COLORS_RGB_RADAR = [
    (100, 100, 100),
    (150, 130, 150),
    (4, 2, 252),
    (4, 142, 44),
    (4, 254, 4),
    (252, 254, 4),
    (252, 202, 4),
    (252, 126, 4),
    (252, 26, 4),
    (172, 2, 220),
]
PRECIP_COLORS_HEX_RADAR = [
    "{:02X}{:02X}{:02X}".format(r, g, b) for r, g, b in PRECIP_COLORS_RGB_RADAR
]

PRECIP_COLORS_RGB_MCH17 = [
    (255, 255, 255),
    # (235, 246, 255),
    (214, 226, 255),
    (181, 201, 255),
    (142, 178, 255),
    (127, 150, 255),
    (114, 133, 248),
    (99, 112, 248),
    (0, 158, 30),
    (60, 188, 61),
    (179, 209, 110),
    (185, 249, 110),
    (255, 249, 19),
    (255, 163, 9),
    (229, 0, 0),
    (189, 0, 0),
    (129, 0, 0),
    # (  0,   0,   0),
]
PRECIP_COLORS_HEX_MCH17 = [
    "{:02X}{:02X}{:02X}".format(r, g, b) for r, g, b in PRECIP_COLORS_RGB_RADAR
]


def create_cmap_precip(
    colors_rgb=PRECIP_COLORS_RGB_RADAR,
    levels=PRECIP_LEVELS_PSEUDO_LOG,
    over="black",
    lognorm=False,
):
    """Create precipitation colormap."""

    if len(levels) != len(colors_rgb):
        err = ("numbers of precip levels and colors differ: {} != {}").format(
            len(levels), len(colors_rgb)
        )
        raise ValueError(err)

    if lognorm:
        levels = np.log10(levels)

    cols = np.array(colors_rgb) / 255
    fct = lambda l: (l - levels[0]) / (levels[-1] - levels[0])
    cols_cmap = [(fct(l), c) for l, c in zip(levels, cols)]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("precip", cols_cmap)
    cmap.set_under("white", alpha=0)

    cmap.set_over(over)

    return cmap


cmap_precip_pseudo_log = create_cmap_precip(
    PRECIP_COLORS_RGB_RADAR, PRECIP_LEVELS_PSEUDO_LOG
)
cmap_precip_pseudo_log__lognorm = create_cmap_precip(
    PRECIP_COLORS_RGB_RADAR, PRECIP_LEVELS_PSEUDO_LOG, lognorm=True
)
cmap_precip_pseudo_log_narrow__lognorm = create_cmap_precip(
    PRECIP_COLORS_RGB_RADAR, PRECIP_LEVELS_PSEUDO_LOG_NARROW, lognorm=True
)

cmap_precip_log = create_cmap_precip(PRECIP_COLORS_RGB_MCH17, PRECIP_LEVELS_LOG)


def plot_precip(
    outfile,
    title,
    fld,
    lon=None,
    lat=None,
    *,
    grid=None,
    levels=None,
    topo=None,
    cmap_topo="terrain",
    cmap=None,
    clabel=None,
    map_limits=None,
    logtrans=False,
    title_standalone=False,
    cbar_standalone=False,
    cbar_extend="max",
    cbar_orientation="horizontal",
    cbar_ticklabel_rotation=None,
    cbar_ticklabel_offset=0,
    cbar_ticklabel_stride=1,
    draw_gridlines=True,
    title_x=0.5,
    title_y=1.02,
    dpi=300,
    title_fs=12,
    fsS=14,
    fsM=16,
    fsL=18,
    fsScale=1,
):
    if title_standalone or cbar_standalone:
        outfile = outfile.replace(".png", ".plot.png")
    print("plot " + outfile)

    if lon is None or lat is None:
        if grid is None:
            raise ValueError("must pass lon/lat or grid")
        lon, lat = grid["lon"], grid["lat"]

    n_levels_default = 10

    auto_levels = levels is None

    fsS *= fsScale
    fsM *= fsScale
    fsL *= fsScale

    fig, ax = plt.subplots()
    w_standalone = 0.6 * fig.get_size_inches()[0]

    m = setup_map_crclim(
        lon,
        lat,
        ax=ax,
        lw_coasts=2,
        map_limits=map_limits,
        draw_gridlines=draw_gridlines,
    )
    mlon, mlat = m(lon, lat)

    if topo is not None:
        # Plot topography

        # SR_TMP<
        topo_mode = "color"
        # SR_TMP>

        if topo_mode == "color":
            levels_topo = np.arange(0, 4001, 500)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=MaskedArrayFutureWarning)
                ax.contourf(mlon, mlat, topo, levels=levels_topo, cmap=cmap_topo)

        elif topo_mode == "contour":
            levels_topo = np.arange(0, 4001, 1000)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=MaskedArrayFutureWarning)
                ax.contour(
                    mlon, mlat, topo, levels=levels_topo, colors="black", linewidths=0.5
                )
        else:
            raise ValueError("invalid topography plot mode: " + topo_mode)

    if auto_levels and logtrans:
        # Try to derive levels somewhat smartly
        # If it fails, leave it to matplotlib
        try:
            logmin = np.log10(np.percentile(fld[fld > 0], 1))
            logmax = np.log10(np.percentile(fld[fld > 0], 99))
            if logmin == logmax:
                levels = None
            else:

                levels = 10 ** np.linspace(logmin, logmax, n_levels_default)
        except:
            levels = None

    if not logtrans:
        fld_plt = fld
        levels_plt = levels
    else:
        # Use logarithmic contour levels
        # Manual transformation rather than LogNorm() to allow 'extend'
        with np.errstate(divide="ignore"):
            fld_plt = np.where(fld > 0, np.log10(fld), np.nan)
        levels_plt = np.log10(levels) if levels is not None else None

    # Plot field
    _lvls = n_levels_default if auto_levels else levels_plt
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MaskedArrayFutureWarning)
        p = ax.contourf(mlon, mlat, fld_plt, _lvls, cmap=cmap, extend=cbar_extend)

    if levels_plt is None:
        # Extract levels that matplotlib computed on its own
        levels_plt = np.asarray(p.levels)
        levels = 10 ** levels_plt if logtrans else levels_plt

    # Determine how to format colorbar labels
    if all(
        int(lvl) == float(lvl)
        for lvl in levels[cbar_ticklabel_offset::cbar_ticklabel_stride]
    ):
        sigdig = None
        stripzero = False
    else:
        # sigdig = max([2, len(str(int(max(levels))))])
        sigdig = max(
            [2, max([len("{:f}".format(l).strip("0").split(".")[1]) for l in levels])]
        )
        stripzero = True

    # Add colorbar (optionally as standalone plot)
    outfile_cbar = outfile.replace(".plot.png", ".cbar.png")
    plot_cbar(
        levels,
        fig=fig,
        p=p,
        standalone=cbar_standalone,
        label=clabel,
        levels_plt=levels_plt,
        sigdig=sigdig,
        stripzero=stripzero,
        cmap=cmap,
        extend=cbar_extend,
        outfile=outfile_cbar,
        orientation=cbar_orientation,
        w=w_standalone,
        dpi=dpi,
        ticklabel_rotation=cbar_ticklabel_rotation,
        tick_offset=cbar_ticklabel_offset,
        tick_stride=cbar_ticklabel_stride,
        fsS=fsS,
        fsM=fsM,
        fsL=fsL,
        fsScale=1,
    )

    if title:
        # Add title (optionally as standalone plot)
        if not title_standalone:
            ax.set_title(title, fontsize=title_fs, x=title_x, y=title_y)
        else:
            outfile_title = outfile.replace(".plot.png", ".title.png")
            plot_title_standalone(
                title, outfile_title, fs=title_fs, w=w_standalone, dpi=dpi
            )

    fig.savefig(outfile, bbox_inches="tight", dpi=dpi)
    plt.close("all")


def plot_title_standalone(title, outfile, *, w=6, dpi=None, fs=12):
    """Save plot title to separate file."""
    fig, ax = plt.subplots(figsize=(w, w / 20))
    ax.axis("off")
    ax.text(
        0.5,
        -1.0,
        title,
        transform=fig.transFigure,
        fontsize=fs,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    fig.savefig(outfile, bbox_inches="tight", dpi=dpi)


def plot_cbar(
    levels,
    *,
    levels_plt=None,
    levels_con=None,
    levels_con_inds=None,
    fig=None,
    p=None,
    w=6,
    dpi=None,
    standalone=False,
    fmt=None,
    sigdig=None,
    stripzero=False,
    outfile=None,
    align_ticklabels="left",
    tick_offset=0,
    tick_stride=1,
    cmap=None,
    label=None,
    extend=None,
    orientation="horizontal",
    ticklabel_rotation=None,
    fsS=14,
    fsM=16,
    fsL=18,
    fsScale=1,
):

    fsS *= fsScale
    fsM *= fsScale
    fsL *= fsScale

    if levels_plt is None:
        levels_plt = levels

    # Select and format tickmark labels
    if align_ticklabels == "left":
        cb_ticks = levels_plt[tick_offset::tick_stride]
        cb_ticklabels = format_ticklabels(
            levels[tick_offset::tick_stride],
            fmt=fmt,
            sigdig=sigdig,
            stripzero=stripzero,
        )
    elif align_ticklabels == "right":
        cb_ticks = levels_plt[::-1][tick_offset::tick_stride][::-1]
        cb_ticklabels = format_ticklabels(
            levels[::-1][tick_offset::tick_stride][::-1],
            fmt=fmt,
            sigdig=sigdig,
            stripzero=stripzero,
        )
    else:
        err = "invalid tickmark label alignment '{}'".format(align_ticklabels)
        raise ValueError(err)

    kwas_plt = dict(levels=levels_plt, cmap=cmap, extend=extend)
    kwas_cb = dict(ticks=cb_ticks, orientation=orientation, extend=extend)

    if not standalone:
        # SR_TMP<
        if levels_con is not None:
            raise NotImplementedError("levels_con and not standalone")
        if levels_con_inds is not None:
            raise NotImplementedError("levels_con_inds and not standalone")
        # SR_TMP>

        # Add cbar to plot
        if orientation == "horizontal":
            kwas_cb.update(dict(shrink=0.55, pad=0.04))
        elif orientation == "vertical":
            kwas_cb.update(dict(shrink=0.85))  # , pad=0.04))
        cb = fig.colorbar(p, **kwas_cb)
        cb.set_label(label, size=fsM)
        _kwas = dict(rotation=ticklabel_rotation, fontsize=fsS)
        if orientation == "horizontal":
            cb.ax.set_xticklabels(cb_ticklabels, **_kwas)
        elif orientation == "vertical":
            cb.ax.set_yticklabels(cb_ticklabels, **_kwas)
    else:
        # Plot cbar to separate file
        save_cbar_standalone(
            outfile,
            kwas_plt,
            kwas_cb,
            w=w,
            dpi=dpi,
            levels_con=levels_con,
            levels_con_inds=levels_con_inds,
            label=label,
            ticklabels=cb_ticklabels,
            ticklabel_rotation=ticklabel_rotation,
            fsS=fsS,
            fsM=fsM,
        )


def save_cbar_standalone(
    outfile,
    kwas_cont,
    kwas_cbar,
    *,
    label=None,
    ticklabels=None,
    w=6,
    dpi=None,
    ticklabel_rotation=None,
    levels_con=None,
    levels_con_inds=None,
    fsS=14,
    fsM=16,
):

    orientation = kwas_cbar.get("orientation")

    fig, ax = plt.subplots(figsize=(w, w / 6))
    ax.axis("off")

    gs = mpl.gridspec.GridSpec(2, 1, bottom=0.6, height_ratios=[0, 1])
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MaskedArrayFutureWarning)
        p = ax0.contourf(
            [[0, 1], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [0, 0]], **kwas_cont
        )
    ax0.set_visible(False)

    cb = fig.colorbar(p, cax=ax1, **kwas_cbar)

    if label is not None:
        cb.set_label(label, size=fsM)

    _kwas = dict(rotation=ticklabel_rotation, fontsize=fsS)
    if orientation == "horizontal":
        cb.ax.set_xticklabels(ticklabels, **_kwas)
    elif orientation == "vertical":
        cb.ax.set_yticklabels(ticklabels, **_kwas)
    else:
        cb.ax.set_xticklabels(ticklabels, **_kwas)
        cb.ax.set_yticklabels(ticklabels, **_kwas)

    if levels_con is not None:
        # Add contour levels
        levels = kwas_cont["levels"]
        for lvl in levels_con:
            lvl_01 = (lvl - levels[0]) / (levels[-1] - levels[0])
            if orientation == "horizontal":
                cb.ax.axvline(lvl_01, c="black")
            elif orientation == "vertical":
                cb.ax.axhline(lvl_01, c="black")
            else:
                raise ValueError("must pass orientation alongsize levels_con")

    if levels_con_inds is not None:
        # Add contour levels based on color levels
        levels = kwas_cont["levels"]
        for ind in levels_con_inds:
            lvl_01 = ind * 1 / (len(levels) - 1)
            if orientation == "horizontal":
                cb.ax.axvline(lvl_01, c="black")
            elif orientation == "vertical":
                cb.ax.axhline(lvl_01, c="black")
            else:
                raise ValueError("must pass orientation alongsize levels_con_inds")

    fig.savefig(outfile, dpi=dpi)


def format_ticklabels(labels, fmt=None, stripzero=False, sigdig=None):

    if fmt is None:
        fmt = "{:g}"

    if sigdig is not None and not (isinstance(sigdig, int) and sigdig > 0):
        raise ValueError("sigdig not a positive number: {}".format(sigdig))

    labels_fmtd = []
    for label in labels:
        if not sigdig:
            try:
                label_fmtd = fmt.format(label)
            except TypeError:
                label_fmtd = label
        else:
            if label == 0:
                label_fmtd = "0"
                if sigdig > 1:
                    # label_fmtd = label_fmtd+"."+"0"*(sigdig - 1)
                    label_fmtd = "0.0"
            else:
                _f = 10 ** (sigdig + 1 - np.ceil(np.log10(label)))
                try:
                    label_fmtd = "{:g}".format(int(label * _f + 0.5) / _f)
                except ValueError:
                    # E.g., in case of NaN
                    label_fmtd = str(label * _f)

                # Append zeros if necessary
                if "." in label_fmtd:
                    pre, post = label_fmtd.split(".")
                else:
                    pre, post = label_fmtd, ""
                if pre == "0":
                    n = len(post.lstrip("0"))
                else:
                    n = len(pre) + len(post)
                if n < sigdig:
                    post += "0" * (sigdig - n)
                    label_fmtd = "{}.{}".format(pre, post)

        if stripzero and label != 0:
            # Remove leading zero bevore decimal point
            label_fmtd = label_fmtd.lstrip("0")

        labels_fmtd.append(label_fmtd)

    return labels_fmtd


# INPUT


def import_lonlat(lonlat_file, lon_name="lon", lat_name="lat"):
    """Import the fields 'lon' and 'lat' from an NPZ archive."""
    try:
        with np.load(lonlat_file) as f:
            lon, lat = f[lon_name], f[lat_name]
    except KeyError as e:
        err = "Field {} not found in file {}".format(e, lonlat_file)
        raise IOError(err)
    except Exception as e:
        err = "Error reading lon/lat file: {}({})".format(e.__class__.__name__, e)
        raise IOError(err)
    else:
        return lon, lat


def import_tracks(
    cls_reader,
    infiles,
    lon,
    lat,
    domain=None,
    smoothing_sigma=None,
    return_config_id=False,
    return_config_tracker=False,
):
    """Import tracks along with the features from a JSON file."""
    if domain is None:
        domain = Domain(list(zip(*path_along_domain_boundary(lon, lat))))
    reader = cls_reader(domain=domain)
    tracks = []

    config_id = None
    config_track = None
    for infile in sorted(infiles):
        data = reader.read_file(infile, include_tracker_config=return_config_tracker)

        # Make sure the identification configs match
        try:
            data["CONFIG"]["IDENTIFY"]
        except KeyError:
            log.warning("CONFIG/IDENTIFY not found in {}".format(infile))

        if "CONFIG" in data:
            if config_id is None:
                config_id = data["CONFIG"]["IDENTIFY"]
            else:
                if config_id != data["CONFIG"]["IDENTIFY"]:
                    msg = "CONIG/IDENTIFY in {} differs from previous!".format(infile)
                    log.warning(msg)

        # Make sure the tracking configs match
        if return_config_tracker:
            try:
                config_track = data["CONFIG"]["TRACKER"]
            except KeyError:
                log.warning("CONFIG/TRACKER not found in {}".format(infile))
            else:
                if config_track != data["CONFIG"]["TRACKER"]:
                    msg = "CONIG/TRACKER in {} differs from previous!".format(infile)
                    log.warning(msg)

        # Extract tracks
        new_tracks = data["TRACKS"]
        tracks.extend(new_tracks)

        log.info("read {} tracks from file {}".format(len(new_tracks), infile))

    results = [tracks]
    if return_config_id:
        results.append(config_id)
    if return_config_tracker:
        results.append(config_track)
    return results


def write_old_tracks(
    outfile, tracks, cls_writer_json, cls_writer_bin, configs=None, block_order=None
):

    outfile_bin = outfile.replace(".json", ".npz")

    # Exctract contours and minima (necessary to re-build cyclones)
    features = [f for t in tracks for f in t.features() if not f.is_mean()]
    contours = [c for f in features for c in f.contours()]
    minima = [m for f in features for m in f.minima()]

    # Initialize and configure JSON writer
    writer_json = cls_writer_json()
    writer_json.set_config(save_paths=False, contour_path_file=outfile_bin)
    if block_order:
        writer_json.set_config(block_order=block_order)

    # If given, add configs
    if configs:
        for name, config in configs.items():
            writer_json.add_config({name: config})

    # Add tracks etc.
    writer_json.add_tracks(tracks)
    writer_json.add_contours(contours)
    writer_json.add_points("MINIMA", minima)

    # Write JSON file
    writer_json.write_file(outfile)

    # Write contour paths to binary
    writer_bin = cls_writer_bin()
    writer_bin.write_contour_path_file(outfile_bin, contours)


# INPUT: NETCDF/NPZ


def read_topography(input_file, field_name):
    """Read the topography field from the respective input file."""
    log.info("read topography field {n} from {f}".format(n=field_name, f=input_file))

    try:
        # Try netCDF file
        with nc4.Dataset(input_file, "r") as fi:
            lon = fi["lon"][:]
            lat = fi["lat"][:]
            fld_raw = fi[field_name][0]  # strip leading time dimension
    except Exception:
        # Try NPZ archive
        try:
            with np.load(input_file) as f:
                fld_raw = f[field_name]
                lon = f["lon"]
                lat = f["lat"]
        except IOError:
            err = "Cannot import file (unknown format): {}".format(input_file)
            raise IOError(err) from None

    fld = Field2D(fld_raw, lon, lat, name=field_name)

    return fld


# INPUT: JSON


class IOReaderJsonBase:
    def __init__(self):
        self._header = {}

    def read_file(self, filename, **kwas):
        self._data_path = os.path.dirname(os.path.abspath(filename))
        with open(filename) as f:
            jstr = f.read()
        jdat = self.read_string(jstr, **kwas)
        return jdat

    def get_header(self):
        return self._header


def _json_remove_excessive_newlines(jstr, ind, n):
    """Remove newline before lines with a certain indent.

    Problem:
    The JSON writer either inserts no newline characters at all, or
    after every entry. The former is impractical, the latter blows up
    JSON files containing large sets of data (e.g. contour coordinates).

    Solution:
    To keep the JSON file structure clear, keep newlines before lines
    with an indent of up to N spaces. Newline characters before every
    line with more indent are removed.

    Arguments:
     - jstr: Indented JSON string.
     - ind: Number of spaces per level indent.
     - n: Lowest level for which newlines are retained.
    """
    # Remove all newlines for >N spaces indent.
    rx1 = "\n {{{n}, }}(?=[^ ])".format(n=n * ind)

    # Remove newline before closing bracket of list entries at indent N
    rx2 = "\n {{{n}}}(?=[\]}}])".format(n=(n - 1) * ind)

    jstr = re.sub(rx1, "", jstr)
    jstr = re.sub(rx2, "", jstr)
    return jstr


# INPUT: BINARY


class IOReaderBinaryBase:
    def __init__(self):
        pass


# OUTPUT: JSON


class IOWriterJsonBase:
    def __init__(self):
        self._cache = {}
        self._header = {}

    def write_file(self, filename):
        """Write the cached data to a file.

        Arguments:
         - filename: Name of the output JSON file.
        """
        jstr = self.write_string()
        with open(filename, mode="w") as f:
            f.write(jstr)

    def write_string(self):
        """Merge the list of cached JSON strings into a single one.

        The various write methods create stand-alone json blocks as strings.
        Merge the blocks into one block.

        If the property block_order is set (list of names), the blocks
        in the list are written in the respective order.
        """
        if len(self._cache) == 0:
            raise ValueError("Nothing to write!")

        # Order the blocks (alphabetically if not specified otherwise)
        if not "block_order" in self._header:
            block_list_raw = {k: v for k, v in sorted(self._cache.items())}
        else:
            block_list_raw = {}
            names_all = list(self._cache.keys())
            for name in self._header["block_order"]:
                if name in names_all:
                    block_list_raw[name] = self._cache[name]
                    names_all.remove(name)
            for name in names_all:
                block_list_raw[name] = self._cache[name]

        # Turn the objects in each block into JSON strings
        block_list = [
            self._write_string_method(name)(objects)
            for name, objects in block_list_raw.items()
        ]

        # Add the header block
        block_list.insert(0, self.write_header())

        # Make the stand-alone blocks appendable
        block_list[:-1] = [re.sub(r"\n\}\Z", ", ", b) for b in block_list[:-1]]
        block_list[1:] = [re.sub(r"\A{\n", "", b) for b in block_list[1:]]

        # Return the blocks as one
        return "\n".join(block_list)

    def write_header(self):
        """Write the header information to a JSON string."""
        name = "HEADER"
        header = {name: order_dict(self._header)}
        jstr = json.dumps(header, indent=2)
        jstr = _json_remove_excessive_newlines(jstr, 2, 3)
        return jstr

    def set_config(self, **kwas):
        """Add configuration parameters to the HEADER block."""
        for key, val in kwas.items():
            if key not in self.__class__.valid_header_param_list:
                msg = (
                    "Invalid HEADER parameter '{k}'." " Valid parameters:\n {pl}"
                ).format(k=key, pl="\n ".join(self.__class__.valid_header_param_list))
                raise ValueError(msg)
            self._header[key] = val

    def _add_to_cache(self, name, objs):

        if isinstance(objs, dict):
            if name not in self._cache:
                self._cache[name] = {}
            self._cache[name].update(objs)

        else:
            # SR_TMP<
            try:
                objs = sorted(objs, key=lambda o: o.id())
            except TypeError:
                objs = sorted(objs, key=lambda o: o.id)
            # SR_TMP>

            if name not in self._cache:
                self._cache[name] = []
            self._cache[name].extend(objs)

    def write_string_objs_info(
        self, name, objs, ind=2, max_ind_lvl=3, tags=None, **kwas
    ):

        json_dict = {}

        if tags:
            json_dict[name] = {}
            for tag in tags:
                json_dict[name][tag] = []
            for obj in sorted(objs):
                tag = obj.type
                json_dict[name][tag].append(obj.get_info())
            max_ind_lvl += 2
        else:
            json_dict[name] = []
            for obj in sorted(objs):
                json_dict[name].append(obj.get_info(**kwas))

        jstr = json.dumps(json_dict, indent=ind)
        jstr = _json_remove_excessive_newlines(jstr, ind, max_ind_lvl)

        return jstr

    def add_config(self, config):
        self._add_to_cache("CONFIG", config)

    def write_string_config(self, config):
        """Write the config dict to JSON.

        The formatted JSON data is returned as a string.

        Arguments:
         - config: The config dict.
        """
        name = "CONFIG"
        jdat = {name: {}}
        for name_conf, conf in sorted(config.items()):
            log.info(
                "write config section {nsc} to {nc}".format(nc=name, nsc=name_conf)
            )
            jdat[name][name_conf] = order_dict(config[name_conf])
        jstr = json.dumps(jdat, indent=2)
        return jstr

    # Abstract methods (must be overridded by subclasses)

    def _write_string_method(self, name, register=None):
        if not register:
            err = (
                "IOWriterJsonBase._write_string_method must be overridden "
                "by subclass!"
            )
            raise NotImplementedError(err)
        for key, fct in register.items():
            if key.endswith("*"):
                if name.startswith(key[:-1]):
                    return fct
            if key == name:
                return fct
        err = "No write_string_* method found for '{}'".format(name)
        raise ValueError(err)


# OUTPUT: BINARY


class IOWriterBinaryBase:
    def __init__(self):
        pass


class FeatureTrackIOWriterBinary(IOWriterBinaryBase):
    def __init__(self):
        super().__init__()

    def write_feature_path_file(self, file_name, features):
        data = {str(feature.id()): feature.shell for feature in features}
        np.savez_compressed(file_name, **data)


# OUTPUT: PLOTTING


def plot_histogram(
    outfile,
    title,
    data,
    nbins=None,
    normalize=False,
    scale_factor=None,
    range=None,
    *kwargs,
):
    """Create a histogram plot.

    Arguments:
     - outfile: name of output file (incl. suffix)
     - title: plot title
     - data: data to plot

    Optional arguments:
     - nbins: number of bins
     - normalize: normalize y data (divide by total)
     - scale_factor: factor the histogram values are multiplied with
     - xrange: data range for x-axis
     - *kwargs: all other arguments are passed on to "bar_plot"
    """
    kwargs_hist = {}
    if nbins:
        kwargs_hist["bins"] = nbins
    if xrange:
        kwargs_hist["range"] = xrange
    hist, bins = np.histogram(data, **kwargs_hist)
    width = 0.8 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    if normalize:
        hist = hist.astype(float) / float(len(data))

    if scale_factor:
        hist = hist.astype(float) * scale_factor
        title += " (scaled by {:.3f})".format(scale_factor)

    bar_plot(outfile, title, center, hist, width=width, **kwargs)


def plot_histogram_2d(
    outfile,
    title,
    xdata,
    ydata,
    xnbins=None,
    ynbins=None,
    xrange=None,
    yrange=None,
    normalize=False,
    scale_factor=None,
    **kwargs,
):
    """Create a 2D histogram plot.

    Arguments:
     - outfile: name of output file (incl. suffix)
     - title: plot title
     - xdata: data to plot (x axis)
     - ydata: data to plot (y axis)

    Optional arguments:
     - xnbins: number of bins
     - ynbins: number of bins
     - xrange: data range for x-axis
     - yrange: data range for y-axis
     - normalize: normalize y data (divide by total)
     - scale_factor: factor the histogram values are multiplied with
     - *kwargs: all other arguments are passed on to "bar_plot"
    """
    if xrange is None:
        xrange = [xdata.min(), ydata.max()]
    if yrange is None:
        yrange = [ydata.min(), ydata.max()]
    range = [xrange, yrange]

    if xnbins is None:
        xnbins = 10
    if ynbins is None:
        ynbins = 10
    nbins = [xnbins, ynbins]

    hist, xbins, ybins = np.histogram2d(xdata, xdata, bins=nbins, yange=range)

    if normalize:
        hist = hist.astype(float) / float(len(xdata) + len(ydata))

    if scale_factor:
        hist = hist.astype(float) * scale_factor
        title += " (scaled by {:.3f})".format(scale_factor)

    color_plot(outfile, title, hist, xbins, ybins, **kwargs)


def color_plot(
    outfile,
    title,
    data,
    *,
    verbose=True,
    lon=None,
    lat=None,
    domain=None,
    xbins=None,
    ybins=None,
    zbounds=None,
    levels=None,
    zticks=None,
    zticklabels=None,
    cmap=None,
    cbar_over=False,
    cbar_under=False,
    add_colorbar=True,
    domain_boundary=None,
):

    if verbose:
        print("plot {}".format(outfile))

    if cmap is None:
        cmap = cm.Greys

    fig, ax = plt.subplots()

    extent = None
    if xbins and ybins is not None:
        extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]

    if cbar_over and cbar_under:
        extend = "both"
    elif cbar_over:
        extend = "max"
    elif cbar_under:
        extend = "min"
    else:
        extend = "neither"

    vmin, vmax = None, None
    if levels is not None:
        vmin, vmax = levels[0], levels[-1]

    if lon is None or lat is None:
        norm = None
        if zbounds:
            norm = mpl.colors.BoundaryNorm(zbounds, cmap.N)
        im = plt.imshow(
            data,
            interpolation="nearest",
            origin="low",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        ax.images.append(im)

    else:
        im = ax.contourf(
            lon,
            lat,
            data,
            levels=levels,
            cmap=cmap,
            extend=extend,
            vmin=vmin,
            vmax=vmax,
            latlon=True,
        )
        m = setup_map_crclim(lon, lat, ax=ax, map_limits=domain)

        if domain_boundary:
            px, py = domain_boundary
            ax.plot(px, py, linewidth=1, color="black")

    if add_colorbar:
        cbar = plt.colorbar(im, extend=extend)

        if zticks:
            cbar.set_ticks(zticks)
        if zticklabels:
            cbar.set_ticklabels(zticklabels)

    ax.set_aspect("auto")
    ax.set_title(title)

    fig.savefig(outfile)
    plt.close()


def bar_plot(
    outfile,
    title,
    center,
    hist,
    width=None,
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    xticks2=None,
    yticks2=None,
    xlabels2=None,
    ylabels2=None,
    xrange=None,
    yrange=None,
    xscale=None,
    yscale=None,
    verbose=True,
):
    """Create a bar plot.

    Arguments:
     - outfile: name of output file (incl. suffix)
     - title: plot title
     - center: Center points of the bars
     - hist: histogram data to plot

    Optional arguments:
     - width: width of the bars
     - plot_bars: lot the data using vertical bars or lines
     - xrange: data range for x-axis
     - xticks: list of tickmarks for bottom x-axis
     - yticks: list of tickmarks for left y-axis
     - xlabels: list of tickmark labels for bottom x-axis
     - ylabels: list of tickmark labels for left y-axis
     - xticks2: list of tickmarks for bottom x-axis
     - yticks2: list of tickmarks for right y-axis
     - xlabels2: list of tickmark labels for top x-axis
     - ylabels2: list of tickmark labels for right y-axis
     - xscale: scale of x axis
     - yscale: scale of y axis
     - verbose: verbosity switch
    """
    if verbose:
        print("plot {}".format(outfile))

    fig, ax = plt.subplots()

    ax.bar(center, hist, align="center", width=width)

    ax.set_title(title)
    ax.grid(True)

    # Set axis ranges ans scales
    if xrange:
        ax.set_xlim(xrange)
    if yrange:
        ax.set_ylim(yrange)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    # Set tick marks and labels
    if xticks:
        ax.set_xticks(xticks)
    if xlabels:
        ax.set_xticklabels(xlabels)
    if yticks:
        ax.set_yticks(yticks)
    if ylabels:
        ax.set_yticklabels(ylabels)

    # Add second x-axis on top
    if any(i for i in [xticks2, ylabels2]):
        ax2 = ax.twiny()
        ax2.set_xbound(ax.get_xbound())
        ax2.set_xlim(ax.get_xlim())
        ax2.grid(True, linestyle="-")

        if xticks2:
            ax2.set_xticks(xticks2)
        if xlabels2:
            ax2.set_xticklabels(xlabels2)

        title_pos_y = ax.title.get_position()[1]
        ax.title.set_y(title_pos_y + 0.05)

    # add second y-axis on the right
    if any(i for i in [yticks2, ylabels2]):
        raise NotImplementedError

    fig.savefig(outfile)


def xy_plot(
    outfile,
    title,
    data_x,
    data_y,
    *,
    type="scatter",
    multi=False,
    color=None,
    verbose=True,
    xlabel=None,
    ylabel=None,
    xlabel2=None,
    ylabel2=None,
    xticks=None,
    yticks=None,
    xlabels=None,
    ylabels=None,
    xticks2=None,
    yticks2=None,
    xlabels2=None,
    ylabels2=None,
    xrange=None,
    yrange=None,
    xscale=None,
    yscale=None,
):
    if verbose:
        print("plot {}".format(outfile))

    fig, ax = plt.subplots()

    if type == "scatter":
        symbol = "o"
    elif type == "line":
        symbol = ""
    else:
        err = "Invalid plot type {}".format(type)
        raise ValueError(err)

    # Plot data
    if not multi:
        data_x, data_y = [data_x], [data_y]
    for dx, dy in zip(data_x, data_y):
        p = ax.plot(dx, dy, symbol)
        if color:
            p.set_color(color)

    # Set axis ranges and scales
    if xrange:
        ax.set_xlim(xrange)
    if yrange:
        ax.set_ylim(yrange)
    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    # Set tick marks and labels
    if xticks:
        ax.set_xticks(xticks)
    if xlabels:
        ax.set_xticklabels(xlabels)
    if yticks:
        ax.set_yticks(yticks)
    if ylabels:
        ax.set_yticklabels(ylabels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add second x-axis on top
    if any(i for i in [xticks2, ylabels2]):
        ax2 = ax.twiny()
        ax2.set_xbound(ax.get_xbound())
        ax2.set_xlim(ax.get_xlim())
        ax2.grid(True, linestyle="-")

        if xticks2:
            ax2.set_xticks(xticks2)
        if xlabels2:
            ax2.set_xticklabels(xlabels2)
        if xlabel2:
            ax2.set_xlabel(xlabel2)

        title_pos_y = ax.title.get_position()[1]
        ax.title.set_y(title_pos_y + 0.05)

    # add second y-axis on the right
    if any(i for i in [yticks2, ylabels2]):
        raise NotImplementedError

    ax.set_title(title)

    fig.savefig(outfile)


def reduce_colormap(cmap, name=None, n=20, first=None, last=None):
    n_colors = 256
    indices = np.round(np.linspace(0, n_colors, n)).astype(int)
    colors = [cmap(i) for i in indices]
    if first is not None:
        colors[0] = first
    if last is not None:
        colors[-1] = last
    return mpl.colors.ListedColormap(colors, name=name)


class InFileTimestep:
    """Represents an input file at a certain timestep.

    To read a file, a timestep is passed, which is used to complete the file
    name from a template where only the timestep information is missing.
    """

    def __init__(self, tmpl, fct, ts_format="%Y%m%d%H", **kwas):
        assert isinstance(tmpl, str)
        self.tmpl = tmpl
        self.fct = fct
        self.ts_format = ts_format
        self.kwas = kwas
        self._ifile_prev = None

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.tmpl)

    def __eq__(self, other):
        if self.tmpl == other.tmpl:
            return True
        return False

    def __hash__(self):
        return id(self)

    def read_fields(self, ts, names, lon, lat, **kwas):
        ifile = self.get_infile(ts)
        if self._ifile_prev == ifile:
            return None
        log.info(
            "[{}] read {} from {}".format(
                ts.strftime(self.ts_format), ", ".join(names), ifile
            )
        )
        kwas.update(self.kwas)
        try:
            fields = self.fct(ifile, names, lon=lon, lat=lat, **kwas)
        except Exception as e:
            err = "Cannot read fields [{}] from {}: {}({})".format(
                ", ".join(names), ifile, e.__class__.__name__, e
            )
            raise IOError(err)
        self._ifile_prev = ifile
        return dict(zip(names, fields))

    def get_infile(self, ts):
        sts = ts.strftime(self.ts_format)
        yyyy, mm, dd, hh, nn = sts[:4], sts[4:6], sts[6:8], sts[8:10], sts[10:12]
        ifile = self.tmpl.format(YYYY=yyyy, MM=mm, DD=dd, HH=hh, NN=nn)
        return ifile


class InField:
    def __init__(
        self,
        name,
        ifile,
        fnames,
        pp_fct=None,
        *,
        assoc=None,
        del_old=True,
        infile_lonlat=None,
        **kwas,
    ):

        self.name = name
        self.ifile = ifile
        self.fnames = [fnames] if isinstance(fnames, str) else fnames
        self.pp_fct = pp_fct
        self.assoc = assoc

        # SR_TMP<
        if infile_lonlat is None:
            self.lon = None
            self.lat = None
        else:
            self.lon, self.lat = self._read_lonlat(infile_lonlat)
        # SR_TMP>

        self.pp_kwas = kwas
        self.del_old = del_old

        self._raw_data = None
        self._data = None

    def __repr__(self):
        return "{}({}: {}, {})".format(
            self.__class__.__name__, self.name, self.ifile, self.fnames
        )

    def __eq__(self, other):
        if (
            self.name == other.name
            and self.ifile == other.ifile
            and self.assoc == other.assoc
            and self.fnames == other.fnames
            and
            # self._raw_data == other._raw_data):
            np.array_equal(self._raw_data, other._raw_data)
        ):
            return True
        return False

    @classmethod
    def manager(cls, *args, **kwas):
        return InFieldManager(*args, **kwas)

    @classmethod
    def track(cls, *args, **kwas):
        return InField_Track(*args, **kwas)

    def _read_lonlat(self, infile):
        with nc4.Dataset(infile, "r") as fi:
            lon = fi["lon"][:]
            lat = fi["lat"][:]
        return lon, lat

    def data(self, ts):
        if not isinstance(self._data, dict):
            return self._data

        if len(self._data) == 0:
            return None

        # Make sure types of timesteps match (int/datetime)
        _some_ts = next(iter(self._data.keys()))
        if isinstance(ts, dt.datetime):
            if not isinstance(_some_ts, dt.datetime):
                # Convert datetime to int
                ts = int(ts.strftime(self.ifile.ts_format))
        elif isinstance(_some_ts, dt.datetime):
            # Convert int to datetime
            ts = dt.datetime.strptime(str(ts), self.ifile.ts_format)

        if self.del_old:
            # Remove old fields
            for key_ts in [k for k in self._data.keys()]:
                if key_ts < ts:
                    del self._data[key_ts]

        return self._data.get(ts)

    def preproc(self, timestep):
        self.pp_kwas["assoc"] = self.assoc
        self.pp_kwas["ts"] = timestep
        self.pp_kwas["fnames"] = self.fnames

        if self._data is None:
            old_data = None
        else:
            assert len(self.fnames) == 1
            old_data = self._data

        if self.pp_fct:
            # Use custom preproc function with arguments
            new_data = self.pp_fct(self._raw_data, old_data=old_data, **self.pp_kwas)
        elif self.pp_kwas:
            # Use default preproc function with arguments
            new_data = self.pp_base(self._raw_data, old_data=old_data, **self.pp_kwas)
        elif len(self._raw_data) == 1:
            # No arguments, no preproc!
            new_data = next(iter(self._raw_data.values()))
        else:
            err = (
                "For multiple fields, a preprocessing function must "
                "be provided to reduce the fields to a single field"
            )
            raise Exception(err)

        self._data = new_data

    @classmethod
    def pp_base(
        cls,
        fld,
        conversion_factor=None,
        smoothing_sigma=None,
        slice=None,
        minval=None,
        maxval=None,
        **kwas,
    ):
        """Basic pre-processing of a single field."""

        if len(fld) != 1:
            err = "pp_base can only handle one field ({} passed)".format(len(fld))
            raise ValueError(err)
        fld = next(iter(fld.values()))

        if slice:
            fld = slice(fld)

        if conversion_factor:
            fld = fld * conversion_factor

        if smoothing_sigma:
            fld = sp.ndimage.gaussian_filter(fld, sigma=smoothing_sigma, order=0)

        if minval is not None:
            fld[fld < minval] = np.nan

        if maxval is not None:
            fld[fld > maxval] = np.nan

        return fld


class InField_Track(InField):
    def __init__(self, *args, **kwas):
        kwas["del_old"] = kwas.get("del_old", True)
        super().__init__(*args, **kwas)

    def preproc(self, timestep):
        super().preproc(timestep)
        tracks, config_id, config_tracks = self._data
        ts_start = min([track.ts_start() for track in tracks])
        ts_end = max([track.ts_end() for track in tracks])
        tracks_ts = select_tracks_ts(tracks, ts_start, ts_end)
        self._data = tracks_ts


class InFieldManager:
    """Manage input fields used for plotting."""

    def __init__(self):
        self.elements = []
        self.files = {}

    def __repr__(self):
        return "{}({} files, {} elements)".format(
            self.__class__.__name__, len(self.files), len(self.elements)
        )

    def __iter__(self):
        return iter(self.elements)

    def n(self):
        return len(self.elements)

    def update(self, other):
        """Update managers with one or more others."""

        # Deal with multiple others (i.e. a list of Managers)
        # Note: sequence hard-coded as list; support for arbitrary
        # containers could be implemented in case it's ever necessary
        if isinstance(other, list):
            for other_ in other:
                self.update(other_)
                return

        # If not a list, other must be the same class as self
        elif not isinstance(other, self.__class__):
            err = "invalid class of other (must be one of [{}]): {}".format(
                ", ".join([c.__name__ for c in [self.__class__, list]]),
                other.__class__.__name__,
            )
            raise ValueError(err)

        # Update files dict and elements list
        # Note: Need to manually compare the InFileTimestep keys ('==')
        # because 'is' comparison won't work (__eq__ vs. __hash__)
        for key_other, val_other in other.files.items():
            for key_self, val_self in self.files.items():
                if key_self == key_other:
                    del self.files[key_self]
                    for vs in val_self:
                        self.elements.remove(vs[0])
                    break
            self.files[key_other] = val_other
            for vs in val_other:
                self.elements.append(vs[0])

    def add_elements(self, elements):
        for element in elements:
            self.add_element

    def add_element(self, element):
        self.elements.append(element)
        field = element.field
        if field:
            # Store input file of field
            if field.ifile:
                if field.ifile not in self.files:
                    self.files[field.ifile] = []
                self.files[field.ifile].append((element, 0))

            # Store input file of associated field
            if field.assoc:
                if field.assoc.ifile:
                    if field.assoc.ifile not in self.files:
                        self.files[field.assoc.ifile] = []
                    self.files[field.assoc.ifile].append((element, 1))

    def read_data(self, ts, lon, lat, trim_bnd_n, **kwas):
        """For every field, read the data from file."""

        # Loop over input files
        fields_priority = []
        fields_other = []
        for file, plot_elements in self.files.items():

            # Collect field names
            fnames = []
            for element, type_ in plot_elements:
                if type_ == 0:
                    fnames.extend(element.field.fnames)
                elif type_ == 1:
                    fnames.extend(element.field.assoc.fnames)
            fnames = sorted(set(fnames))

            # Read data (if it's not in memory already)
            try:
                new_data = file.read_fields(
                    ts, fnames, lon, lat, trim_bnd_n=trim_bnd_n, **kwas
                )
            except Exception as e:
                err = "error reading file {}: {}({})".format(file, type(e).__name__, e)
                raise Exception(err)
            if new_data is None:
                continue

            # Add new data
            for element, type_ in plot_elements:

                if type_ == 0:
                    field = element.field
                    if field not in fields_priority and field not in fields_other:
                        fields_other.append(field)

                elif type_ == 1:
                    field = element.field.assoc
                    if field in fields_other:
                        fields_other.remove(field)
                    if field not in fields_priority:
                        fields_priority.append(field)

                if field._raw_data is None:
                    field._raw_data = {}
                for fname in field.fnames:
                    field._raw_data[fname] = new_data[fname]

        # Post-process the new raw data
        # First, make sure all fields which are associates of others have
        # been post-processed, as these are used for post-processing those
        # fields which they are associates of
        for field in fields_priority:
            field.preproc(ts)
        for field in fields_other:
            field.preproc(ts)


class PlotElement:
    def __init__(self, name=None, field=None, **kwas):

        if "fld" in kwas:
            msg = (
                "{}: initialized with argument 'fld'; " "did you mean 'field'?"
            ).format(self.__class__.__name__)
            log.warning(msg)

        self.name = name if name else "noname"
        self.field = field

        self.pltkw = kwas.pop("pltkw", {})

        self.kwas_misc = kwas  # SR_TMP

    def __repr__(self):
        return "{}({}: {})".format(self.__class__.__name__, self.name, self.field)

    def __eq__(self, other):
        if self.name == other.name and self.field == other.field:
            return True
        return False

    @classmethod
    def contour(cls, *args, **kwas):
        return PlotElement_Contour(*args, **kwas)

    @classmethod
    def color(cls, *args, **kwas):
        return PlotElement_Color(*args, **kwas)

    @classmethod
    def shading(cls, *args, **kwas):
        return PlotElement_Shading(*args, **kwas)

    @classmethod
    def line(cls, *args, **kwas):
        return PlotElement_Line(*args, **kwas)

    @classmethod
    def feature_shading(cls, *args, **kwas):
        return PlotElement_FeatureShading(*args, **kwas)

    @classmethod
    def feature_contour(cls, *args, **kwas):
        return PlotElement_FeatureContour(*args, **kwas)

    @classmethod
    def feature_track_old(cls, *args, **kwas):
        return PlotElement_FeatureTrack_Old(*args, **kwas)

    @classmethod
    def feature_track(cls, *args, **kwas):
        return PlotElement_FeatureTrack(*args, **kwas)

    def derive(self, name=None, **kwas):
        if name is None:
            name = "derived({})".format(self.name)
        kwas_other = dict(field=self.field)
        kwas_other.update(self._derive_kwas())
        for key, val in kwas.copy().items():
            if isinstance(val, dict) and key in kwas_other:
                tmp = kwas.pop(key)
                kwas_other[key].update(val)
        kwas_other.update(self.kwas_misc)
        kwas_other.update(kwas)
        other = self.__class__(name, **kwas_other)
        return other

    def _derive_kwas(self):
        kwas = dict(
            pltkw=self.pltkw.copy(),
        )
        try:
            kwas["cbar"] = (self.cbar_kwas.copy(),)
        except AttributeError:
            pass
        return kwas


class PlotElement_Contour(PlotElement):
    def __init__(self, *args, **kwas):
        super().__init__(*args, **kwas)

    def plot(self, ax, m, mlon, mlat, ts, **kwas):
        if self.field.data(ts) is None:
            return
        if np.nansum(self.field.data(ts)) > 0:
            ax.contour(mlon, mlat, self.field.data(ts), **self.pltkw)


class PlotElement_Color(PlotElement):
    def __init__(self, *args, cbar=None, **kwas):
        if "cmap" in kwas:
            kwas["pltkw"]["cmap"] = kwas.pop("cmap")
        super().__init__(*args, **kwas)
        self.add_cbar = bool(cbar)
        try:
            self.cbar_kwas = {k: v for k, v in cbar.items()}
        except AttributeError:
            self.cbar_kwas = {}
        self.labelsize = self.cbar_kwas.pop("labelsize", 20)

        # SR_TMP<
        deprec = ["color_under", "color_over", "alpha_under", "alpha_over"]
        if any(i in kwas for i in deprec):
            err = "arguments deprecated: {}".format(deprec)
            raise ValueError(err)
        # SR_TMP>

    def plot(self, ax, m, mlon, mlat, ts, **kwas):
        if self.field.data(ts) is None:
            return
        p = ax.contourf(mlon, mlat, self.field.data(ts), **self.pltkw)
        if self.add_cbar:
            cbar = ax.figure.colorbar(p, **self.cbar_kwas)
            cbar.ax.tick_params(labelsize=self.labelsize)
            if "label" in self.cbar_kwas:
                cbar.set_label(self.cbar_kwas["label"], size=self.labelsize)
                # cbar.ax.set_xlabel(self.cbar_kwas["label"], size=self.labelsize)

    def _derive_kwas(self):
        return dict(
            cbar=self.cbar_kwas.copy(),
            pltkw=self.pltkw.copy(),
        )


class PlotElement_Shading(PlotElement):
    def __init__(self, *args, lower=None, upper=None, **kwas):
        super().__init__(*args, **kwas)
        if lower is None and upper is None:
            err = "{}: either lower or upper threshold must be passed".format(
                self.__class__.__name__
            )
            raise ValueError(err)
        self.lower = lower
        self.upper = upper

    def plot(self, ax, m, mlon, mlat, ts, **kwas):
        if self.field is None:
            err = "cannot plot {}: field is None".format(self.name)
            raise Exception(err)
        if self.field.data(ts) is None:
            return
        lower = self.field.data(ts).min() if self.lower is None else self.lower
        upper = self.field.data(ts).max() if self.upper is None else self.upper
        p = ax.contourf(
            mlon,
            mlat,
            self.field.data(ts),
            levels=[lower, upper],
            vmin=lower,
            vmax=upper,
            **self.pltkw,
        )

    def _derive_kwas(self):
        return dict(
            pltkw=self.pltkw.copy(),
        )


class PlotElement_Line(PlotElement):
    def __init__(self, *args, **kwas):
        super().__init__(*args, **kwas)

    def plot(self, ax, m, mlon, mlat, ts, **kwas):
        if self.field.data(ts) is None:
            return
        for fid, line in self.field.data(ts).items():
            px, py = line
            if m:
                px, py = m(px, py)
            ax.plot(px, py, **self.pltkw)


class PlotElement_FeatureShading(PlotElement):
    def __init__(self, *args, **kwas):
        super().__init__(*args, **kwas)

    def plot(self, ax, m, mlon, mlat, ts, **kwas):

        if self.field.data(ts) is None:
            return

        if self.pltkw.get("color") is None:
            return

        features = self.field.data(ts)
        if features is not None:
            for feature in features:
                ax_add_feature_shading(
                    ax,
                    m,
                    feature,
                    mlon=mlon,
                    mlat=mlat,
                    convert_lonlat=True,
                    **self.pltkw,
                )


class PlotElement_FeatureContour(PlotElement):
    def __init__(self, *args, cmode=None, pltkw=None, cmap=None, cbar=None, **kwas):
        super().__init__(*args, **kwas)
        self.cmode = cmode
        self.pltkw = {} if pltkw is None else pltkw
        self.cmap = cmap
        self.cbarkw = {} if cbar is None else cbar.copy()

    def plot(self, ax, m, mlon, mlat, ts, *, label_features, label_events, **kwas):

        # Prepare some parameters
        vmin = self.kwas_misc.get("vmin", 0)
        vmax = self.kwas_misc.get("vmax")
        if self.cmode is not None and not vmax:
            features = self.field.data(ts)
            if features is not None:
                if self.cmode == "size":
                    vmax = max([f.n for f in features])
                elif self.cmode == "size/log10":
                    vmax = max([np.log10(f.n) for f in features])
                else:
                    err = "cmode {}: vmax".format(self.cmode)
                    raise NotImplementedError(err)

        # Select features by timestep
        if isinstance(ts, dt.datetime):
            ts_int = int(ts.strftime(self.ts_format))
        else:
            ts_int = int(ts)

        features = self.field.data(ts)
        if features is None:
            features = []
        else:
            features = [f for f in features if f.timestep == ts_int]

        for feature in features:

            # Determine color of outline
            if self.cmode is None:
                pass
            elif self.cmode.startswith("size"):
                if not self.cmap:
                    raise Exception(
                        "{}({}): missing cmap".format(
                            self.__class__.__name__, self.name
                        )
                    )
                if self.cmode == "size":
                    n = feature.n
                elif self.cmode == "size/log10":
                    n = np.log10(feature.n)
                else:
                    raise NotImplementedError("cmode {}".format(self.cmode))
                self.pltkw["color"] = self.cmap(n / vmax)
            else:
                raise ValueError(
                    "{}: invalid cmode: {}".format(self.__class__.__name__, self.cmode)
                )

            # Plot feature
            ax_add_feature_contour(
                ax,
                m,
                feature,
                mlon=mlon,
                mlat=mlat,
                label_feature=label_features,
                label_event=label_events,
                convert_lonlat=True,
                pltkw=self.pltkw,
            )

        # Add colorbar
        if self.cbarkw:
            # SR_TMP Re-use method from PlotElement_FeatureTrack
            # SR_TODO Implement this properly (e.g. pull out a base class
            # SR_TODO "Cbar" or sth like that to share these methods)
            _color_mode_ax_add_colorbar(
                ax,
                cmap=self.cmap,
                label=self.cmode,
                vmin=vmin,
                vmax=vmax,
                **self.cbarkw,
            )

    def _derive_kwas(self):
        return dict(
            pltkw=self.pltkw.copy(),
            cmode=self.cmode,
            cmap=self.cmap,
            cbarkw=self.cbarkw.copy(),
        )


class PlotElement_FeatureTrack_Old(PlotElement):
    def __init__(self, *args, draw_features=True, draw_edges=True, _data=None, **kwas):
        super().__init__(*args, **kwas)

        self.draw_features = draw_features
        self.draw_edges = draw_edges

    def plot(self, ax, m, mlon, mlat, ts, **kwas):

        # Plot features
        if self.draw_features:
            self._plot_features(ax, m, mlon, mlat, ts)

        # Plot tracks
        if self.draw_edges:
            self._plot_edges(ax, m, mlon, mlat, ts)

    def _plot_features(self, ax, m, mlon, mlat, ts):
        keys_feat = [
            "linewidth",
            "color",
            "cmap",
            "plot_center",
            "label_feature",
            "label_event",
            "label_track",
            "scale",
            "convert_lonlat",
        ]
        kwas_feat = {k: v for k, v in self.pltkw.items() if k in keys_feat}
        kwas_feat.update({"mlon": mlon, "mlat": mlat})
        tracks = self.field.data(ts)
        if tracks is not None:
            for track in tracks:
                for feature in track.features_ts(ts):
                    ax_add_feature_contour(ax, m, feature, **kwas_feat)

    def _plot_edges(self, ax, m, mlon, mlat, ts):
        keys_track = [
            "scale",
            "edge_color",
            "convert_lonlat",
        ]
        kwas_track = {k: v for k, v in self.pltkw.items() if k in keys_track}
        kwas_track.update({"mlon": mlon, "mlat": mlat, "ts_end": ts})
        tracks = self.field.data(ts)
        if tracks is not None:
            for track in tracks:
                ax_add_track_graph_old(ax, m, track, **kwas_track)

    def _derive_kwas(self):
        return dict(
            draw_features=self.draw_features,
            draw_edges=self.draw_edges,
            pltkw=self.pltkw.copy(),
        )


class PlotElement_FeatureTrack(PlotElement):
    def __init__(
        self,
        *args,
        draw_features=True,
        draw_edges=True,
        shadingkw=None,
        colmodkw=None,
        graphkw=None,
        _data=None,
        cbar=None,
        **kwas,
    ):
        super().__init__(*args, **kwas)

        self.draw_features = draw_features
        self.draw_edges = draw_edges

        self.shadingkw = {} if shadingkw is None else shadingkw

        self.colmodkw = {} if colmodkw is None else colmodkw
        # SR_TMP< TODO group all the colmodkw setup stuff in one place
        if "cmap" in self.colmodkw:
            if isinstance(self.colmodkw["cmap"], str):
                self.colmodkw["cmap"] = mpl.cm.get_cmap(colmodkw["cmap"])
            if "cmap_under" in colmodkw:
                self.colmodkw["cmap"].set_under(colmodkw["cmap_under"])
            if "cmap_over" in colmodkw:
                self.colmodkw["cmap"].set_under(colmodkw["cmap_over"])
        # SR_TMP>

        self.graphkw = {} if graphkw is None else graphkw
        # SR_TMP< TODO group all the graphkw setup stuff in one place
        if "cmap" in self.graphkw:
            if isinstance(self.graphkw["cmap"], str):
                self.graphkw["cmap"] = mpl.cm.get_cmap(graphkw["cmap"])
            if "cmap_under" in graphkw:
                self.graphkw["cmap"].set_under(graphkw["cmap_under"])
            if "cmap_over" in graphkw:
                self.graphkw["cmap"].set_over(graphkw["cmap_over"])
        # SR_TMP>

        self.cbar_kwas = {} if cbar is None else cbar

    def _derive_kwas(self):
        return dict(
            draw_features=self.draw_features,
            draw_edges=self.draw_edges,
            pltkw=self.pltkw.copy(),
            shadingkw=self.shadingkw.copy(),
            colmodkw=self.colmodkw.copy(),
            graphkw=self.graphkw.copy(),
            cbar_kwas=self.cbar_kwas.copy(),
        )

    def plot(self, ax, m, mlon, mlat, ts, **kwas):

        # Plot features
        if self.draw_features:
            self._plot_features(ax, m, mlon, mlat, ts, **kwas)

        # Plot tracks
        if self.draw_edges:
            tracks = self.field.data(ts)
            for track in tracks if tracks is not None else []:
                ax_add_track_graph(
                    ax,
                    m,
                    track,
                    mlon=mlon,
                    mlat=mlat,
                    ts_end=ts,
                    ts_format=self.field.ifile.ts_format,
                    **self.graphkw,
                )

            # Add colorbar for edge colors
            if self.graphkw.get("edge_style", "solid") != "solid":
                if "cbar" in self.graphkw:
                    ax_add_cbar_edge(ax, **self.graphkw)

    def _plot_features(self, ax, m, mlon, mlat, ts, **kwas):
        _name_ = self.__class__.__name__ + "._plot_features"
        keys_feat = [
            "plot_center",
            "label_feature",
            "label_event",
            "label_track",
            "scale",
        ]
        kwas_feat = {k: v for k, v in self.pltkw.items() if k in keys_feat}
        kwas_feat.update({"mlon": mlon, "mlat": mlat})
        cmode_lst, kwas_cmode = self._get_color_mode()
        # SR_TODO introduce proper dicts (now all args in pltkw)
        pltkw = {
            k: v
            for k, v in self.pltkw.items()
            if k not in keys_feat and k not in kwas_cmode and k != "color_mode"
        }

        # Add feature shading and contours
        tracks = self.field.data(ts)
        for track in tracks if tracks is not None else []:
            for feature in track.features_ts(ts):

                # Add shading
                if self.shadingkw.get("color") is not None:
                    ax_add_feature_shading(
                        ax,
                        m,
                        feature,
                        convert_lonlat=True,
                        mlon=mlon,
                        mlat=mlat,
                        **self.shadingkw,
                    )

                # Add contour
                if cmode_lst and "solid" not in cmode_lst:
                    pltkw["color"] = _color_mode_get_color(
                        name=self.name,
                        feature=feature,
                        ts=ts,
                        mode=cmode_lst,
                        kwas=kwas_cmode,
                        scale=self.colmodkw.get("scale", 1),
                    )
                ax_add_feature_contour(
                    ax, m, feature, pltkw=pltkw, convert_lonlat=True, **kwas_feat
                )

        # Add colorbar for feature boundary color
        if cmode_lst and "solid" not in cmode_lst:
            if "cmap" in self.colmodkw and self.colmodkw.get("add_cbar"):
                kwas_cb = self.colmodkw.copy()
                kwas_cb.update(self.cbar_kwas)
                if "label" not in kwas_cb:
                    kwas_cb["label"] = "/".join(cmode_lst)
                _color_mode_ax_add_colorbar(ax, **kwas_cb)

    def _get_color_mode(self):
        mode = self.colmodkw.get("color_mode")
        if not mode:
            return None, {}
        if isinstance(mode, str):
            mode = mode.split("/")
        mode0, mode1 = (mode[0], None) if len(mode) == 1 else mode

        # Check mode0 TODO: give meaningful name to various mode(s) variables!
        modes0 = ["solid", "p_tot", "p_size", "p_overlap", "method", "graph_attribute"]
        if mode0 not in modes0:
            err = ("{}: invalid color mode '{}' (must be one of {}").format(
                self.name, mode0, ", ".join(modes0)
            )
            raise ValueError(err)

        kwas = {}

        # Get colormap
        cmap = self.colmodkw.get("cmap")
        if cmap is None:
            if mode0 not in ["solid"]:
                err = "{}: color mode {}: {} must be passed".format(
                    self.name, mode0, "cmap"
                )
                raise ValueError(err)
        else:
            kwas["cmap"] = cmap

        # Get vmin, vmax, levels
        levels = self.colmodkw.get("levels")
        vmin = self.colmodkw.get("vmin")
        vmax = self.colmodkw.get("vmax")
        if levels is not None:
            kwas["levels"] = levels
            kwas["vmin"] = vmin
            kwas["vmax"] = vmax
        else:
            if vmin is not None:
                kwas["vmin"] = vmin
            if vmax is None:
                if mode0 == "method":
                    err = "{}: color mode {}: {} must be passed".format(
                        self.name, mode0, "vmax"
                    )
                    raise ValueError(err)
            else:
                kwas["vmax"] = vmax

        # Get extend
        if "extend" in self.pltkw:
            kwas["extend"] = self.colmodkw["extend"]

        return mode, kwas


def _color_mode_get_color(
    mode, kwas, name="???", feature=None, track=None, ts=None, scale=1
):

    if isinstance(mode, str):
        mode = mode.split("/", 1)

    # Compatibility check
    if feature is not None:
        if mode[0] == "solid" and feature.__class__.__name__ != "Feature":
            err = "{}: {} object imcompatible with color mode {}".format(
                name, feature.__class__.__name__, mode[0]
            )
            raise ValueError(err)

    # Successor probabilities
    if mode[0] in ["p_tot", "p_size", "p_overlap"]:
        if feature is None:
            err = "must pass feature for mode '{}'".format(mode[0])
            raise ValueError(err)
        return _track_feature_probability_get_color(mode, kwas["cmap"], feature, scale)

    # Track method (e.g. duration)
    if mode[0] == "method":
        if track is None:
            if feature is None:
                err = ("must pass track or tracked feature for mode '{}'").format(
                    mode[0]
                )
                raise ValueError(err)
            track = feature.track()
        method_str = mode[1]
        # SR_TMP<
        # val = getattr(track, method_str)()
        # SR_TMP-
        if method_str == "duration":
            val = track.duration(total=True) * scale
        elif method_str == "age":
            val = track.age(ts, total=True) * scale
        else:
            val = getattr(track, method_str)() * scale
        # SR_TMP>

        levels = kwas.get("levels")
        cmap = kwas["cmap"]
        if isinstance(cmap, str):
            cmap = mpl.cm.get_cmap(cmap)
        if levels is not None:
            if val < levels[0]:
                fact = -1.0
            elif val > levels[-1]:
                fact = 2.0
            else:
                for i, lvl1 in enumerate(levels[:-1]):
                    lvl2 = kwas["levels"][i + 1]
                    if lvl1 <= val < lvl2:
                        fact = float(i) / len(levels)
                        break
                else:
                    fact = 2.0
            vmin, vmax = levels[0], levels[-1]

        else:
            vmin, vmax = kwas.get("vmin", 0), kwas["vmax"]
            fact = (val - vmin) / (vmin + vmax)

        return cmap(fact)

    err = "no color found"
    raise Exception(err)


def _color_mode_ax_add_colorbar(
    ax, cmap, label, *, vmin=0, vmax=1, levels=None, ticks=None, **kwas
):

    # SR_TODO move most options to cbar dict in config file
    if not cmap:
        return

    if levels is not None:
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    kwas_axes = dict(shrink=0.4, pad=0.04)
    kwas_cbar = dict(cmap=cmap, norm=norm)
    keys_both = ["orientation"]
    keys_axes = ["shrink", "pad"]
    keys_cbar = ["extend", "ticks"]
    keys_skip = ["labelsize"]
    for key, val in sorted(kwas.items()):
        if key in keys_both:
            kwas_cbar[key] = kwas[key]
            kwas_axes[key] = kwas[key]
        elif key in keys_cbar:
            kwas_cbar[key] = kwas[key]
        elif key in keys_axes:
            kwas_axes[key] = kwas[key]
        elif key in keys_skip:
            pass

    # SR_TMP<
    if "orientation" not in kwas_cbar:
        kwas_cbar["orientation"] = "horizontal"
        kwas_axes["orientation"] = "horizontal"
    # SR_TMP>

    cax, kw = mpl.colorbar.make_axes(ax, **kwas_axes)
    cb1 = mpl.colorbar.ColorbarBase(cax, **kwas_cbar)
    labelsize = kwas.get("labelsize", 20)
    if ticks is not None:
        cb1.set_ticks(ticks)
    cb1.set_label(label, size=labelsize)
    cb1.ax.tick_params(labelsize=labelsize)


def plot_extrema(outfile, slp, levels):
    """Plot minima and maxima on top of the SLP field."""
    log.info("create plot {n}".format(n=outfile))

    bounds = [
        (slp.lon[0, -1], slp.lat[0, -1]),  # SW edge
        (slp.lon[-1, 0], slp.lat[-1, 0]),
    ]  # NE edge

    fig, ax = plt.subplots()

    # plot SLP field
    ax.contour(slp.lon, slp.lat, slp, levels=levels, colors="k", linewidths=0.5)

    # plot extrema
    px, py = zip(*[[m.x, m.y] for m in slp.maxima()])
    ax.plot(px, py, "ro", color="g", markersize=4.0)
    px, py = zip(*[[m.x, m.y] for m in slp.minima()])
    ax.plot(px, py, "ro", color="r", markersize=4.0)

    # Domain boundary
    px, py = path_along_domain_boundary(slp.lon, slp.lat)
    ax.plot(px, py, linewidth=1, color="black")

    fig.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_cyclones(outfile, cyclones, fld, levels, stride=2, extrema=True):
    """Plot a list of Cyclone objects on top of the SLP/Z field."""
    log.info("create plot {n}".format(n=outfile))

    fig, ax = plt.subplots()
    m = setup_map_crclim(fld.lon, fld.lat, ax=ax, lw_coasts=2)
    mlon, mlat = m(fld.lon, fld.lat)

    # Plot SLP field
    ax.contour(mlon, mlat, fld, levels=levels[::stride], colors="k", linewidths=0.5)

    # Plot cyclones
    depressions = [c.as_depression() for c in cyclones]
    _plot_depressions_core(ax, depressions, m, fld.lon, fld.lat)

    if extrema:
        # Plot minima and maxima
        if fld.maxima():
            px, py = zip(*[[m.x, m.y] for m in fld.maxima()])
            ax.plot(px, py, "ro", color="g", markersize=4.0)
        if fld.minima():
            px, py = zip(*[[m.x, m.y] for m in fld.minima()])
            ax.plot(px, py, "ro", color="r", markersize=4.0)

    fig.savefig(outfile, bbox_inches="tight")
    plt.close()


def plot_depressions(outfile, clusters, slp, levels, stride=2):
    """Plot a list of Depression objects on top of the SLP field."""
    log.info("create plot {n}".format(n=outfile))

    lon, lat = slp.lon, slp.lat

    fig, ax = plt.subplots()
    m = setup_map_crclim(lon, lat, ax=ax, lw_coasts=2)
    mlon, mlat = m(lon, lat)

    # plot SLP field
    ax.contour(mlon, mlat, slp, levels=levels[::stride], colors="k", linewidths=0.5)

    _plot_depressions_core(ax, clusters, m, lon, lat)

    fig.savefig(outfile, bbox_inches="tight")
    plt.close()


def _plot_depressions_core(ax, clusters, m, lon, lat):

    # plot cluster surface as gray transparent overlay
    color = "black"
    alpha = 0.4

    for i, clust in enumerate(clusters):

        # plot enlosing cluster contour
        plon, plat = clust.contour.boundary.coords.xy
        mpx, mpy = m(plon, plat)
        ax.plot(mpx, mpy, linewidth=2, color=color)

        # plot cluster surface
        path = geo.Polygon([tuple(m(x, y)) for x, y in clust.contour.path()])
        p = PolygonPatch(path, color=color, alpha=alpha)
        ax.add_patch(p)

        # plot minima
        if len(clust.minima()) > 0:
            plon, plat = zip(*[[p.x, p.y] for p in clust.minima()])
            mpx, mpy = m(plon, plat)
            ax.plot(mpx, mpy, "ro")

    return ax


def plot_contours(
    outname,
    contours,
    points=None,
    *,
    fld=None,
    bounds=None,
    alpha=0.1,
    color="blue",
    labels=None,
):
    """Plot a list of contours."""
    outfile = outname
    log.info("create plot {n}".format(n=outfile))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # if not given, compute domain bounds (SW and NE edge coordinates)
    if bounds is None:
        bounds = _poly_bounding_box(contours, 0.1)
    bounds = np.array(bounds)

    # plot points labels
    if points and labels:
        box = dict(boxstyle="square, pad=0.2", fc="w", ec="r")
        for point in points:
            ax.annotate(
                labels(point),
                bbox=box,
                size=9,
                xy=(point.x, point.y),
                xycoords="data",
                xytext=(-5, 7),
                textcoords="offset points",
                horizontalalignment="right",
            )

    # plot contours
    for cont in contours:

        # plot contour surface
        patch = PolygonPatch(cont, alpha=alpha, color=color)
        ax.add_patch(patch)

        # plot contour outline
        px, py = cont.exterior.coords.xy
        ax.plot(px, py, color="k", linewidth=1)

        # plot contour level labels
        if labels:
            lx, ly = int(len(px) / 6.0), int(len(py) / 6.0)
            box = dict(boxstyle="square, pad=0.2", fc="w", ec="b")
            try:
                ax.annotate(labels(cont), xy=(px[lx], py[ly]), bbox=box, size=9)
            except AttributeError as e:
                # In absence of label proberty, skip the labels
                # (the case if a raw shapely Polygon has been passed)
                log.warning(
                    (
                        "plot_contours: retrieving label from object "
                        "{} failed with error: {}"
                    ).format(cont, e)
                )
                pass

    # plot points
    if points:
        px, py = list(zip(*[[m.x, m.y] for m in points]))
        ax.plot(px, py, "ro")

    # Domain boundary
    if fld is not None:
        px, py = path_along_domain_boundary(fld.lon, fld.lat)
        ax.plot(px, py, linewidth=1, color="black")

    # Axes ranges
    ax.set_xlim(*bounds[:, 0])
    ax.set_ylim(*bounds[:, 1])
    # ax.set_aspect(1)

    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def _poly_bounding_box(poly, fact=None):
    """Get the bounding box of a list of polygons.

    If <fact> is given (0..1), the box is increased accordingly.
    A tuple of the coordinates of the lower-left and upper-right points
    is returned.

    The tuple can for instance be used to set the axis ranges for a plot:

      bnd = poly_bounding_box(contours, 0.1)
      ...
      ax.set_xlim(*bnd[:, 0])
      ax.set_ylim(*bnd[:, 1])
    """
    bnd_lst = list(zip(*[p.bounds for p in poly]))
    bnd = np.array(
        [[min(bnd_lst[0]), min(bnd_lst[1])], [max(bnd_lst[2]), max(bnd_lst[3])]]
    )
    if fact:
        # increase domain by factor <fact> (if given)
        len = np.array([(bnd[1, 0] - bnd[0, 0]), (bnd[1, 1] - bnd[0, 1])])
        bnd += 0.1 * np.array([-len, len])
    return bnd


def plot_tracks(outfile, title, tracks, xlim, ylim, domain_boundary=None):
    """Plot all tracks in one plot."""
    log.info("plot {}".format(outfile))
    fig, ax = plt.subplots()

    m = lambda x, y: (x, y)

    for track in tracks:
        ax_add_track_graph(ax, m, track)

    if domain_boundary:
        ax_add_domain_boundary(ax, m, domain_boundary)

    plt.axis("scaled")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.set_size_inches(3 * 18.5, 3 * 10.5)
    fig.savefig(outfile, bbox_inches="tight", dpi=100)

    plt.close()


def plot_over_time(
    lon,
    lat,
    outfile_fmt,
    mk_outdir=True,
    title=None,
    skip_title=False,
    timesteps=None,
    ts_dt_fmt="%Y%m%d%H",
    tss_extend=None,
    ts_start_data=None,
    ts_end_data=None,
    nts_add_end=0,
    track_ids=None,
    trim_bnd_n=0,
    label_features=False,
    label_events=False,
    label_tracks=False,
    scale=1,
    fields=None,
    map=True,
    map_limits=None,
    mapfigkw=None,
    coastline_width=4,
    track_convert_to_lonlat=False,
    parallelize=False,
    num_procs=8,
):
    """Plot all features with associated tracks at every timestep.

    All features of a given timestep are plotted, along with the full
    associated track up to that timestep.

    In addition, recently finished tracks are shown, which should allow
    to judge whether a track has been rightly or mistakenly finished when
    clicking trough a time series of these plots.
    """
    print("plot tracks over time...")

    # Check some input arguments
    if tss_extend is not None:
        try:
            _, _ = tss_extend
        except:
            err = ("tss_extend must be a two-element iterable, not {}").format(
                tss_extend
            )
            raise ValueError(err) from None

    # Convert int timesteps to datetime objects
    timesteps = [
        (
            ts
            if isinstance(ts, dt.datetime)
            else dt.datetime.strptime(str(ts), ts_dt_fmt)
        )
        for ts in timesteps
    ]
    dts = None if len(timesteps) < 2 else timesteps[1] - timesteps[0]
    if ts_start_data is None:
        ts_start_data = min(timesteps)
    elif not isinstance(ts_start_data, dt.datetime):
        ts_start_data = dt.datetime.strptime(str(ts_start_data), ts_dt_fmt)
    if ts_end_data is None:
        ts_end_data = max(timesteps)
    elif not isinstance(ts_end_data, dt.datetime):
        ts_end_data = dt.datetime.strptime(str(ts_end_data), ts_dt_fmt)

    # -- Run sequentially or in parallel

    args_base = [
        lon,
        lat,
        outfile_fmt,
        mk_outdir,
        title,
        skip_title,
        track_ids,
        map_limits,
        trim_bnd_n,
        label_features,
        label_events,
        label_tracks,
        scale,
        fields,
        map,
        mapfigkw,
        coastline_width,
        track_convert_to_lonlat,
        ts_dt_fmt,
        tss_extend,
        dts,
    ]

    if parallelize:
        fct = _plot_over_time_par
        args = args_base + [ts_start_data, ts_end_data, timesteps, num_procs]

    else:
        if dts is None:
            err = "dts is None ({} timesteps)".format(len(timesteps))
            raise Exception(err)
        ts_start_read, ts_end_read = _get_ts_start_end(
            timesteps, dts, ts_start_data, ts_end_data, tss_extend
        )

        fct = _plot_over_time_core
        args = args_base + [ts_start_read, ts_end_read, timesteps]

    fct(*args)


def _plot_over_time_par(
    lon,
    lat,
    outfile_fmt,
    mk_outdir,
    title,
    skip_title,
    track_ids,
    map_limits,
    trim_bnd_n,
    label_features,
    label_events,
    label_tracks,
    scale,
    fields,
    map,
    mapfigkw,
    coastline_width,
    track_convert_to_lonlat,
    ts_dt_fmt,
    tss_extend,
    dts,
    ts_start_data,
    ts_end_data,
    timesteps,
    num_procs,
):

    # Sort timesteps into chunks such that each process gets
    # a continuous set of timesteps to optimize track input
    # (then each only needs to rebuild a small set of features)
    n_ts_tot = len(timesteps)
    n_chunk = int(np.ceil(n_ts_tot / num_procs))
    chunks_tss = []
    for i, ts in enumerate(timesteps):
        if i % n_chunk == 0:
            chunks_tss.append([])
        chunks_tss[-1].append(ts)
    chunks = []
    for tss in chunks_tss:
        ts_start_read, ts_end_read = _get_ts_start_end(
            tss, dts, ts_start_data, ts_end_data, tss_extend
        )
        chunks.append((ts_start_read, ts_end_read, tss))
    print("\n" + ("+" * 40))
    print("parallelize {} timesteps with {} threads".format(n_ts_tot, num_procs))
    print("nts: {}".format(" ".join(["{:7,}".format(len(c[2])) for c in chunks])))
    print(("+" * 40) + "\n")

    # Run in parallel
    fct = functools.partial(
        _plot_over_time_core,
        lon,
        lat,
        outfile_fmt,
        mk_outdir,
        title,
        skip_title,
        track_ids,
        map_limits,
        trim_bnd_n,
        label_features,
        label_events,
        label_tracks,
        scale,
        fields,
        map,
        mapfigkw,
        coastline_width,
        track_convert_to_lonlat,
        ts_dt_fmt,
        tss_extend,
        dts,
    )
    pool = Pool(num_procs, maxtasksperchild=1)
    pool.starmap(fct, chunks)


def _get_ts_start_end(timesteps, dts, ts_start_data, ts_end_data, tss_extend):

    ts_start = min(timesteps)
    ts_end = max(timesteps)

    if tss_extend is not None:
        ts_start -= dts * tss_extend[0]
        ts_end += dts * tss_extend[1]

    ts_start = max(ts_start_data, ts_start)
    ts_end = min(ts_end_data, ts_end)

    return ts_start, ts_end


def _plot_over_time_core(
    lon,
    lat,
    outfile_fmt,
    mk_outdir,
    title,
    skip_title,
    track_ids,
    map_limits,
    trim_bnd_n,
    label_features,
    label_events,
    label_tracks,
    scale,
    fields,
    map,
    mapfigkw,
    coastline_width,
    track_convert_to_lonlat,
    ts_dt_fmt,
    tss_extend,
    dts,
    ts_start_read,
    ts_end_read,
    timesteps_plot,
):

    # SR_TMP<
    if not any(i in outfile_fmt for i in ["{YYYY}", "{MM}", "{DD}", "{HH}", "{NN}"]):
        err = "outfile: no {YYYY} etc. to insert timestep: " + outfile_fmt
        raise ValueError(err)
    # SR_TMP>

    if fields:
        # Read data between ts_start_read and ts_end_read not in timesteps
        # Note: might (!) fail for ts_end_read > max(timesteps) due to merging
        ts = ts_start_read
        while ts <= ts_end_read:
            if ts not in timesteps_plot:
                fields.read_data(
                    ts,
                    lon,
                    lat,
                    trim_bnd_n,
                    ts_dt_fmt=ts_dt_fmt,
                    ts_start=ts_start_read,
                    ts_end=ts_end_read,
                    dts=dts,
                )
            ts += dts

    # Save plot at every timestep
    for ts in timesteps_plot:
        ts_str = ts2str(ts, ts_dt_fmt)

        if fields:
            fields.read_data(
                ts,
                lon,
                lat,
                trim_bnd_n,
                ts_dt_fmt=ts_dt_fmt,
                ts_start=ts_start_read,
                ts_end=ts_end_read,
                dts=dts,
            )

        outfile = outfile_fmt.format(
            YYYY=ts_str[:4],
            MM=ts_str[4:6],
            DD=ts_str[6:8],
            HH=ts_str[8:10],
            NN=ts_str[10:12],
        )
        if mk_outdir:
            # Create output directory if it doesn't exist yet
            outdir = os.path.dirname(outfile)
            os.makedirs(outdir, exist_ok=True)

        if skip_title:
            title_ts = None
        else:
            if title is not None and any(
                i in title for i in ["{YYYY}", "{MM}", "{DD}", "{HH}", "{NN}"]
            ):
                title_ts = title.format(
                    YYYY=ts_str[:4],
                    MM=ts_str[4:6],
                    DD=ts_str[6:8],
                    HH=ts_str[8:10],
                    NN=ts_str[10:12],
                )
            else:
                title_ts = "" if title is None else "{} ".format(title)
                title_ts = "{} [{}]".format(title, ts_str)

        _plot_over_time_core_ts(
            outfile,
            title_ts,
            ts,
            dts,
            lon,
            lat,
            fields=fields,
            label_features=label_features,
            label_events=label_events,
            label_tracks=label_tracks,
            scale=scale,
            map=map,
            map_limits=map_limits,
            mapfigkw=mapfigkw,
            coastline_width=coastline_width,
            ts_dt_fmt=ts_dt_fmt,
            track_convert_to_lonlat=track_convert_to_lonlat,
        )

    del fields


def _plot_over_time_core_ts(
    outfile,
    title,
    ts,
    dts,
    lon,
    lat,
    *,
    fields,
    label_features,
    label_events,
    label_tracks,
    scale,
    map,
    map_limits,
    mapfigkw,
    coastline_width,
    ts_dt_fmt,
    track_convert_to_lonlat,
):

    log.info("[{}] plot {}".format(ts2str(ts, ts_dt_fmt), outfile))

    # SR_TMP< TODO rename fields to elements
    elements = fields
    fields = None
    # SR_TMP>

    if elements is None:
        elements = []

    fig, ax = plt.subplots()

    # Set up map
    if map:
        m = setup_map_crclim(
            lon,
            lat,
            ax=ax,
            map_limits=map_limits,
            mapfigkw=mapfigkw,
            lw_coasts=coastline_width,
        )
        mlon, mlat = m(lon, lat)
    else:
        m = lambda x, y: (x, y)
        mlon, mlat = lon, lat

    # Plot elements
    for element in elements:
        # SR_TMP<
        if element.field.lon is None:
            element.field.lon = lon
            element.field.lat = lat
        # SR_TMP>
        try:
            # SR_TMP<
            if map:
                mlon, mlat = m(element.field.lon, element.field.lat)
            else:
                mlon, mlat = element.field.lon, element.field.lat
            # SR_TMP>
            element.plot(
                ax,
                m,
                mlon,
                mlat,
                ts,
                dts=dts,
                # SR_TODO< Remove from here (move into config)
                label_features=label_features,
                label_events=label_events,
                # SR_TODO>
                lon=element.field.lon,
                lat=element.field.lat,
            )
        except Exception as e:
            err = ("cannot plot element '{}' (field '{}'): {}({})").format(
                element.name, element.field.name, type(e).__name__, e
            )
            raise Exception(err)
            # log.warning(err)

    if title is not None:
        ax.set_title(title, fontsize=24, y=1.02)
    fig.set_size_inches(2 * 18.5, 2 * 10.5)
    fig.savefig(outfile, bbox_inches="tight")

    fig.clf()
    plt.close()


def select_tracks_ts(
    tracks, ts_start, ts_end, dts, track_ids=None, ts_format="%Y%m%d%H"
):

    tracks_ts = {}

    if track_ids is None:

        # Check for some required argument for the case of no tracks
        required_args = ["ts_start", "ts_end", "dts"]
        locals_ = locals()
        if any(locals_[a] is None for a in required_args):
            err = (
                "if no tracks are given, all of the following variables "
                "are required: {}"
            ).format(", ".join(required_args))
            raise ValueError(err)
        selected_tracks = tracks

    else:
        # If a list if track IDs has been passed, restrict plotting to those
        if track_ids is not None:
            selected_tracks = [track for track in tracks if track.id() in track_ids]
        else:
            selected_tracks = [track for track in tracks]

        # Determine timestep range of tracks (if not already given)
        if ts_start is None:
            ts_start = min([track.ts_start() for track in selected_tracks])
        if ts_end is None:
            ts_end = max([track.ts_end() for track in selected_tracks])
        if ts_end < ts_start:
            err = "Invalid timestep range (ts_end < ts_start): {} < {}".format(
                ts_end, ts_start
            )
            raise Exception(err)
        nts = round((ts_end - ts_start) / dts + 1)
        if isinstance(ts_start, int) and ts_start < 0:
            raise Exception("invalid timestep: {}".format(ts_start))

    # Rather dirty check for timestep format in tracks (convert to datetime)
    if isinstance(selected_tracks[0].ts_start(), dt.datetime):
        f = lambda ts_int: dt.datetime.strptime(str(ts_int), ts_format)
    else:
        f = lambda ts_dt: ts_dt

    # Determine all active tracks at every timestep
    ts = ts_start
    while ts <= ts_end:
        tracks_ts[ts] = [
            track
            for track in selected_tracks
            if f(track.ts_start()) <= ts <= f(track.ts_end())
        ]
        ts += dts

    return tracks_ts


def ts2str(ts, ts_dt_fmt, zeropad=None):
    try:
        return ts.strftime(ts_dt_fmt)
    except AttributeError:
        return "{:05}".format(ts)


def ax_add_feature_shading(
    ax,
    m,
    feature,
    *,
    convert_lonlat=False,
    mlon=None,
    mlat=None,
    color="gray",
    alpha=0.5,
    **kwas,
):

    # SR_TMP<
    if not convert_lonlat:
        raise NotImplementedError("convert_lonlat=False (deprecated)")
    # SR_TMP>

    levels = [0.9, 1.1]
    cmap = mpl.colors.ListedColormap([color], "feature mask")

    try:
        mask = feature.to_mask(*mlon.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MaskedArrayFutureWarning)
            ax.contourf(mlon, mlat, mask, levels=levels, cmap=cmap, alpha=alpha)

    except Exception as e:
        err = ("Cannot add Feature {} to shading plot: {}({})").format(
            feature.id, e.__class__.__name__, e
        )
        raise Exception(err)


def ax_add_feature_contour(
    ax,
    m,
    feature,
    *,
    mlon=None,
    mlat=None,
    convert_lonlat=False,
    plot_center=True,
    cmap=None,
    scale=1,
    label_feature=False,
    label_event=False,
    label_track=False,
    pltkw=None,
):
    if convert_lonlat and (mlon is None or mlat is None):
        err = "mlon and mlat must be passed for convert_lonlat=True"
        raise ValueError(err)

    if feature.is_mean():
        return

    if pltkw is None:
        pltkw = {}

    # SR_TMP<
    if feature.is_periodic():
        for subfeature in feature.features():
            pltkw_tmp = pltkw.copy()
            pltkw_tmp["color"] = "green"
            ax_add_feature_contour(
                ax,
                m,
                subfeature,
                mlon=mlon,
                mlat=mlat,
                plot_center=False,
                scale=scale,
                pltkw=pltkw,
            )
            if plot_center:
                ax_add_center(ax, m, feature, label_feature, label_event, label_track)
            return
    # SR_TMP>

    if not pltkw.get("linewidth"):
        pltkw["linewidth"] = 3 * np.sqrt(scale)

    # Add feature contour
    for path in feature.shells + feature.holes:
        px, py = path.T
        if convert_lonlat:
            px, py = inds2lonlat(px, py, mlon, mlat)
        elif m:
            px, py = m(px, py)
        p = ax.plot(px, py, **pltkw)
    if plot_center:
        ax_add_center(ax, m, feature, label_feature, label_event, label_track)


def _track_feature_probability_get_color(color_mode, cmap, feature, scale=1):
    graph = feature.track.graph
    neighbors_fw = [
        n for n in feature.vertex.neighbors() if n["ts"] > feature.vertex["ts"]
    ]
    neighbors_bw = [
        n for n in feature.vertex.neighbors() if n["ts"] < feature.vertex["ts"]
    ]
    edges_fw = [
        feature.track.graph.es.find(
            _between=((feature.vertex.index,), (neighbor.index,))
        )
        for neighbor in neighbors_fw
    ]
    edges_bw = [
        feature.track.graph.es.find(
            _between=((feature.vertex.index,), (neighbor.index,))
        )
        for neighbor in neighbors_bw
    ]

    # Fetch probabilities
    p_fw, p_bw = 0.0, 0.0
    if len(edges_fw) > 0:
        p_fw = np.mean([e[color_mode[0]] for e in edges_fw]) * scale
    if len(edges_bw) > 0:
        p_bw = np.mean([e[color_mode[0]] for e in edges_bw]) * scale
    if color_mode[1] == "mean":
        p_eff = np.mean([p_fw, p_bw]) * scale
    elif color_mode[1] == "min":
        p_eff = np.min([p_fw, p_bw]) * scale
    elif color_mode[1] == "max":
        p_eff = np.max([p_fw, p_bw]) * scale

    # Pick color
    if color_mode[1] == "fw":
        color = cmap(p_fw)
    elif color_mode[1] == "bw":
        color = cmap(p_bw)
    elif color_mode[1] in ["mean", "min", "max"]:
        if any(i in feature.vertex["type"] for i in ["start", "genesis"]):
            color = cmap(p_fw)
        elif any(i in feature.vertex["type"] for i in ["stop", "lysis"]):
            color = cmap(p_bw)
        else:
            color = cmap(p_eff)
    else:
        raise NotImplementedError("/".join(color_mode))

    return color


def ax_add_center(ax, m, feature, label_feature, label_event, label_track):

    # SR_TMP<
    try:
        x, y = feature.center()
    except TypeError:
        x, y = feature.center
    # SR_TMP>

    if m:
        x, y = m(x, y)
    ax.scatter(x, y, color="black", s=150, marker="x", linewidth=3)

    bbox_props = dict(color="w", alpha=0.5)

    if label_feature:
        px, py = np.array([x, y]) + [-6.5, 1.5]
        txt = "{:>6}".format(feature.id_str())
        ax.text(px, py, txt, bbox=bbox_props)

    if label_event:
        px, py = np.array([x, y]) + [1, 1.5]
        txt = "{:<6}".format(feature.event().id())
        ax.text(px, py, txt, bbox=bbox_props)

    if label_track:
        px, py = np.array([x, y]) + [1, -2.5]
        txt = "{:<6}".format(feature.event().track().id())
        ax.text(px, py, txt, bbox=bbox_props)


def ax_add_domain_boundary(ax, m, domain_boundary):
    px, py = domain_boundary
    if m:
        px, py = m(px, py)
    ax.plot(px, py, linewidth=4, color="black")


def ax_add_track_graph(
    ax,
    m,
    track,
    mlon,
    mlat,
    ts_start=None,
    ts_end=None,
    ts_format=None,
    plot_vertices=True,
    plot_edges=True,
    **graphkw,
):
    """Plot edges and vertices of a track graph."""

    # Select vertices and edges
    vs, es = track.graph_vs_es_ts(ts_start, ts_end)

    # Determine opacity by timestep
    timesteps = sorted(set(vs["ts"]))
    if "fade_out" not in graphkw:
        alphas_ts = {ts: 1.0 for ts in timesteps}
    else:
        n_fade = graphkw["fade_out"]
        if n_fade == 0:
            alphas_ts = {timesteps[-1]: 1.0}
        else:
            delta = 1.0 / n_fade
            alphas = np.arange(1, 0, -delta)
            alphas_ts = {ts: a for ts, a in zip(timesteps[::-1], alphas)}

    # Remove invisible vertices/edges
    vs = [vx for vx in vs if vx["ts"] in alphas_ts]
    es = [eg for eg in es if track.graph.vs[eg.source]["ts"] in alphas_ts]

    # SR_TMP<
    markeredgewidth_ts_end = graphkw.get("markeredgewidth", 1)
    # SR_TMP>

    # SR_TMP<
    def ts_equal(ts0, ts1):
        if not isinstance(ts0, int):
            ts0 = int(ts0.strftime(ts_format))
        if not isinstance(ts1, int):
            ts1 = int(ts1.strftime(ts_format))
        return ts0 == ts1

    # SR_TMP>

    if plot_vertices:
        # Plot vertices
        vs_plotted = set()
        for vx in sorted(vs, key=lambda vx: vx["ts"]):
            if vx.index not in vs_plotted:
                ts = vx["ts"]
                # SR_TMP<
                if ts_equal(ts, ts_end):
                    graphkw["markeredgewidth"] = markeredgewidth_ts_end
                else:
                    graphkw["markeredgewidth"] = 0.0
                # SR_TMP>
                # SR_TMP<
                if graphkw.get("event_marker_style") == "solid":
                    graphkw["color"] = _color_mode_get_color(
                        track=track,
                        ts=ts,
                        mode=graphkw["edge_mode"],
                        kwas=graphkw["colmodkw"],
                        scale=graphkw["colmodkw"]["scale"],
                    )
                # SR_TMP>
                vs_plotted.add(vx.index)
                alpha = alphas_ts[ts]
                ax_add_track_vertex(ax, vx, mlon, mlat, alpha, **graphkw)

    # SR_TMP<
    graphkw["markeredgewidth"] = markeredgewidth_ts_end
    # SR_TMP>

    if plot_edges:
        # Plot edges
        for eg in es:
            ts = track.graph.vs[eg.target]["ts"]
            # SR_TMP<
            if graphkw["edge_style"] == "solid":
                color = _color_mode_get_color(
                    track=track,
                    ts=ts,
                    mode=graphkw["edge_mode"],
                    kwas=graphkw["colmodkw"],
                    scale=graphkw["scale"],
                )
                graphkw["edge_color"] = color
            # SR_TMP>
            alpha = alphas_ts[ts]
            ax_add_track_edge(ax, eg, mlon, mlat, alpha, **graphkw)


def ax_add_cbar_edge(ax, **graphkw):
    levels = np.arange(0, 1.001, 0.05)
    ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
    _color_mode_ax_add_colorbar(
        ax, graphkw["cmap"], levels=levels, ticks=ticks, **graphkw["cbar"]
    )


def ax_add_track_edge(
    ax,
    eg,
    mlon,
    mlat,
    alpha,
    edge_style="solid",
    edge_color="k",
    scale=1,
    cmap=None,
    **graphkw_rest,
):

    # Determine edge style
    kw = dict(lw=5 * scale, alpha=alpha)
    if edge_style == "solid":
        kw["c"] = edge_color
    elif edge_style in ["p_overlap", "p_size", "p_tot"]:
        if cmap is None:
            err = "edge style {}: missing cmap".format(edge_style)
            raise Exception(err)
        kw["c"] = cmap(eg[edge_style])
    else:
        err = "unsupported edge style: {}".format(edge_style)
        raise NotImplementedError(err)

    # Plot edge
    vx_source = eg.graph.vs[eg.source]
    vx_target = eg.graph.vs[eg.target]
    x0, y0 = vx_source["feature"].center
    x1, y1 = vx_target["feature"].center
    if any(i < 0 for i in [x0, y0, x1, y1]):
        fid0, fid1 = vx_source["feature"].id, vx_target["feature"].id
        err = ("invalid center coordinates: ({}, {}) [{}], ({}, {}) [{}]").format(
            x0, y0, fid0, x1, y1, fid1
        )
        raise Exception(err)
    px, py = np.array(
        [
            [mlon[x0, y0], mlat[x0, y0]],
            [mlon[x1, y1], mlat[x1, y1]],
        ]
    ).T
    ax.plot(px, py, **kw)


def ax_add_track_vertex(
    ax,
    vx,
    mlon,
    mlat,
    alpha,
    scale=1,
    color=None,
    marker=None,
    markersize=None,
    markeredgecolor="black",
    markeredgewidth=1,
    **graphkw_rest,
):
    type = vx["type"]

    # Color the node according to type
    kw = dict(
        marker="o",
        markersize=60 * scale,
        alpha=alpha,
        markeredgecolor=markeredgecolor,
        markeredgewidth=scale * markeredgewidth,
    )

    # Marker color
    if color is None:
        if ("start" in type or "genesis" in type) and ("end" in type or "stop" in type):
            color = "yellow"
        elif "start" in type or "genesis" in type:
            color = "green"
        elif "stop" in type or "lysis" in type:
            color = "red"
        elif "continuation" in type:
            color = "blue"
        elif "merging" in type and "splitting" in type:
            color = "purple"
        elif "merging" in type:
            color = "orange"
        elif "splitting" in type:
            color = "lime"
        else:
            err = "set color: not implemented vertex type: {}".format(type)
            raise NotImplementedError(err)
    kw["color"] = color

    # Marker type
    if marker is not None:
        if markersize is None:
            markersize = 15 * scale
    else:
        if ("start" in type or "genesis" in type) and (
            "stop" in type or "lysis" in type
        ):
            marker = "p"  # pentagon
            if markersize is None:
                markersize = 15 * scale
        elif "start" in type or "genesis" in type:
            marker = "*"  # star
            if markersize is None:
                markersize = 20 * scale
        elif "stop" in type or "lysis" in type:
            hw = 0.25  # half-width
            of = 0.15  # y-offset of center
            marker = [
                (hw, -1.0),
                (hw, of - hw),
                (1 - of - hw, of - hw),
                (1 - of - hw, of + hw),
                (0.1, of + hw),
                (hw, 1.0),
                (-hw, 1.0),
                (-hw, of + hw),
                (-1 + of + hw, of + hw),
                (-1 + of + hw, of - hw),
                (-hw, of - hw),
                (-hw, -1.0),
                (hw, -1.0),
            ]  # cross (t)
            if markersize is None:
                markersize = 15 * scale
        elif "continuation" in type:
            marker = "."  # point
            if markersize is None:
                markersize = 15 * scale
        elif "merging" in type and "splitting" in type:
            marker = "D"  # diamond
            if markersize is None:
                markersize = 10 * scale
        elif "merging" in type:
            marker = ">"  # triangle
            if markersize is None:
                markersize = 12 * scale
        elif "splitting" in type:
            marker = "<"  # triangle
            if markersize is None:
                markersize = 12 * scale
        else:
            err = "set marker: not implemented vertex type: {}".format(type)
            raise NotImplementedError(err)
    kw["marker"] = marker
    kw["markersize"] = markersize

    # Plot center
    x, y = vx["feature"].center
    ax.plot(mlon[x, y], mlat[x, y], **kw)


def ax_add_track_graph_old(ax, m, track, **kwas):
    _ax_add_track_graph_old_rec(
        ax,
        m,
        events=track.starts(),
        prev=None,
        mergings=set(),
        mergings_unplotted=set(),
        **kwas,
    )


def _ax_add_track_graph_old_rec(
    ax,
    m,
    events,
    prev,
    mergings,
    mergings_unplotted,
    # ts_stop, *, scale=1, mlat=None, mlon=None,
    ts_stop,
    scale,
    *,
    mlat=None,
    mlon=None,
    convert_lonlat=False,
    edge_color="k",
):

    if convert_lonlat and (mlon is None or mlat is None):
        err = (
            "_ax_add_track_graph_old_rec: for convert_lonlat=True both lon "
            "and lat must be given!"
        )
        raise ValueError(err)

    # Check whether we're coming from a merging to continue only once
    if prev in mergings:
        if prev in mergings_unplotted:
            mergings_unplotted.remove(prev)
        else:
            return

    for this in events:

        if ts_stop is not None and this.timestep() > ts_stop:
            continue

        # If we're at a merging, make sure only to continue once
        if this.is_merging() and this not in mergings:
            mergings.add(this)
            mergings_unplotted.add(this)

        # Plot line between previous and current feature
        if prev:
            x0, y0 = prev.feature().center()
            x1, y1 = this.feature().center()

            # Deal with tracks across the periodic boundary
            dist0 = prev.feature().center_distance(this.feature())
            dist1 = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

            # If the true distance between the features (which considers
            # the periodic boundary) is smaller than the distance
            # calculated across the domain (which does not), the section
            # of the track crosses the boundary
            linewidth = 2 * np.sqrt(scale)
            if np.isclose(dist0, dist1):

                # Connect the features inside the domain
                px, py = (x0, x1), (y0, y1)
                if convert_lonlat:
                    px, py = inds2lonlat(px, py, mlon, mlat)
                elif m:
                    px, py = m(px, py)
                ax.plot(px, py, color=edge_color, linewidth=linewidth)

            else:
                # Determine the relative position of the features
                if x0 < x1:
                    left, right = prev.feature(), this.feature()
                else:
                    left, right = this.feature(), prev.feature()

                # Compute the points where the track cuts the boundary
                lon0, lon1 = left.domain().lon0(), left.domain().lon1()
                (lx, ly), (rx, ry) = left.center(), right.center()
                dlx, drx = (lx - lon0), (lon1 - rx)
                dly = (ly - ry) * dlx / (dlx + drx)
                dry = (ly - ry) * drx / (dlx + drx)
                plx, ply = (lx - dlx), (ly - dly)
                prx, pry = (rx + drx), (ry + dry)

                # Connect the features across the periodic boundary
                px, py = (plx, lx), (ply, ly)
                if convert_lonlat:
                    px, py = inds2lonlat(px, py, mlon, mlat)
                    # if m:
                    px, py = m(px, py)
                ax.plot(px, py, color=edge_color, linewidth=linewidth)
                px, py = (rx, prx), (ry, pry)
                if convert_lonlat:
                    px, py = inds2lonlat(px, py, mlon, mlat)
                elif m:
                    px, py = m(px, py)
                ax.plot(px, py, color="k", linewidth=linewidth)

        # Color the node according to type
        kw = {"marker": "o", "s": 60 * scale}

        # Marker color
        if this.is_start() and this.is_end():
            kw["color"] = "yellow"
        elif this.is_start():
            kw["color"] = "green"
        elif this.is_end():
            kw["color"] = "red"
        elif this.is_continuation():
            kw["color"] = "blue"
        elif this.is_merging() and this.is_splitting():
            kw["color"] = "purple"
        elif this.is_merging():
            kw["color"] = "orange"
        elif this.is_splitting():
            kw["color"] = "lime"
        else:
            cls = this.__class__.__name__
            err = "COLOR: unconsidered: {}".format(cls)
            raise Exception(err)

        # Marker type
        if this.is_stop():
            kw["marker"] = "x"
            kw["linewidth"] = 3 * np.sqrt(scale)
            kw["s"] = 150 * scale
        elif this.is_continuation():
            kw["s"] = 10 * scale
        elif this.is_merging() or this.is_splitting():
            kw["marker"] = (3, 0, 0)
            kw["s"] = 160 * scale
        elif this.is_genesis():
            kw["marker"] = "*"
            kw["s"] = 100 * scale
        elif this.is_lysis():
            kw["marker"] = "+"
            kw["linewidth"] = 2 * np.sqrt(scale)
            kw["s"] = 80 * scale
        else:
            cls = this.__class__.__name__
            err = "MARKER: unconsidered: {}".format(cls)
            log.warning(err)

        if kw:
            x, y = this.feature().center()
            px, py = [x], [y]
            if convert_lonlat:
                px, py = inds2lonlat(px, py, mlon, mlat)
            elif m:
                px, py = m(px, py)
            ax.scatter(px, py, **kw)

        # Continue along branch unless its end has been reached
        if not this.is_end():
            _ax_add_track_graph_old_rec(
                ax,
                m,
                this.next(),
                this,
                mergings,
                mergings_unplotted,
                ts_stop,
                scale,
                edge_color=edge_color,
                convert_lonlat=convert_lonlat,
                mlon=mlon,
                mlat=mlat,
            )


def setup_map_crclim(
    lon,
    lat,
    ax=None,
    map_limits=None,
    grid_north_pole_lon=-170,
    grid_north_pole_lat=43,
    col_coasts="darkslategray",
    lw_coasts=4,
    col_boundary="black",
    mapfigkw=None,
    resolution="l",
    draw_coasts=True,
    draw_gridlines=True,
):

    if map_limits is None:
        map_limits = [lon[0, 0], lon[-1, -1], lat[0, 0] + 0.5, lat[-1, -1] - 0.15]
    lonmin, lonmax, latmin, latmax = map_limits

    kwas = dict(
        projection="stere",
        lon_0=180 + grid_north_pole_lon,
        lat_0=90 - grid_north_pole_lat,
        lat_ts=90 - grid_north_pole_lat,
        llcrnrlon=lonmin,
        urcrnrlon=lonmax,
        llcrnrlat=latmin,
        urcrnrlat=latmax,
        resolution=resolution,
    )

    if mapfigkw is not None:
        kwas.update(mapfigkw)

    if ax is not None:
        kwas["ax"] = ax

    m = Basemap(**kwas)

    if draw_coasts:
        m.drawcoastlines(color=col_coasts, linewidth=lw_coasts)
    if draw_gridlines:
        m.drawparallels(np.arange(-90, 90, 10))
        m.drawmeridians(np.arange(-180, 180, 10))

    # SR_TMP< TODO consider reimplementing
    # if ax is not None:
    #    # Draw boundaries of computational and analysis domain
    #    bpx, bpy = path_along_domain_boundary(lon, lat)
    #    ax.plot(bpx, bpy, linewidth=1, color=col_boundary)
    #    bpx, bpy = path_along_domain_boundary(lon, lat, nbnd=23)
    #    ax.plot(bpx, bpy, linewidth=1, color=col_boundary)
    # SR_TMP>

    return m


# Various


def netcdf_write_similar_file(infile, outfile, *, replace=None):

    # Retain most contents from coarse grid file
    # Only exclude non-selected variables
    with nc4.Dataset(infile, "r") as fi, nc4.Dataset(outfile, "w") as fo:

        netcdf_derive_file(fi, fo)

        # Replaced coordinate data variables
        if replace is not None:
            for name_var, flds_by_name in replace.items():
                vi = fi.variables[name_var]
                if not isinstance(flds_by_name, dict):
                    flds_by_name = {None: flds_by_name}
                for name_fld, fld in flds_by_name.items():
                    if not name_fld:
                        name = name_var
                    else:
                        name = "{}_{}".format(name_var, name_fld)
                    try:
                        vo = fo.variables[name]
                    except KeyError:
                        vo = fo.createVariable(name, vi.datatype, vi.dimensions)
                        vo.setncatts({a: vi.getncattr(a) for a in vi.ncattrs()})
                    try:
                        vo[:] = fld[:]
                    except (ValueError, IndexError):
                        vo[:] = np.expand_dims(fld, axis=0)


def netcdf_derive_file(fi, fo, retain=None, omit=None):
    """Derive a netCDF file from another by retaining coordinates etc."""

    if retain is None:
        # Default non-dimension variables to be retained
        retain = [
            "time_bnds",
            "rotated_pole",
            "lon",
            "lat",
            "slonu",
            "slatu",
            "slonv",
            "slatv",
            "vcoord",
        ]
    if omit is None:
        omit = []
    retain = [name for name in retain if name not in omit]

    # Global attributes
    fo.setncatts({a: fi.getncattr(a) for a in fi.ncattrs()})

    # Dimensions, incl. variables and arrays
    for dim in fi.dimensions:
        if dim in omit:
            continue
        fo.createDimension(dim)
        try:
            vi = fi.variables[dim]
        except KeyError:
            pass
        else:
            vo = fo.createVariable(vi.name, vi.datatype, vi.dimensions)
            vo.setncatts({a: vi.getncattr(a) for a in vi.ncattrs()})
            vo[:] = vi[:]

    # Other variables to be retained
    for vname, vi in fi.variables.items():
        if vname in retain and vname not in omit:
            vo = fo.createVariable(vi.name, vi.datatype, vi.dimensions)
            vo.setncatts({a: vi.getncattr(a) for a in vi.ncattrs()})
            if vi.shape:
                vo[:] = vi[:]


def nc_write_flds(outfile, grid, flds_by_name, **kwas):
    """Write named fields to NetCDF."""

    dims = ["time", "rlat", "rlon"]

    with nc4.Dataset(outfile, "w") as fo:

        nc_prepare_file(
            fo,
            dims,
            rlat=grid["rlat"],
            rlon=grid["rlon"],
            lat=grid["lat"],
            lon=grid["lon"],
            **kwas,
        )

        for name, fld in flds_by_name.items():

            # SR_TMP<
            err = ("{}: unexpected shape: {} != {}").format(
                name, fld.shape, grid["lon"].shape
            )
            assert fld.shape == grid["lon"].shape, err
            fld = np.expand_dims(fld, axis=0)
            # SR_TMP>

            if fld.dtype == np.bool:
                fld = fld.astype(np.uint8)

            var = fo.createVariable(name, fld.dtype, dims)
            var.grid_mapping = "rotated_pole"
            var.coordinates = "lon lat"
            var[:] = fld


def cmaps_append(cmap1, cmap2, name=None):
    """Appent two color maps to each other."""

    if isinstance(cmap1, str):
        cmap1 = mpl.cm.get_cmap(cmap1)

    if isinstance(cmap2, str):
        cmap2 = mpl.cm.get_cmap(cmap2)

    if name is None:
        name = "{}+{}".format(cmap1.name, cmap2.name)

    ntot = cmap1.N + cmap2.N
    cols = []

    for i in range(0, cmap1.N, 2):
        cols.append(cmap1(float(i) / cmap1.N))

    for i in range(0, cmap2.N, 2):
        cols.append(cmap2(float(i) / cmap2.N))

    return mpl.colors.ListedColormap(cols, name)


def reverse_cmap(cmap, name=None, check_builtins=True):
    """Revert a colormap.

    source: https://stackoverflow.com/a/34351483/4419816
    """
    if cmap is None:
        raise ValueError("cmap is None")

    if name is None:
        # Derive name if not given
        if cmap.name.endswith("_r"):
            name = cmap.name[:-2]
        else:
            name = "{}_r".format(cmap.name)

    if check_builtins:
        # Check if the reverse colormap is available
        try:
            return mpl.cm.get_cmap(name)
        except ValueError:
            pass

    # Try to directly reverse the list of colors
    try:
        colors = cmap.colors
    except AttributeError:
        pass
    else:
        cmap_r = mpl.colors.ListedColormap(colors[::-1], name=name)
        return cmap_r

    # Try something else; TODO: recover source
    reverse = []
    k = []
    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []
        if type(channel) is type(lambda: None):
            raise NotImplementedError("'channel' is a function")
        else:
            for t in channel:
                data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))
    LinearL = dict(zip(k, reverse))
    cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)

    return cmap_r


def plot_cyclones_depressions_extrema(filename, cyclones, depressions, slp, conf):

    suffix = conf["GENERAL"]["image-format"].lower()
    lvl = slp.contour_levels

    outfile_path = conf["GENERAL"]["output-path"]
    outfile_name = os.path.basename(filename).split(".")[0]
    if outfile_name.endswith("p"):
        outfile_name = outfile_name[:-1]

    out = lambda tag: "{p}/{t}_{n}.{s}".format(
        p=outfile_path, t=tag, n=outfile_name, s=suffix
    )
    plt = lambda *args: any(i in conf["GENERAL"]["plots"] for i in args)

    if plt("all", "extrema"):
        plot_extrema(out("extrema"), slp, lvl)

    if plt("all", "depressions"):
        plot_depressions(out("depressions"), depressions, slp, lvl)

    if plt("all", "cyclones"):
        plot_cyclones(out("cyclones"), cyclones, slp, lvl)


if __name__ == "__main__":
    pass

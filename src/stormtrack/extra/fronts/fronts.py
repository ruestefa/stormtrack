#!/usr/bin/env python

# Third-party
import cython
import numpy as np

# Local
from .libfronts import abs2d
from .libfronts import clustering
from .libfronts import equpot
from .libfronts import filt2d
from .libfronts import grad
from .libfronts import norm2d
from .libfronts import pottemp
from .libfronts import scal2d


def identify_fronts(
    *,
    lon,
    lat,
    P,
    T,
    QV,
    U,
    V,
    tvar,
    min_grad,
    minsize,
    n_diffuse,
    stride=1,
    nan=-999,
    print_prefix="",
    var_sfx="",
    verbose=False,
):

    # Check consts arrays
    if not len(lon.shape) == 1 or not len(lat.shape) == 1:
        err = "lon, lat must be 1-dimensional"
        raise ValueError(err)

    # Collect configuration parameters
    conf = dict(stride=stride, n_diffuse=n_diffuse, min_grad=min_grad, minsize=minsize,)

    # Collect constants
    consts = dict(
        nx=lon.size,
        ny=lat.size,
        nz=1,
        shape=(lon.size, lat.size),
        xmin=lon[0],
        xmax=lon[-1],
        ymin=lat[0],
        ymax=lat[-1],
        dx=(lon[-1] - lon[0]) / (lon.size - 1),
        dy=(lat[-1] - lat[0]) / (lat.size - 1),
        dtype=T.dtype,
        nan=nan,
        eps=1.0e-6,
        vb=verbose,  # SR_TMP
        pp=print_prefix,  # SR_TMP
    )

    # Check input fields
    flds = [P, T, QV, U, V]
    if not all(fld.shape == consts["shape"] for fld in flds):
        err = "not all fields have shape {}: {}".format(
            consts["shape"], ", ".join([str(fld.shape) for fld in flds])
        )
        raise ValueError(err)

    # Determine thermal input field
    tvar_choices = ["T", "TH", "THE", "QV"]
    if tvar.startswith("TH"):
        tfld = np.empty(consts["shape"], consts["dtype"], order="F")
        if tvar == "TH":
            if consts["vb"]:
                print(consts["pp"] + " compute potential temperature")
            pottemp(tfld, T, P)
        elif tvar == "THE":
            if consts["vb"]:
                print(consts["pp"] + " compute equivalent potential temperature")
            equpot(tfld, T, QV, P)
        else:
            err = "invalid tvar '{}'; must be in {}".format(tvar, tvar_choices)
            raise ValueError(err)
    elif tvar == "T":
        tfld = T
    elif tvar == "QV":
        tfld = QV
    else:
        err = "invalid tvar '{}'; must be in {}".format(tvar, tvar_choices)
        raise ValueError(err)

    # Apply smoothing
    if consts["vb"]:
        print(
            consts["pp"] + " apply diffusive filter {} times".format(conf["n_diffuse"])
        )
    tfld_orig = tfld.copy()
    for _ in range(conf["n_diffuse"]):
        filt2d(tfld, tfld, 1.0, consts["nan"], 0, 0, 0, 0)

    # Compute frontal area field
    farea, fmask, gradx, grady = compute_farea(tfld, conf, consts)

    # Compute frontal velocity field
    fvel = np.empty(consts["shape"], consts["dtype"], order="F")
    scal2d(U, V, gradx, grady, fvel, consts["nan"])

    # Remove small objects
    object_size_filter(fmask, conf["minsize"], consts)

    # Gap filling: keep only clusters sourrounded by farea
    if consts["vb"]:
        print(consts["pp"] + " warning: not implemented: gap filling (skipped)")

    # Set non-frontal pixels to missing value
    farea[~fmask] = consts["nan"]
    fvel[~fmask] = consts["nan"]

    return {
        "tfld": tfld_orig,
        "fmask": fmask,
        "farea": farea,
        "fvel": fvel,
    }


def compute_farea(tfld, conf, consts):

    nan = consts["nan"]
    shape = consts["shape"]
    dtype = consts["dtype"]

    # Vertical coordinate array
    vert = np.ones(shape, dtype, order="F")

    # Thermal gradient fields
    gradx = np.empty(shape, dtype, order="F")
    grady = np.empty(shape, dtype, order="F")
    grad(
        gradx,
        grady,
        tfld,
        vert,
        consts["xmin"],
        consts["ymin"],
        consts["dx"],
        consts["dy"],
        conf["stride"],
        nan,
    )

    # Absolute and normalized gradients
    farea = np.empty(shape, dtype, order="F")
    abs2d(gradx, grady, farea, nan)
    norm2d(gradx, grady, nan)

    # Frontal area mask
    fmask = (farea != nan) & (farea > conf["min_grad"])
    farea[~fmask] = nan

    return farea, fmask, gradx, grady


def object_size_filter(fmask, minsize, consts):

    if minsize <= 1:
        return

    # Cluster objects
    clusters = np.empty(consts["shape"], np.int32, order="F")
    ntot = clustering(clusters, fmask, consts["nan"])
    if consts["vb"]:
        print(consts["pp"] + " identified {} clusters".format(ntot))

    # Identify small clusters
    clusts_n = np.array(
        [np.count_nonzero(clusters == i) for i in range(ntot)], np.int32, order="F"
    )
    (clusts_del,) = np.where(clusts_n < minsize)

    if consts["vb"]:
        print(
            consts["pp"]
            + " remove {}/{} small clusters (n < {})".format(
                len(clusts_del), ntot, minsize
            )
        )

    # Remove small clusters
    for ind in clusts_del:
        mask = clusters == ind
        clusters[mask] = 0
        fmask[mask] = False

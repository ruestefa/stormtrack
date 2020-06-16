#!/usr/bin/env python3

# Standard library
import argparse
import sys

# Third-party
from IPython.terminal.embed import embed

# Local
from ..utils.various import print_args
from ..utils.various import extract_args
from ..core.io import read_feature_files


DEFAULT_ARGS_WITH_TYPE = {
    "infile_type": (str, "json"),
    "read_pixelfile": (bool, True),
    "read_pixels": (bool, True),
    "rebuild_pixels": (bool, False),
    "rebuild_pixels_if_necessary": (bool, False),
    "minsize": (int, 1),
    "maxsize": (int, -1),
    "timesteps": (int, None),
    "ts_start": (int, None),
    "ts_end": (int, None),
    "tss_excl_mode_tracks": (str, "liberal"),
    "tss_excl_mode_tracked_features": (str, "liberal"),
    "read_tracks": (bool, True),
    "track_store_mode": (str, "json"),
    "discard_untracked_features": (bool, False),
    "retain_n_biggest_tracks": (int, None),
    "optimize_pixelfile_input": (bool, False),
    "timestep": (int, None),
    "silent": (bool, False),
    "counter": (bool, True),
    "ignore_missing_neighbors": (bool, True),
    "ignore_edges_pshare_0": (bool, True),
    "ignore_missing_total_track_stats": (bool, False),
    "ignore_missing_missing_features_stats": (bool, False),
    "silent_core": (bool, None),
    "counter_core": (bool, True),
    "is_subperiod": (bool, True),
}


def main(infiles, feature_name, args_in):

    # Set actual arguments
    args = dict(feature_name=feature_name)
    for arg, (type_, val_def) in DEFAULT_ARGS_WITH_TYPE.items():
        try:
            val_in = args_in.pop(arg)
        except KeyError:
            val = val_def
        else:
            try:
                val = str2bool(val_in) if type_ is bool else type_(val_in)
            except TypeError:
                raise Exception(f"argument '{arg}': {type_.__name__}('{val_in}') fails")
        args[arg] = val

    # Check that there are no remaining input arguments
    if len(args_in) > 0:
        if len(args_in) == 1:
            err = "invalid argument"
        else:
            err = f"{len(args_in)} invalid arguments"
        _str = ", ".join([f"{k}={v}" for k, v in args_in.items()])
        err += f": {_str}\nOptions: {', '.join(DEFAULT_ARGS_WITH_TYPE)}"
        raise Exception(err)

    # Print arguments
    n1 = max([len(k) for k in args.keys()])
    n2 = max([len(str(v)) for t, v in DEFAULT_ARGS_WITH_TYPE.values()]) + 2
    n3 = max([len(str(v)) for v in args.values()]) + 2
    n4 = max([len(t.__name__) for t, v in DEFAULT_ARGS_WITH_TYPE.values()])
    cols = ["ARGUMENT", "VALUE", "DEFAULT", "TYPE"]
    n1, n2, n3, n4 = [max(n, len(c)) for n, c in zip([n1, n2, n3, n4], cols)]
    fmt = f"  {{:<{n1}}}  {{:<{n2}}}  {{:<{n3}}}  {{:<{n4}}}"
    print()
    print(fmt.format(*cols))
    for arg, val in args.items():
        if arg == "feature_name":
            type_, def_ = str, None
        else:
            type_, def_ = DEFAULT_ARGS_WITH_TYPE[arg]
        val = str(val) if not isinstance(val, str) else f"'{val}'"
        def_ = str(def_) if not isinstance(def_, str) else f"'{def_}'"
        type_ = type_.__name__
        print(fmt.format(arg, val, def_, type_))
    print()

    # Print input file(s)
    _n = len(infiles)
    print(
        f"reading {len(infiles)} infile{'s' if len(infiles) > 1 else ''}:\n"
        + "\n".join([f"  {f}" for f in infiles])
    )

    # Read tracks
    r = read_feature_files(infiles, **args)
    features = r["features"]
    tracks = r["tracks"]
    timesteps = r["timesteps"]

    # Drop into iPython shell
    print("\ndropping into iPython shell..")
    msg = f"INSPECT {len(tracks)} TRACKS"
    embed(header=msg)


def str2bool(str_):
    try:
        return {"True": True, "False": False}[str_]
    except KeyError:
        raise TypeError(type(str_).__name__)


def setup_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        help="feature track input file(s) (.pickle)",
        metavar="infile[s]",
        nargs="+",
        dest="infiles",
    )
    parser.add_argument(
        "-f",
        "--feature-name",
        help="name of features in infiles",
        metavar="name",
        dest="feature_name",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--arg",
        help="argument for read_feature_files",
        metavar=("name", "value"),
        nargs=2,
        action="append",
        default=[],
        dest="args_lst",
    )
    return parser


def preproc_args(parser, args):
    kwargs = vars(args)
    kwargs["args_in"] = dict(kwargs.pop("args_lst"))
    return kwargs


def cli():
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    args = parser.parse_args()
    kwargs = preproc_args(parser, args)
    main(**kwargs)


if __name__ == "__main__":
    cli()

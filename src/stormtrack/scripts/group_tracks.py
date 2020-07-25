#!/usr/bin/env python3

# Standard library
import argparse
import cProfile
import os
import pstats
import sys

# Local
from stormtrack.core.io import distribute_tracks_across_outfiles
from stormtrack.core.io import read_feature_files
from stormtrack.core.io import write_feature_file
from stormtrack.utils.various import TimestepGenerator
from stormtrack.utils.various import TimestepStringFormatter
from stormtrack.utils.various import print_args


def main(
    infiles_ts,
    feature_name,
    outfiles,
    skip_output_start,
    skip_output_end,
    skip_output,
    profiling,
    args_grouping,
    ignore_missing_stats,
):

    pr = None
    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    # Read features and tracks
    infiles = [fi for ts, fi in sorted(infiles_ts.items())]
    _r = read_feature_files(
        infiles,
        feature_name=feature_name,
        counter=True,
        ignore_missing_neighbors=True,
        ignore_missing_total_track_stats=ignore_missing_stats,
        ignore_missing_features_stats=ignore_missing_stats,
        ignore_edges_pshare_0=True,
    )
    tracks = _r["tracks"]
    jdats = _r["jdats"]

    # Collect timestep ranges for infiles
    timestep_range_files = []
    for timesteps, infile in sorted(infiles_ts.items()):
        ts_range = (timesteps[0], timesteps[-1])
        timestep_range_files.append((ts_range, infile))

    # Replace infiles by outfiles in the data structure
    insert_outfiles(timestep_range_files, outfiles, skip_output_start, skip_output_end)

    # Prepare output: distribute tracks and split them up again
    # Optionally sort tracks into groups by thresholding a method
    outfiles_jdats_subtracks = distribute_tracks_across_outfiles(
        tracks, timestep_range_files, jdats, **args_grouping
    )

    # Write features/tracks back to disk
    if not skip_output:
        write_features_tracks(outfiles_jdats_subtracks, feature_name)

    if profiling:
        pr.disable()
        ps = pstats.Stats(pr)
        nlines = 20
        ps.strip_dirs().sort_stats("tottime").print_stats(nlines)
        ps.strip_dirs().sort_stats("cumtime").print_stats(nlines)


def insert_outfiles(timestep_range_files, outfiles, nskip_start, nskip_end):
    assert len(timestep_range_files) == len(outfiles)
    ntot = len(outfiles)
    for i, ((ts_range, infile), outfile) in enumerate(
        zip(timestep_range_files, outfiles)
    ):
        if i < nskip_start or (ntot - i) <= nskip_end:
            skip = True
        else:
            skip = False
        timestep_range_files[i] = (ts_range, outfile, skip)


def write_features_tracks(outfiles_jdats_subtracks, feature_name, **kwas):

    ntot = len(outfiles_jdats_subtracks)
    for i, (outfile, (jdat, subtracks)) in enumerate(
        sorted(outfiles_jdats_subtracks.items())
    ):
        print(
            "write {} tracks with {} features to {}".format(
                len(subtracks),
                len([f for t in subtracks for f in t.features()]),
                outfile,
            )
        )

        # SR_TMP< TODO properly implement read/write w/ some File class
        key_info_features = "info_{}".format(feature_name)
        nx, ny = jdat["info_tracks"]["nx"], jdat["info_tracks"]["ny"]
        # SR_TMP>

        write_feature_file(
            outfile,
            tracks=subtracks,
            feature_name=feature_name,
            info_tracks=jdat["info_tracks"],
            info_features=jdat[key_info_features],
            nx=nx,
            ny=ny,
            **kwas,
        )


def str_to_bool(str_):
    """Convert string to bool."""
    if str_ == "True":
        return True
    elif str_ == "False":
        return False
    else:
        raise TypeError(str_)


description = """Separate tracks by a threshold based on on a track method.

Tracks are sorted into two groups and written to different files.
If multiple track files are passed, partial tracks are first remerged.

The name of the method should be passed (e.g. duration), along with optional
key value pairs for method arguments.

If no method name is passed, the tracks are written back to the same file
(albeit optionally in a different folder). This can be useful to refresh
data in the json file.
"""


def setup_parser():
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input = parser.add_argument_group("input")
    input.add_argument(
        "-i",
        "--infile-fmt",
        help="input file template (with {YYYY} etc.)",
        metavar="format-string",
        dest="infile_format",
        required=True,
    )
    input.add_argument(
        "-f",
        "--feature-name",
        help="type name of tracked features in input file",
        metavar="name",
        dest="feature_name",
        required=True,
    )
    input.add_argument(
        "-s",
        "--timesteps",
        help="timesteps used to reconstruce input file names",
        metavar="YYYYMMDDHH",
        nargs="+",
        type=int,
        dest="tss_list",
    )
    input.add_argument(
        "-S",
        "--timesteps-range",
        help="timestep range (format: YYYYMMDDHH; stride in hours)",
        metavar=("start", "end", "stride"),
        nargs=3,
        type=int,
        action="append",
        dest="tss_range_lst",
    )
    input.add_argument(
        "--ignore-missing-stats",
        help=(
            "ignore missing features stats of partial tracks in "
            "input, which might be useful for outdated files "
            "where those stats present might trigger errors; "
            "keep in mind, however, that they can only be safely "
            "ignored if partial tracks are fully re-merged, i.e., "
            "if the entire tracking period is read at once; "
            "otherwise incomplete track files with broken partial "
            "tracks result"
        ),
        action="store_true",
        dest="ignore_missing_stats",
    )
    output = parser.add_argument_group("output")
    output.add_argument(
        "-o",
        "--outfile-fmt",
        help=(
            "template for output file (with {YYYY} etc, but same "
            "as infile format); can contain '{GROUP}' to place "
            "group names (given or generated) manually"
        ),
        metavar="format-string",
        dest="outfile_format",
    )
    output.add_argument(
        "-O",
        "--outdir",
        help="output directory; overrides any other directories",
        metavar="directory",
        dest="outdir",
    )
    output.add_argument(
        "--skip-output-n",
        help=("skip N output steps in the beginning and M output " "steps in the end"),
        metavar=("N", "M"),
        nargs=2,
        type=int,
        default=[0, 0],
        dest="skip_output_n",
    )
    output.add_argument(
        "--skip-output",
        help="skip output (for profiling/debugging)",
        action="store_true",
        dest="skip_output",
    )
    grouping = parser.add_argument_group("grouping")
    grouping.add_argument(
        "-m",
        "--method-name",
        help=(
            "name of method used for sorting tracks; if multiple "
            "methods are passed, they are combined with AND"
        ),
        metavar="name",
        action="append",
        dest="method_name_lst",
    )
    grouping.add_argument(
        "-t",
        "--threshold",
        help=(
            "threshold for sorting by method; must be passed as "
            "often as --method-name"
        ),
        metavar="value",
        action="append",
        type=float,
        dest="threshold_lst",
    )
    # SR_TODO add --invert flag to invert comparison for certain method names,
    # SR_TODO meaning that fronts below the threshold are sorted into the
    # SR_TODO "at/above" group and vice versa
    grouping.add_argument(
        "-g",
        "--group-names",
        help=(
            "names for two groups; by default, these contain the "
            "tracks falling below and at/above the threshold, "
            "respectively (TODO: implement --invert flag); "
            "must be passed alongside --method-name etc."
        ),
        nargs=2,
        metavar=("group1", "group2"),
        dest="group_names",
    )
    # SR_TMP< currently not supported
    grouping.add_argument(
        "--ab",
        "--method-arg-bool",
        help="key value pair for bool attribute",
        metavar=("key", "val"),
        nargs=2,
        action="append",
        default=[],
        dest="method_kwas_bool",
    )
    grouping.add_argument(
        "--ai",
        "--method-arg-int",
        help="key value pair for int attribute",
        metavar=("key", "val"),
        nargs=2,
        action="append",
        default=[],
        dest="method_kwas_int",
    )
    # SR_TMP>
    exe = parser.add_argument_group("execution")
    exe.add_argument(
        "--profile", help="run profiling", action="store_true", dest="profiling",
    )
    return parser


def preproc_args(parser, **kwas):

    # Prepare timesteps
    try:
        timesteps = TimestepGenerator.from_args(
            kwas.pop("tss_list"), kwas.pop("tss_range_lst")
        )
    except ValueError as e:
        parser.error(e)

    # Check consistency of infile and outfile format strings
    infile_format = kwas.pop("infile_format")
    outfile_format = kwas.pop("outfile_format")
    if outfile_format is None:
        outfile_format = infile_format
    else:
        for key in ["{YYYY}", "{MM}", "{DD}", "{HH}"]:
            # Only check file names, not paths
            if os.path.basename(infile_format).count(key) != os.path.basename(
                outfile_format
            ).count(key):
                parser.error(
                    "infile and outfile format file names must "
                    "contain the same number of timestep format elements"
                )

    # Prepare input and output files
    outdir = kwas.pop("outdir")
    if outdir is not None:
        outfile_format = "{}/{}".format(outdir, os.path.basename(outfile_format))
    infiles_ts = TimestepStringFormatter(infile_format).run(timesteps)
    outfiles_ts = TimestepStringFormatter(outfile_format).run(timesteps)
    kwas["infiles_ts"] = infiles_ts
    kwas["outfiles"] = sorted(outfiles_ts.values())

    # Separate skip_output_n argument
    skip_output_n = kwas.pop("skip_output_n")
    kwas["skip_output_start"], kwas["skip_output_end"] = skip_output_n

    # Check argument consistencies depending on whether grouping is conducted
    if not kwas["method_name_lst"]:
        if kwas["group_names"]:
            parser.error(
                "inconsistent options: --group-names only valid "
                "alongside --method-name etc."
            )
        if kwas["threshold_lst"]:
            parser.error(
                "inconsistent options: --threshold only valid "
                "alongside --method-name"
            )
    else:
        if not kwas["threshold_lst"]:
            parser.error(
                "inconsistent options: --threshold must be passed "
                "alongside --method-name"
            )
        if len(kwas["method_name_lst"]) != len(kwas["threshold_lst"]):
            parser.error(
                "must pass --method-name and --threshold the same " "number of times"
            )
        if not kwas["group_names"]:
            parser.error(
                "inconsistent options: --group-names must be passed "
                "alongside --method-name"
            )

    # Convert method arguments to specific types
    # SR_TMP<
    if kwas["method_kwas_bool"] or kwas["method_kwas_int"]:
        parser.error(
            "method arguments currently not supported because it is "
            "not streight-forward to pass arbitrary numbers of arguments "
            "to arbitrary numbers of methods, and only one argument "
            "per type and method doesn't make too much sense"
        )
    del kwas["method_kwas_bool"], kwas["method_kwas_int"]
    method_name_lst = kwas["method_name_lst"]
    n_methods = 0 if method_name_lst is None else len(method_name_lst)
    kwas["method_kwas_lst"] = [{} for _ in range(n_methods)]

    # Collect grouping arguments
    kwas["args_grouping"] = {}
    for key in ["group_names"]:
        kwas["args_grouping"][key] = kwas.pop(key)

    # Collect name, threshold, and arguments for all methods
    args_methods = []
    tmp = [
        kwas.pop(key) for key in ["method_name_lst", "threshold_lst", "method_kwas_lst"]
    ]
    if n_methods > 0:
        for method_name, threshold, method_kwas in zip(*tmp):
            args_methods.append(
                dict(name=method_name, threshold=threshold, kwas=method_kwas)
            )
    kwas["args_grouping"]["args_methods"] = args_methods

    return kwas


def cli():
    parser = setup_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)
    args = parser.parse_args()
    try:
        print_args(args)
    except NameError:
        pass
    kwas = preproc_args(parser, **vars(args))
    main(**kwas)


if __name__ == "__main__":
    cli()
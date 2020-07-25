#!/bin/bash -e

# Location of this script
DIR="$(dirname $(readlink -f "${BASH_SOURCE[0]}"))"

# Useful functions
source "${DIR}/../run_tools"

# Executable
EXE="group-tracks"
which ${EXE} >/dev/null 2>&1 || {
    echo "error: executable now found: ${EXE}" >&2
    exit 1
}

#------------------------------------------------------------------------------

args=(feature_name)
opts=(dts ts_start ts_end skip_start skip_end "criterion [criterion ...]")
n_args=$#
if [ ${n_args} -lt "${#args[@]}" ]; then
    echo "error: expected ${#args[@]} arguments, got ${n_args}" >&2
    echo "usage: $(basename ${0}) ${args[@]} [${opts[@]}]" >&2
    exit 1
fi

# Feature name
feature_name=${1}

# Defaults for optional arguments
default_skip_start=0
default_skip_end=0
default_dts=3
default_ts_start="2000090100"
default_ts_end="2000113021"
default_criteria=(
    duration
    features_ts_n__p80
    footprint_n_ts_rel__p80
)

# Optional: Skip the first N output files
skip_start=${2:-${default_skip_start}}

# Optional: Skip the last N output files
skip_end=${3:-${default_skip_end}}

# Optional: Time step delta (e.g., in hours for YYYYMMDDHH time steps)
dts=${4:-${default_dts}}

# Optional: Start time step (e.g., YYYYMMDDHH)
ts_start=${5:-${default_ts_start}}

# Optional: End time step (e.g., YYYYMMDDHH)
ts_end=${6:-${default_ts_end}}

n_shift=$((${n_args} > 6 ? 6 : ${n_args}))
shift ${n_shift}

# Optional: Grouping criteria by name
criteria=(${@})
[ ${#criteria[@]} -eq 0 ] && criteria=(${default_criteria[@]})

#------------------------------------------------------------------------------

# Default thresholds for default criteria
# Note: The thresholds are currently not part of the interface of this script
default_duration_threshold=24
default_features_ts_n__p80_threshold=400
default_footprint_n_ts_rel__p80_threshold=8

#------------------------------------------------------------------------------
# Command line arguments
#------------------------------------------------------------------------------

# Input
flags_in=()
indir="data.tracks.fronts/${feature_name}/{YYYY}"
infile="tracks_${feature_name}_raw_{YYYY}{MM}.pickle"
flags_in+=(-i="${indir}/${infile}")
flags_in+=(-f=${feature_name})
flags_in+=(-S ${ts_start} ${ts_end} ${dts})
flags_in+=(--skip-output-n ${skip_start} ${skip_end})

# Output
flags_out=()
outdir="${indir}"
outfile="tracks_${feature_name}_{GROUP}_{YYYY}{MM}.pickle"
flags_out+=(-o="${outdir}/${outfile}")

# Grouping
flags_grp=()
flags_grp+=(-g local synop)
for var in ${criteria[@]}; do
    flags_grp+=(-m="${var}")
    case "${var}" in
    duration)
        flags_grp+=(-t ${default_duration_threshold})
        ;;
    features_ts_n__p80)
        flags_grp+=(-t ${default_features_ts_n__p80_threshold})
        ;;
    footprint_n_ts_rel__p80)
        flags_grp+=(-t ${default_footprint_n_ts_rel__p80_threshold})
        ;;
    *)
        echo "no threshold defined for '${var}'" >&2
        exit 5
        ;;
    esac
done
# Pass one or more method name (`-m`)/threshold (`-t`) flag pair.
# The name must correspond to a `FeatureTrack` method that takes no arguments.
# The threshold is applied to the value returned by this method for each track.
# Tracks where any of the values is below the respective threshold are grouped
# into the first group defined by `-g`, while tracks where all values are above
# the respective threshold are grouped in the second group defined by `-g`
# Default flags:
# ```
# -g local synop \
# -m duration -t 24 \
# -m features_ts_n__p80 400 \
# -m footprint_n_ts_rel__p80 8
# ```
# Some notes on this interface:
# - It is very ad-hoc and inflexible in many ways, as the author is well aware.
# - The `FeatureTrack` methods like `features_ts_n__p80` have ONLY been added
#   for this grouping interface and have never be intended to be there long-term
#   in their current form. It has simply been a convenient, if very hacky and
#   dirty, way of getting the grouping running.
# - Note that there are additional similar methods defined on `FeatureTrack`
#   that can be used for the grouping; just check the code or explore them in
#   an interactive iPython session with tracks loaded.
# - It is obviously less than optimal that there is no way to specify whether
#   a value should be below or above the respective threshold for the track to
#   land in a certain group, but that instead all thresholds are upper/lower
#   thresholds with respect to the first/second group.
# - That being said, for now it works as it is! In case of change requests,
#   do not hesitate to contact the author.

#------------------------------------------------------------------------------
# Run program
#------------------------------------------------------------------------------

${EXE} \
    "${flags_in[@]}" \
    "${flags_out[@]}" \
    "${flags_grp[@]}" ||
    {
        err=${?}
        echo "exec error: exit ${err}" >&2
        exit ${err}
    }

#------------------------------------------------------------------------------

#!/bin/bash -e

# Location of this script
DIR="$(dirname $(readlink -f "${BASH_SOURCE[0]}"))"

# Executable
EXE="track-features"
which ${EXE} >/dev/null 2>&1 || {
    echo "error: executable now found: ${EXE}" >&2
    exit 1
}

# ------------------------------------------------------------------------------
# Command line arguments
# ------------------------------------------------------------------------------

args=(num_procs front_type)
opts=(dts ts_start ts_end)
if [ ${#} -lt ${#args[@]} ]
then
    echo "error: expected ${#args[@]} arguments, got ${#}" >&2
    echo "usage: $(basename ${0}) ${args[@]} [${opts[@]}]" >&2
    exit 1
fi

# Number of parallel processes:
# - 1: Purely sequential execution
# - 2: Identification and tracking run decoupled, but each sequentially
# - *: Identification runs additionally parallelized
num_procs=${1}

# Front type: 'cold' or 'warm'
front_type=${2}

# Defaults for optional arguments
default_dts=3
default_ts_start="2000090100"
default_ts_end="2000113021"

# Optional: Time step delta
dts=${3:-${default_dts}}

# Optional: Start time step
ts_start=${4:-${default_ts_start}}

# Optional: End time step
ts_end=${5:-${default_ts_end}}

# ------------------------------------------------------------------------------

# Check num_procs and determine whether to decouple the identification
if [ ${num_procs} -eq 1 ]
then
    decoupled_id=0
    num_procs_id=0
elif [ ${num_procs} -ge 2 ]
then
    decoupled_id=1
    num_procs_id=$((num_procs - 1))
else
    echo "invalid num_proc=${num_proc} (must be > 0)" >&2
    exit 1
fi

# Check front type
case ${front_type} in
    cold|warm)
    ;;
    *)
        echo "error: front type: '${front_type}' neither 'cold' nor 'warm'" >&2
        exit 1
    ;;
esac

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

# Run
skip_start=0  # Skip first N outputs
skip_end=0  # Skip last N outputs

# Data
level=850
feature_name="${front_type}front${level}"
temp_var="THE"
vname_farea="farea_${temp_var}"
vname_fvel="fvel_${temp_var}"

# Identification
n_smooth=25
minsize=10
thresh_fvel=1
n_bnd_trim=10
split_thresh=-1
split_peak=-1
topo_thresh=-1

# Mid-monthly identification thresholds
refyear="${ts_start:0:3}0"  # Either 2000 (clim) or 2080 (pgw)
case ${refyear} in
    2000) threshs_farea_monthly=(
          # Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
            4.0  4.0  5.0  6.0  7.0  8.0  8.0  8.0  7.0  6.0  5.0  4.0
        ) ;;
    2080) threshs_farea_monthly=(
          # Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec
            4.5  4.5  5.75 7.0  8.25 9.5  9.5  9.5  8.25 7.0  5.75 4.5
        ) ;;
    *) echo "invalid refyear '${refyear}'" >&2; exit 1;;
esac

# Tracking
min_p_size=0.6
min_p_overlap=0.1
min_p_tot=0.4
alpha=0.5
max_children=6
min_duration=$((6/dts))
split_tracks=-1

# ------------------------------------------------------------------------------
# Files and folders
# ------------------------------------------------------------------------------

# Input
infile_const="data/crclim_const_lm_c.nc"
case $((dts % 3)) in
    0) indir="data/3hrly/{YYYY}/{MM}";;
    *) indir="data/1hrly/{YYYY}/{MM}";;
esac
infile="lffd{YYYY}{MM}{DD}{HH}p.nc"

# Ouptut
outdir="data.tracks.fronts/${feature_name}"
mkdir -pv "${outdir}"
outdir+="/{YYYY}"
outfile="tracks_${feature_name}_raw_{YYYY}{MM}"

# ------------------------------------------------------------------------------
# Command line options
# ------------------------------------------------------------------------------

# Input
flags_in=()
flags_in+=(-S ${ts_start} ${ts_end} ${dts})
flags_in+=(--lonlat-names lon lat)
flags_in+=(--infile-lonlat="${infile_const}")
flags_in+=(--topo-file="${infile_const}")
flags_in+=(--infile-fmt="${indir}/${infile}")
flags_in+=(--fronts-var-suffix="_hy50")
flags_in+=(--fronts-temp-var="${temp_var}")

# Output
flags_out=()
flags_out+=(--outfile-fmt="${outdir}/${outfile}.pickle")
flags_out+=(--output-skip-start=${skip_start})
flags_out+=(--output-skip-end=${skip_end})

# Identification
flags_idfy=()
flags_idfy+=(--comp="fronts")
flags_idfy+=(--level=${level})
flags_idfy+=(--feature-name="${feature_name}")
flags_idfy+=(--fld-varname="${vname_farea}")
flags_idfy+=(--mask-varname="${vname_fvel}")
flags_idfy+=(--mask-gt="${thresh_fvel}")
flags_idfy+=(--replace-varname="${vname_fvel}")
flags_idfy+=(--lower-thresholds-monthly ${threshs_farea_monthly[@]})
flags_idfy+=(--minsize=${minsize})
[ ${split_thresh} -gt 0 ] && flags_idfy+=(--split-levels=${split_thresh})
[ ${split_thresh} -gt 0 ] && flags_idfy+=(--split-seed-minstrength=${split_peak})
flags_idfy+=(--trim-boundaries=${n_bnd_trim})
flags_idfy+=(--topo-filter-threshold=${topo_thresh})
case ${front_type} in
    cold ) flags_idfy+=(--mask-nomirror) ;;
    warm ) flags_idfy+=(--mask-mirror) ;;
    *) echo "unknown front type: ${front_type}" >&2; exit 4;;
esac
flags_idfy+=(--fronts-diffuse=${n_smooth})

# Tracking
flags_track=()
flags_track+=(--min-p-size=${min_p_size})
flags_track+=(--min-p-overlap=${min_p_overlap})
flags_track+=(--min-p-tot=${min_p_tot})
flags_track+=(--alpha=${alpha})
flags_track+=(--minsize=${minsize})
flags_track+=(--max-children=${max_children})
flags_track+=(--min-duration=${min_duration})
[ ${split_thresh} -gt 0 ] && flags_track+=(--merge-features)
flags_track+=(--split-tracks=${split_tracks})

# Execution
flags_exe=()
[ ${decoupled_id} -ne 0 ] && flags_exe+=(--overlay-id-track)
flags_exe+=(--num-procs-id=${num_procs_id})
# flags_exe+=(--profile=30)

# ------------------------------------------------------------------------------
# Run program
# ------------------------------------------------------------------------------

${EXE} \
    ${flags_in[@]} \
    ${flags_out[@]} \
    ${flags_idfy[@]} \
    ${flags_track[@]} \
    ${flags_exe[@]} \
|| { err=${?}; echo "exec error: exit ${err}" >&2; exit ${err}; }

# ------------------------------------------------------------------------------

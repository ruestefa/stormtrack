#!/bin/bash -e

# Location of this script
DIR="$(dirname $(readlink -f "${BASH_SOURCE[0]}"))"

# Executable
EXE=track-features
which ${EXE} >/dev/null 2>&1 || {
    echo "error: executable now found: ${EXE}" >&2
    exit 1
}

#------------------------------------------------------------------------------
# Command line arguments
#------------------------------------------------------------------------------

args=(num_procs)
if [ ${#} -ne ${#args[@]} ]
then
    echo "error: expected ${#args[@]} arguments, got ${#}" >&2
    echo "usage: $(basename ${0}) ${args[@]}" >&2
    exit 1
fi

#------------------------------------------------------------------------------

# Number of parallel processes:
# - 1: Purely sequential execution
# - 2: Identification and tracking are decoupled, though each sequential.
# - *: Identification is additionally parallelized.
num_procs=${1}

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

# Run
make_plots=0            # Plot the features
ts_start=2000090100     # Start time step
ts_end=2000113021       # End time step
dts=6                   # Time step delta
skip_start=0            # Skip first N outputs
skip_end=0              # Skip last N outputs

# Data
varname_base="FI"
varname="${varname_base}_hy50"
input_field_level=0
feature_name=cyclone

# Identification
smooth_sig=7
contour_interval=1
contour_level_start=50
contour_level_end=250
contour_length_max="5000"
min_cyclone_depth=${contour_interval}
refval=$((contour_level_start + (contour_level_end - contour_level_start)/2))

# Tracking
mindur=1
min_p_size=0.4
min_p_ovlp=0.4
min_p_tot=0.4
alpha=0.5
minsize=100
max_child=4
tsplit=-1

#------------------------------------------------------------------------------
# Files and folders
#------------------------------------------------------------------------------

# Input
ifile_const="data/crclim_const_lm_c.nc"
indir="data/${dts}hrly/{YYYY}/{MM}"
infile="lffd{YYYY}{MM}{DD}{HH}p.nc"

# Output
outdir="tracks"
mkdir -pv "${outdir}"
outfile="tracks_${feature_name}_duration-ge-${mindur}h_{YYYY}{MM}.pickle"
outdir_img="png"
[ ${make_plots} -ne 0 ] && mkdir -pv "${outdir_img}"

# Setup
cycl_setup_file="${outdir}/${feature_name}_${varname}.ini"

#------------------------------------------------------------------------------
# Create cyclone identification setup file
#------------------------------------------------------------------------------

cat > "${cycl_setup_file}" <<EOF
[GENERAL]

# Plotting
image-format = PNG
make-plots = $([ ${make_plots} -eq 0 ] && echo False || echo True)
plots = cyclones, extrema
output-path = ${outdir_img}

# level (in case of 3D input field)
input-field-level = ${input_field_level}
input-field-name = ${varname}

# Save the paths of contours to a binary file.
save-contour-paths-binary = True

# Measure local timings.
#timings = True
timings = False

topo-field-name = HSURF
topofile = LMCONSTANTS

[IDENTIFY]

# max. fraction of boundary-crossing contours (BCCs) per feature (Depression/Cyclone) [0..1]
bcc-fraction = 0.2

# sampling interval for contours (hPa or gpdm)
contour-interval = ${contour_interval}

# maximal contour length [km]
contour-length-max = ${contour_length_max}

# minimal contour length [km]
contour-length-min = -1.0

# start and end levels for contours (hPa or gpdm)
contour-level-start = ${contour_level_start}
contour-level-end = ${contour_level_end}

# minimal number of contours for Depressions.
depression-min-contours = 1

# side length of the neighbourhood for finding local minima/maxima
#extrema-identification-size = 3
extrema-identification-size = 9
#extrema-identification-size = 21

# if True, only closed contours (never leave the domain) are considered for cyclones
force-contours-closed = False
#force-contours-closed = True

# use datetime-based object IDs
ids-datetime = True

# for datetime-based IDs, no. digits of ID
ids-datetime-digits = 3

# max. no. minima per cyclone
max-minima-per-cyclone = 3

# minimal cyclone depth measured from the outer-most contour to the deepest minimum [hPa]
min-cyclone-depth = ${min_cyclone_depth}

# min. distance in grid points of minima/maxima from domain boundary
size-boundary-zone = 2

# smoothing factor (sigma value of Gaussian filter)
smoothing-sigma = ${smooth_sig}

# cut-off level for topography [km|hPa]
#topo-cutoff-level = 1500.0
topo-cutoff-level = -1
EOF

#------------------------------------------------------------------------------
# Command line flags
#------------------------------------------------------------------------------

# Input
flags_in=()
flags_in+=(-S ${ts_start} ${ts_end} ${dts})
flags_in+=(--lonlat-names lon lat)
flags_in+=(--infile-lonlat="${ifile_const}")
flags_in+=(--topo-file="${ifile_const}")
flags_in+=(-i="${indir}/${infile}")

# Output
flags_out=()
flags_out+=(-o="${outdir}/${outfile}")
flags_out+=(--output-skip-start=${skip_start})
flags_out+=(--output-skip-end=${skip_end})

# Identification
flags_idfy=()
flags_idfy+=(-n=${feature_name})
flags_idfy+=(--comp=${feature_name}s)
flags_idfy+=(--reference-value=${refval})
flags_idfy+=(--cyclones-inifile="${cycl_setup_file}")
flags_idfy+=(--level=${input_field_level})

# Tracking
flags_track=()
flags_track+=(--min-p-size=${min_p_size})
flags_track+=(--min-p-overlap=${min_p_ovlp})
flags_track+=(--min-p-tot=${min_p_tot})
flags_track+=(--alpha=${alpha})
flags_track+=(--minsize=${minsize})
flags_track+=(--max-children=${max_child})
flags_track+=(--min-duration=${mindur})
flags_track+=(--split-tracks=${tsplit})

# Execution
flags_exe=()
[ ${decoupled_id} -ne 0 ] && flags_exe+=(--overlay-id-track)
flags_exe+=(--num-procs-id=${num_procs_id})
# flags_exe+=(--profile=30)

#------------------------------------------------------------------------------
# Run program
#------------------------------------------------------------------------------

${EXE} \
    ${flags_in[@]} \
    ${flags_out[@]} \
    ${flags_cycl[@]} \
    ${flags_idfy[@]} \
    ${flags_track[@]} \
    ${flags_exe[@]} \
|| { err=${?}; echo "execution error: exit ${err}" >&2; exit ${err}; }

#------------------------------------------------------------------------------

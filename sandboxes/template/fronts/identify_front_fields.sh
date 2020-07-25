#!/bin/bash -e

# Location of this script
DIR="$(dirname $(readlink -f "${BASH_SOURCE[0]}"))"

# Useful functions
source "${DIR}/../run_tools"

# Executable
EXE="identify-front-fields"
which ${EXE} >/dev/null 2>&1 || {
    echo "error: executable now found: ${EXE}" >&2
    exit 1
}

# ------------------------------------------------------------------------------

args=(num_procs temp_var level dts "tss_expr [tss_expr ...]")
opts=()
n_args=$#
if [ ${n_args} -lt "${#args[@]}" ]; then
    echo "error: expected ${#args[@]} arguments, got ${n_args}" >&2
    echo "usage: $(basename ${0}) ${args[@]} [${opts[@]}]" >&2
    exit 1
fi

# No. processes: 1: sequential; >1: parallel
num_procs=${1}

# Temperature variable, e.g., T, TH, THE
temp_var=${2}

# Vertical level (hPa), e.g., 850, 700
level=${3}

# Time step delta
dts=${4}

shift 4

# Time step expressions, e.g., 2004, 2008011900-2008012123, 200201+200202+200212
tss_exprs=(${@})

# ------------------------------------------------------------------------------

# Setup
n_smooth=25
min_grad=0
minsize=10
stride=1

# ------------------------------------------------------------------------------
# Command line options
# ------------------------------------------------------------------------------

# Input
flags_in=()
tss_ranges_lists=($(expand_month_datetime_lists_ranges yyyymmddhh \
    ${tss_exprs[@]})) || {
    echo "error: expand_month_datetime_lists_ranges failed" >&2
    exit 1
}
for tss_ranges in ${tss_ranges_lists[@]}; do
    flags_in+=(-S $(echo ${tss_ranges//-/ } | sed "s/,/ ${dts} -S /g") ${dts})
done
case $((dts % 3)) in
0) indir="data/3hrly/{YYYY}/{MM}" ;;
*) indir="data/1hrly/{YYYY}/{MM}" ;;
esac
infile="lffd{YYYY}{MM}{DD}{HH}p.nc"
flags_in+=(--infile-fmt="${indir}/${infile}")
flags_in+=(--level=${level})
flags_in+=(--lonlat-names rlon rlat)
flags_in+=(--temp-var="${temp_var}")
flags_in+=(--var-name-p="pressure")
flags_in+=(--var-name-t="T_hy50")
flags_in+=(--var-name-qv="QV_hy50")
flags_in+=(--var-name-u="U_hy50")
flags_in+=(--var-name-v="V_hy50")
flags_in+=(--var-uv-staggered)

# Output
flags_out=()
outdir="data.fronts/${temp_var}.${level}"
outdir+=".smt${n_smooth}"
outdir+="/{YYYY}/{MM}"
outfile="fronts_${temp_var}_${level}_{YYYY}{MM}{DD}{HH}.nc"
flags_out+=(--outfile-fmt="${outdir}/${outfile}")
flags_out+=(--outvars
    finp
    fmask
    farea
    fvel
)

# Setup
flags_setup=()
flags_setup+=(--diffuse="${n_smooth}")
flags_setup+=(--min-grad="${min_grad}")
flags_setup+=(--minsize=${minsize})
flags_setup+=(--stride=${stride})

# Execution
flags_exe=()
flags_exe+=(--verbose)
# flags_exe+=(--profile)
flags_exe+=(--$([ ${num_procs} -eq 1 ] && echo seq || echo par))
flags_exe+=(--num-procs=${num_procs})

# ------------------------------------------------------------------------------

${EXE} \
    ${flags_in[@]} \
    ${flags_out[@]} \
    ${flags_setup[@]} \
    ${flags_exe[@]} ||
    {
        err=${?}
        echo "execution error: exit ${err}" >&2
        exit ${err}
    }

# ------------------------------------------------------------------------------

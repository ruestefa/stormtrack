#/bin/bash
#
# Collect Python test files and run the tests.
#
# Motivation:
#   Alternative to  pytest, which does not work with Cython code.
#
# Author:
#   Stefan Ruedisuehli
#   stefan.ruedisuehli@env.ethz.ch
#   ETH Zurich, IAC, Atmosdyn
#
# History:
#   2020-02-14: Basic implementation: collect and run all test suites
#   2020-02-21: Refinements: add SKIP option; fine-tune verbosity
#               Implement env-var-based and env-var-like options
#
#------------------------------------------------------------------------------
# Options: Pass as env vars or env-var-like options
#------------------------------------------------------------------------------

# Export env-var-like CLI options as env vars
# Example: `$ script.sh FOO=BAR` should behave like `$ FOO=BAR script.sh`
export_env_vars()
{
    local arg; for arg in "${@}"
    do
        local name_val=(${arg/=/ })
        export ${name_val[0]}="${name_val[1]}"
    done
}
export_env_vars "${@}"

# Path to test root directory
DEFAULT_TEST_DIR="./tests"
TEST_DIR="${TEST_DIR:-${DEFAULT_TEST_DIR}}"

# Mdules (dot-path; may be partial) to skip (comma-separated list)
DEFAULT_SKIP_MODULES=''
SKIP_MODULES="${SKIP_MODULES:-${DEFAULT_SKIP_MODULES}}"

# Dry run without running the tests
DEFAULT_DRY_RUN=false
DRY_RUN="${DRY_RUN:-${DEFAULT_DRY_RUN}}"

# Verbosity (0 silent; 1 regular; 2 verbose; >2 debug)
DEFAULT_VERBOSITY=1
VERBOSITY="${VERBOSITY:-${DEFAULT_VERBOSITY}}"

# Path to Python executable
DEFAULT_PYTHON="python"
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"

#------------------------------------------------------------------------------

[ ${VERBOSITY} -ge 2 ] && {
    echo -e "\n=============================="
    echo "SETUP"
    echo -e "------------------------------"
    echo "TEST_DIR      : ${TEST_DIR}"
    echo "SKIP_MODULES  : ${SKIP_MODULES}"
    echo "DRY_RUN       : ${DRY_RUN}"
    echo "VERBOSITY     : ${VERBOSITY}"
    echo "PYTHON        : ${PYTHON}"
    echo -e "==============================\n"
}

# Modules to skip: convert cmma-separated list to bash array
SKIP_MODULES=(${SKIP_MODULES//,/ })

# Check the Python executable
case ${VERBOSITY} in
    0) ;&
    1) ${PYTHON} --version >/dev/null || exit 1 ;;
    *) ${PYTHON} --version || exit 1 ;;
esac

#------------------------------------------------------------------------------
# Main script
#------------------------------------------------------------------------------

main()
{
    local _name_="main"

    # Paths of all directories containing package directories
    local package_roots=($(collect_package_roots))
    pdbg ${_name_} "package_roots" "${package_roots[@]}"

    # Paths of all test files
    local test_file_paths=($(\find "${TEST_DIR}" -name 'test_*.py' | \sort))
    pdbg ${_name_} "test_file_paths" "${test_file_paths[@]}"

    # Module import paths of all test files
    local modules_pypath_root=($(collect_modules))
    pdbg ${_name_} "module_pypaths" "${module_pypaths[@]}"

    # Run tests
    local n_tot=0
    local n_pass=0
    local n_fail=0
    local n_skip=0
    for module_pypath_root in "${modules_pypath_root[@]}"
    do
        pdbg ${_name_} "module_pypath_root" "${module_pypath_root}"
        run_test_suite $(echo "${module_pypath_root/@/ }")
        case ${?} in
            0) n_pass=$((n_pass + 1)) ;;
            1) n_fail=$((n_fail + 1)) ;;
            2) n_skip=$((n_skip + 1)) ;;
        esac
        n_tot=$((n_tot + 1))
    done
    [ ${n_tot} -ne ${#modules_pypath_root[@]} ] && {
        echo "warning: expected ${#modules_pypath_root[@]} test modules, but counted ${n_tot}" >&2
    }
    [ ${VERBOSITY} -ge 2 ] && echo ''
    [ ${VERBOSITY} -ge 1 ] && {
        local pc="$(\printf "%3d%%" $((100 * n_pass / (n_tot - n_skip))))"
        echo "[ ${pc} ] ${n_pass} passed, ${n_fail} failed of ${n_tot} total, ${n_skip} skipped"
    }
    return ${n_fail}
}

collect_modules()
{
    local _name_="collect_modules"
    local test_file_path
    local modules=()
    for test_file_path in ${test_file_paths[@]}
    do
        pdbg ${_name_} "test_file_path" "${test_file_path}"
        local found=false
        local package_root
        for package_root in ${package_roots[@]}
        do
            pdbg ${_name_} "package_root" "${package_root}"
            startswith "${test_file_path}" "${package_root}" || continue
            found=true
            pdbg ${_name_} "match" "${package_root} in ${test_file_path}"
            local n=$((${#package_root} + 1))
            local module="$(echo "${test_file_path:n: -3}" | \sed -e 's@/@.@g')"
            module+="@${package_root}"
            pdbg ${_name_} "module" "${module}"
            modules+=("${module}")
        done
        ${found} || {
            echo "error: no package_root found for test_file_path"\
                 "${test_file_path} among ${package_roots[@]}" >&2
            return 1
        }
    done
    echo "${modules[@]}" | \sort -u
}

collect_package_roots()
{
    local _name_="collect_package_roots"

    # Paths of all (sub)package directories
    local subpackage_paths=($(dirname $(\find "${TEST_DIR}" -name __init__.py)))
    pdbg ${_name_} "subpackage_paths" "${subpackage_paths[@]}"

    # Paths of all package directories
    local package_paths=($(elim_subpaths "${subpackage_paths[@]}"))
    pdbg ${_name_} "package_paths" "${package_paths[@]}"

    # Paths of all directories containing package directories
    dirname "${package_paths[@]}" | \sort -u
}

run_test_suite()
{
    local _name_="run_test_suite"
    local module_pypath="${1}"
    local module_root="${2}"
    pdbg ${_name_} "module_pypath" "${module_pypath}"
    pdbg ${_name_} "module_root" "${module_root}"
    local test_name="${module_pypath} @ ${module_root}"

    # Check if test suite shall be skipped
    local skip_module
    for skip_module in ${SKIP_MODULES[@]}
    do
        echo "${module_pypath}" | \grep -q "${skip_module//./\\.}" && {
            [ ${VERBOSITY} -ge 1 ] && echo "[ SKIP ] ${test_name}"
            return 2
        }
    done

    # Make packages accessible to Python
    local pypath="${module_root}:${PYTHONPATH}"
    pdbg ${_name_} "pypath" "${pypath}"

    # Run tests
    [ ${VERBOSITY} -ge 3 ] && echo ''
    [ ${VERBOSITY} -ge 2 ] && echo -e "[      ] ${test_name}"
    local cmd="PYTHONPATH=${pypath} ${PYTHON} -m ${module_pypath}"
    pdbg ${_name_} "command" "${cmd}"
    if ${DRY_RUN}
    then
        [ ${VERBOSITY} -ge 1 ] && echo "[  DRY ] ${test_name}"
    else
        case ${VERBOSITY} in
            0) ;&
            1) eval "${cmd}" >/dev/null 2>&1 ;;
            *) eval "${cmd}" ;;
        esac
        local stat=${?}
        case ${VERBOSE}/${stat} in
            0/*) ;;
            */0) echo "[ PASS ] ${test_name}" ;;
            */*) echo "[ FAIL ] ${test_name}" ;;
        esac
    fi
    return ${stat}
}

#------------------------------------------------------------------------------
# Utilities
#------------------------------------------------------------------------------

# Print debug message
pdbg()
{
    [ ${VERBOSITY} -le 2 ] && return 0
    local fmt="%-15s"
    local i
    for i in $(seq $((${#} - 2)))
    do
        fmt+=" : %-20s"
    done
    fmt+=" : %s"
    printf "${fmt}\n" "${@}" >&2
}

# Check whether STR starts with SUBSTR
startswith()
{
    local _name_="starswith"
    local str="${1}"
    local substr="${2}"
    echo "${str}" | \grep -q "^${substr}"
    local stat=${?}
    case ${stat} in
    (0) pdbg ${_name_} "'${str}' starts with '${substr}'" ;;
    (*) pdbg ${_name_} "'${str}' doesn't start with '${substr}'" ;;
    esac
    return ${stat}
}

# From a list of paths, remove those that are subpaths of others
elim_subpaths()
{
    local _name_="elim_subpaths"
    local paths=("${@}")
    local roots=()
    local path
    for path in "${paths[@]}"
    do
        pdbg ${_name_} "path" "${path}"
        is_root=true
        for path_ref in "${paths[@]}"
        do
            [ "${path_ref}" == "${path}" ] && continue
            pdbg ${_name_} "path_ref" "${path_ref}"
            startswith "${path}" "${path_ref}" && {
                is_root=false
                break
            }
        done
        pdbg ${_name_} "is_root" "${is_root}"
        ${is_root} && roots+=("${path}")
        pdbg ${_name_} "roots" "${roots[@]}"
    done
    echo "${roots[@]}" | \sort -u
}

#------------------------------------------------------------------------------

main "${@}"

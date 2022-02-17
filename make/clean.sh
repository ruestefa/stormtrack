#!/bin/bash


cli()
{
    local actions=(${@})
    possible_actions=(python cython pyc build test all)
    usage="$(basename ${0}) ACTION[S]\nACTIONS: ${possible_actions[@]}"
    [ "${#actions[@]}" -eq 0 ] && { echo -e "${usage}" >&2; return 1; }
    calls=()
    for action in "${actions[@]}"; do
        case "${action}" in
            python) calls+=(clean_python);;
            cython) calls+=(clean_cython);;
            pyc) calls+=(clean_pyc);;
            build) calls+=(clean_build);;
            test) calls+=(clean_test);;
            all) calls+=(clean_all);;
            *) echo -e "error: invalid action '${action}'\n${usage}" >&2; return 1;;
        esac
    done
    for call in "${calls[@]}"; do
        "${call}" || return
    done
}


clean_python()
{
    echo -e "[clean_python] removing Python build artifacts"
    \rm -vrf "build/"
    \rm -vrf "dist/"
    \rm -vrf ".eggs/"
    \find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec \rm -vrf '{}' \+
    \find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg-info' -exec \rm -vrf '{}' \+
}


clean_cython()
{
    echo -e "[clean_cython] removing Cython build artifacts"
    \find src -name '*.html' -exec rm -vf {} \+
    \find src -name '*.c' -exec rm -vf {} \+
    \find src -name '*.so' -exec rm -vf {} \+
}


clean_pyc()
{
    echo -e "[clean_pyc] removing Python file artifacts"
    \find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec \rm -vrf '{}' \+
    \find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec \rm -vrf '{}' \+
    \find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec \rm -vrf '{}' \+
    \find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec \rm -vrf '{}' \+
}


clean_build()
{
    echo -e "[clean_build] removing all build artifacts"
    clean_python || { echo "error in clean_python" >&2; return 1; }
    clean_cython || { echo "error in clean_cython" >&2; return 1; }
    clean_pyc || { echo "error in clean_pyc" >&2; return 1; }
}


clean_test()
{
    echo -e "[clean_test] removing testing artifacts"
    \rm -vrf ".tox/"
    \rm -vf ".coverage"
    \rm -vrf "htmlcov/"
    \rm -vrf ".pytest_cache"
    \rm -vrf ".mypy_cache"
}


clean_all()
{
    echo -e "[clean_all] clean up everything"
    clean_build || { echo "error in clean_build" >&2; return 1; }
    clean_test || { echo "error in clean_test" >&2; return 1; }
}


cli "${@}"

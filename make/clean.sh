#!/bin/bash

echo "error: script not tested yet" >&2
exit 1


clean_python()
{
    echo -e "\n[make clean-python] removing Python build artifacts"
    \rm -rf "build/"
    \rm -rf "dist/"
    \rm -rf ".eggs/"
    \rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
    \rm -ff $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.egg' -exec echo "rm -ff '{}'" \;)
}


clean_cython()
{
    echo -e "\n[make clean-cython] removing Cython build artifacts"
    \find src -name '*.html' -exec rm -f {} \+
    \find src -name '*.c' -exec rm -f {} \+
    \find src -name '*.so' -exec rm -f {} \+
}


clean_pyc()
{
    echo -e "\n[make clean-pyc] removing Python file artifacts"
    \rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyc'       -exec echo "rm -rf '{}'" \;)
    \rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*.pyo'       -exec echo "rm -rf '{}'" \;)
    \rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '*~'          -exec echo "rm -rf '{}'" \;)
    \rm -rf $(\find . -not -path './venv*' -and -not -path './ENV*' -name '__pycache__' -exec echo "rm -rf '{}'" \;)
}


clean_test()
{
    echo -e "\n[make clean-test] removing testing artifacts"
    \rm -rf ".tox/"
    \rm -f ".coverage"
    \rm -rf "htmlcov/"
    \rm -rf ".pytest_cache"
    \rm -rf ".mypy_cache"
}


# TODO

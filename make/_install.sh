#!/bin/bash

echo "error: script not tested yet" >&2
exit 1

PREFIX=
VENV_DIR=
PIP_OPTS=


dev_install()
{
    echo -e "\n[make install-dev] installing the package as editable with development dependencies"
    # conda install --yes --prefix "${VENV_DIR}" --file requirements/dev-requirements.txt  # pinned
    echo conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.in  # unpinned
    echo conda install --yes --prefix "${VENV_DIR}" --file requirements/dev-requirements.in  # unpinned
    # ${PREFIX}python -m pip install -U pip
    echo ${PREFIX}python -m pip install --editable . ${PIP_OPTS}
    echo ${PREFIX}pre-commit install
    echo ${PREFIX}stormtrack -V
}


install()
{
	echo -e "\n[make install] installing the package"
	conda env update --prefix "${VENV_DIR}" --file=environment.yml
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.txt  # pinned
	# conda install --yes --prefix "${VENV_DIR}" --file requirements/requirements.in  # unpinned
	# ${PREFIX}python -m pip install -U pip
	echo ${PREFIX}python -m pip install . ${PIP_OPTS}
	echo ${PREFIX}stormtrack -V
}


# TODO

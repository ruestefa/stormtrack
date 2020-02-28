SHELL := /bin/bash

.DEFAULT_GOAL := help

#==============================================================================
# Options
#==============================================================================

ECHO_PREFIX ?= \nMAKE: #OPT Prefix of command echos
IGNORE_VENV ?= 0#OPT Don't create and/or use a virtual environment.
INSTALL ?= 0#OPT Whether to install project and dependencies if necessary.
VENV_DIR ?= venv#OPT Path to virtual environment to be created and/or used.
VENV_NAME ?= stormtrack#OPT Name of virtual environment if one is created.

#------------------------------------------------------------------------------

PREFIX_VENV = ${VENV_DIR}/bin/#
export PREVIX_VENV

ifneq (${IGNORE_VENV}, 0)
# Ignore virtual env
PREFIX ?=#
else
ifneq (${VIRTUAL_ENV},)
# Virtual env is active
PREFIX =#
else
# Virtual env is not active
PREFIX = ${PREFIX_VENV}
endif
endif
export PREFIX

#
# Targets conditional on ${INSTALL}.
#
# Use these to specify the respective targets for post-installation targets
# like ``test``, for which installation is optionally activated with INSTALL=1.
#
# Note that for interdependencies between venv/installation targets -- like
# of ``install`` on ``venv``, or if ``install-dev`` on ``install-test`` -- the
# targets are hard-coded w/o these variables, such that they are always executed
# regardless of ${INSTALL}.
#
ifeq (${INSTALL}, 0)
	_VENV :=
	_INSTALL :=
	_INSTALL_EDIT :=
	_INSTALL_TEST :=
	_INSTALL_DEV :=
	_INSTALL_PINNED :=
	_INSTALL_TEST_PINNED :=
else
	_VENV := venv
	_INSTALL := install
	_INSTALL_EDIT := install-edit
	_INSTALL_TEST := install-test
	_INSTALL_DEV := install-dev
	_INSTALL_PINNED := install-pinned
	_INSTALL_TEST_PINNED := install-test-pinned
endif
export _VENV
export _INSTALL
export _INSTALL_EDIT
export _INSTALL_TEST
export _INSTALL_DEV
export _INSTALL_PINNED
export _INSTALL_TEST_PINNED

#==============================================================================
# Python script: Print help
#==============================================================================

define PRINT_HELP_PY
import re
import sys

def parse_makefile(lines):
    options = {}
    commands = {}
    rx_opt = re.compile(
        r"^(?P<name>[A-Z_]+) *\?= *(?P<value>[^ ]*) *#OPT (?P<help>.*)? *$$"
    )
    rx_cmd = re.compile(
        r"^(?P<name>[a-zA-Z_-]+):.*?#CMD (?P<help>.*) *$$"
    )
    for line in lines:
        match_opt = rx_opt.match(line)
        match_cmd = rx_cmd.match(line)
        if match_opt:
            m = match_opt
            help = m.group("help").split(r"\n")
            options[m.group("name")] = (m.group("value"), help)
        elif match_cmd:
            m = match_cmd
            commands[m.group("name")] = m.group("help").split(r"\n")
    return options, commands

def format_options(items):
    s = ""
    for name, (value, help) in items.items():
        name_value = f"{name}={value}"
        for idx, line in enumerate(help):
            s += f"  {name_value if idx == 0 else '':25s} {line}\n"
    return s.rstrip()

def format_commands(items):
    s = ""
    for name, help in items.items():
        for idx, line in enumerate(help):
            s += f"  {name if idx == 0 else '':25s} {line}\n"
    return s.rstrip()

options, commands = parse_makefile(sys.stdin)

print(f"""\
Usage: make COMMAND [OPTIONS]

Options:
{format_options(options)}

Commands:
{format_commands(commands)}
""")
endef
export PRINT_HELP_PY

#==============================================================================
# Python script: Open browser
#==============================================================================

define BROWSER_PY
import os
import sys
import webbrowser

try:
    from urllib import pathname2url
except:
    from urllib.request import pathname2url

webbrowser.open(f"file://{pathname2url(os.path.abspath(sys.argv[1]))}")
endef
export BROWSER_PY
browser = ${PREFIX}python -c "$$BROWSER_PY"

#==============================================================================
# Help
#==============================================================================

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PY" < $(MAKEFILE_LIST)

#==============================================================================
# Cleanup
#==============================================================================

.PHONY: clean-all
clean-all: clean-venv clean-test clean-build clean-cython clean-pyc #CMD Remove all build, test, coverage and Python artifacts.
	@echo -e "${ECHO_PREFIX}cleaning up"

.PHONY: clean-build
clean-build: #CMD Remove Python build artifacts.
	@echo -e "${ECHO_PREFIX}removing build artifacts"
	\rm -rf "build/"
	\rm -rf "dist/"
	\rm -rf ".eggs/"
	\find . -not -path "${VENV_DIR}" -name '*.egg' -exec rm -f {} \+
	\find . -not -path "${VENV_DIR}" -name '*.egg-info' -exec rm -rf {} \+

.PHONY: clean-cython
clean-cython: #CMD Remove Cython build artifacts.
	\find src -name '*.html' -exec rm -f {} \+
	\find src -name '*.c' -exec rm -f {} \+
	\find src -name '*.so' -exec rm -f {} \+

.PHONY: clean-pyc
clean-pyc: #CMD Remove Python file artifacts.
	@echo -e "${ECHO_PREFIX}removing Python file artifacts"
	\find . -not -path "${VENV_DIR}" -name '*.pyc' -exec rm -f {} \+
	\find . -not -path "${VENV_DIR}" -name '*.pyo' -exec rm -f {} \+
	\find . -not -path "${VENV_DIR}" -name '*~' -exec rm -f {} \+
	\find . -not -path "${VENV_DIR}" -name '__pycache__' -exec rm -rf {} \+

.PHONY: clean-test
clean-test: #CMD Remove testing artifacts.
	@echo -e "${ECHO_PREFIX}removing testing artifacts"
	\rm -rf ".tox/"
	\rm -f ".coverage"
	\rm -rf "htmlcov/"
	\rm -rf ".pytest_cache"

.PHONY: clean-venv
clean-venv: #CMD Remove virtual environment.
	@echo -e "${ECHO_PREFIX}removing virtual environment at '${VENV_DIR}'"
	\rm -rf "${VENV_DIR}"

#==============================================================================
# Virtual Environments
#==============================================================================

.PHONY: venv
venv: #CMD Create a virtual environment.
ifeq (${IGNORE_VENV}, 0)
	$(eval PREFIX = ${PREFIX_VENV})
	@export PREFIX
ifeq (${VIRTUAL_ENV},)
	@echo -e "${ECHO_PREFIX}creating virtual environment '${VENV_NAME}' at '${VENV_DIR}'"
	which python
	python --version
	python -m venv ${VENV_DIR} --prompt='${VENV_NAME}'
	${PREFIX}python --version
	${PREFIX}python -m pip install -U pip
endif
endif

#==============================================================================
# Installation
#==============================================================================

.PHONY: install
install: venv #CMD Install the package with unpinned runtime dependencies.
	@echo -e "${ECHO_PREFIX}installing the package"
	${PREFIX}python -m pip install .

.PHONY: install-edit
install-edit: venv #CMD Install the package as editable with unpinned runtime dependencies.
	@echo -e "${ECHO_PREFIX}installing the package as editable"
	${PREFIX}python -m pip install -e .

.PHONY: install-pinned
install-pinned: venv #CMD Install the package with pinned runtime dependencies.
	@echo -e "${ECHO_PREFIX}installing the package with pinned dependencies"
	${PREFIX}python -m pip install -r requirements/run-pinned.txt
	${PREFIX}python -m pip install .

.PHONY: install-test-pinned
install-test-pinned: venv #CMD Install the package with pinned runtime and testing dependencies.
	@echo -e "${ECHO_PREFIX}installing the package as editable with pinned testing dependencies"
	${PREFIX}python -m pip install -r requirements/test-pinned.txt
	${PREFIX}python -m pip install -e .

.PHONY: install-test
install-test: install-edit #CMD Install the package with unpinned runtime and testing dependencies.
	@echo -e "${ECHO_PREFIX}installing the package with testing dependencies"
	${PREFIX}python -m pip install -r requirements/test-unpinned.txt

.PHONY: install-dev
install-dev: install-test #CMD Install the package as editable with unpinned runtime,\ntesting, and development dependencies.
	@echo -e "${ECHO_PREFIX}installing the package as editable with testing and development dependencies"
	${PREFIX}python -m pip install -r requirements/dev-unpinned.txt

#==============================================================================
# Version control
#==============================================================================

.PHONY: git
git: clean-all #CMD Initialize a git repository and make initial commit.
ifeq ($(shell git tag >/dev/null 2>&1 && echo 0 || echo 1), 0)
	@echo -e "${ECHO_PREFIX}git already initialized"
else
	@echo -e "${ECHO_PREFIX}initializing Git repository"
	\git init
	\git add .
	\git commit -m 'initial commit'
	\git --no-pager log -n1 --stat
endif

#==============================================================================
# Versioning
#==============================================================================

.PHONY: bump-patch
bump-patch: ${_INSTALL_DEV} #CMD Bump patch component of version number (x.y.Z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component"
	${PREFIX}bumpversion patch

.PHONY: bump-minor
bump-minor: ${_INSTALL_DEV} #CMD Bump minor component of version number (x.Y.z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component"
	${PREFIX}bumpversion minor

.PHONY: bump-major
bump-major: ${_INSTALL_DEV} #CMD Bump minor component of version number (X.y.z), incl. git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component"
	${PREFIX}bumpversion major

.PHONY: bump-patch-dry
bump-patch-dry: ${_INSTALL_DEV} #CMD Bump patch component of version number (x.y.Z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment patch component (dry run)"
	${PREFIX}bumpversion patch --no-commit --no-tag

.PHONY: bump-minor-dry
bump-minor-dry: ${_INSTALL_DEV} #CMD Bump minor component of version number (x.Y.z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment minor component (dry run)"
	${PREFIX}bumpversion minor --no-commit --no-tag

.PHONY: bump-major-dry
bump-major-dry: ${_INSTALL_DEV} #CMD Bump minor component of version number (X.y.z), without git commit and tag
	@echo -e "${ECHO_PREFIX}bumping version number: increment major component (dry run)"
	${PREFIX}bumpversion major --no-commit --no-tag

#==============================================================================
# Formatting and linting
#==============================================================================

.PHONY: format
format: ${_INSTALL_DEV} #CMD Reformat the code to conform with standards like PEP 8.
	@echo -e "${ECHO_PREFIX}reformatting the code"
	${PREFIX}black src tests

.PHONY: lint
lint: ${_INSTALL_DEV} #CMD Check the code style.
	@echo -e "${ECHO_PREFIX}checking the code style (linting)"
	${PREFIX}flake8 src tests

#==============================================================================
# Testing
#==============================================================================

.PHONY: test
test: ${_INSTALL_TEST} #CMD Run all tests with the default Python version.
	@echo -e "${ECHO_PREFIX}running tests"
	${PREFIX}pytest

.PHONY: test-cov
test-cov: ${_INSTALL_TEST} #CMD Check code coverage of tests.
	@echo -e "${ECHO_PREFIX}running tests with coverage check"
	${PREFIX}pytest --cov=src

.PHONY: test-cov-html
test-cov-html: ${_INSTALL_TEST} #CMD Check code coverage of tests and show results in browser.
	@echo -e "${ECHO_PREFIX}running tests with coverage check and browser report"
	${PREFIX}pytest --cov=src --cov-report=html
	${browser} htmlcov/index.html

.PHONY: test-all
test-all: ${_INSTALL_TEST} #CMD Run tests on all specified Python versions with tox.
	@echo -e "${ECHO_PREFIX}running tests in isolated environments"
	${PREFIX}tox

#==============================================================================
# Documentation
#==============================================================================

# .PHONY: docs
# docs: ${_INSTALL_DEV} #CMD Generate HTML documentation, including API docs.
#	@echo -e "${ECHO_PREFIX}generating HTML documentation"
# 	\rm -f docs/stormtrack.rst
# 	\rm -f docs/modules.rst
# 	${PREFIX}sphinx-apidoc -o docs/ src/stormtrack
# 	$(MAKE) -C docs clean
# 	$(MAKE) -C docs html
# 	${browser} docs/_build/html/index.html

# .PHONY: servedocs
# servedocs: docs #CMD Compile the docs watching for changes.
#	@echo -e "${ECHO_PREFIX}continuously regenerating HTML documentation"
# 	${PREFIX}watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

#==============================================================================

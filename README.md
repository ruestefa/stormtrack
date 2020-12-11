# StormTrack

Identify two-dimensional features like cyclones and fronts and track them over time in high-resolution gridded weather and climate data.

## Features

- Identify two-dimensional features in any fields using thresholds
- Identify cyclones and fronts using pre-defined algorithms (see [R&uuml;dis&uuml;hli et al., 2020](https://wcd.copernicus.org/articles/1/675/2020/))
- Track the features over time using an algorithm based on relative overlap and feature size
- Control merging and splitting by limiting the number of features involved in a branching event
- Process the features and tracks as objects in Python
- Export the features and tracks to NetCDF for processing with standard tools

## Language

On its surface, StormTrack is a Python package.
However, it's core is written primarily in [Cython](https://cython.org/), which is installation involved compilation and therefore takes a bit of time.

## Installation

### Clone the repository

```bash
git clone git@github.com:ruestefa/stormtrack.git
cd stormtrack
```

Various commands for installation, testing and development are provided by the `Makefile`.
To see all available commands, run `make help`.
To see what's behind the commands (usually only a few commands), take a peek in the `Makefile`.


### Install the package

Install the package (and its dependencies) into a virtual environment:

```bash
# Location where you install packages
venv_dir="${HOME}/local/venvs/stormtrack"

make install VENV_DIR=${venv_dir} CHAIN=1
```

To make the commands available system-wide, link them to a location in `$PATH`:

```bash
# A directory that is in $PATH
bin_dir="${HOME}/local/bin"

cd ${bin_dir}

# Main tools
ln -s ${venv_dir}/bin/identify-features .
ln -s ${venv_dir}/bin/identify-front-fields .
ln -s ${venv_dir}/bin/track-features .

# Additional tools
ln -s ${venv_dir}/bin/group-tracks .
ln -s ${venv_dir}/bin/inspect-tracks .
ln -s ${venv_dir}/bin/project-tracks .
```

To use the commands, you don't need to activate the virtual environment that the tool and its dependencies have been installed into, that is handled automatically behind the scenes.

### Run the tests

To make sure that the tool runs on your system, you may want to run the tests before the installation.
These commands will install the package into a temporary local virtual environment, run the tests and clean up again:

```bash
make test-all CHAIN=1
make clean-all
```

If all goes well, you can confidently install and use the tool.

### Create sandboxes

In order to test the tools, StormTrack provides sandboxes, which are smallish test cases based on real data (which must not be used for other purposes).
They are most easily set up by using the virtual environment created during installation, be it locally (in which case `venv_dir="./venv"`) or elsewhere (with `VENV_DIR="${venv_dir}"` during install as described above):

```bash
# Working directory with lots of disk space
sandbox_path="${SCRATCH}/stormtrack"

source "${venv_dir}/bin/activate"
./sandboxes/create_sandboxes.py -h  # show options
./sandboxes/create_sandboxes.py "${sandbox_path}" --dry  # check what will be done
./sandboxes/create_sandboxes.py "${sandbox_path}"  # add -f when re-running
```

This will create a subdirectory `sandboxes` containing the individual sandboxes, copy (or link) the run scripts and download the input data.

NOTE: Currently no data is availble by default, so the script will fail, but the data may be provided upon personal request.

To run the test cases, you need the StormTrack commands in your `$PATH`, either by properly installing them or by just keeping the virtual environment active.

Example:

```bash
cd "${sandbox_path}/sandboxes/cyclones"
./identify_and_track_cyclones.sh 4  # use 4 parallel processes
```

The features and tracks will be stored in a custom format with several files per output period.
To explore them in Python, use the command `inspect-tracks`.
You may use `src/stormtrack/scripts/inspect_tracks.py` as a basis for your own scripts processing the features and tracks.
To export the features and tracks to NetCDF, use the command `project-tracks` (based on `src/stormtrack/scripts/project_tracks.py`).

---

### Credits

This project was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [MeteoSwiss-APN/mch-python-blueprint](https://github.com/MeteoSwiss-APN/mch-python-blueprint) project template.

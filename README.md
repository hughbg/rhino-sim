# rhino-sim

Simulation pipeline for the RHINO 21cm global signal experiment.

There are a few entities in this system that need to be understood: beams, beam specifications, notebooks, batch system for running notebooks.

The system is complex because it is designed to be automated from a single "make" command. To use a beam for simulation and calibration, as a Jupyter notebook, see the **Notebooks** section below.

## Beams

The beams that are used in the simulation notebook are either [UVBeam](https://pyuvdata.readthedocs.io/en/latest/uvbeam.html)s or sparse beams. Sparse beams are built on top of a UVBeam *power* beam, so ultimately all beams have to be turned into a UVBeam at some point. UVBeams are stored on disk in FITS files, which are loaded into a UVBeam object in Python when a notebook runs.  Some beams are generated by Matlab or [XFdtd](https://www.remcom.com/applications/antenna-simulation-design-software), and the output is a text file. Such beams have to be converted to a UVBeam/FITS, by using the appropriate `process_*.py` script. These scripts have to be modified as the format of the text files change, if they do.

UVBeams can be power beams or efield beams. Both beams can be used in simulations, but a sparse beam can only be created from a power beam. The type of UVBeam is indicated in the UVBeam object by the `beam_type` attribute. 

A sparse beam is created from a UVBeam/FITS when needed inside a running notebook, so sparse beams do not exist on disk. (There is no way to save a sparse beam). In fact the sparse beam is not used directly in the simulation notebook, because it is wrapped in a `sim_sparse_beam`. The purpose of the `sim_sparse_beam` is to do some pre-calculation and caching, which speeds up the use of the sparse beam. A sparse beam could be used by itself in the simulation, but it will be very slow. A sparse beam can be used directly in other notebooks and scripts that are not running simulations, for example the beam plotting notebook `plot_beam.ipynb`.

The simulation can also use pyuvsim [AnalyticBeam](https://github.com/RadioAstronomySoftwareGroup/pyuvsim/blob/main/src/pyuvsim/analyticbeam.py). This is a simple beam type and it can be used for testing. You can have a Gaussian beam, airy beam, uniform beam. These beams are created in the simulation notebook when they specified. Nothing exists on disk for these beams, they are a Python object.

### HERA UVBeam/Fits

They are not in this repository, they are on COSMA in /cosma8/data/dp270/dc-gars1. They would also be on NRAO lustre but  don't know where.

## Beam specifications

The beams are specified in `beams.yaml`. A specification for a beam can have several purposes: to point to a UVBeam/FITS file, to specify how to generate the UVBeam/FITS file, to specify parameters of the beam when it is used.

The simplest specification is like this:

    vivaldi: 
	    type: uvbeam
	    file: NF_HERA_Vivaldi_power_beam.fits

The name of the beam is `vivaldi`, and this is important because this is how beams are referred to everywhere, and is used to create unique file names. This beam is defined to be a UVBeam, and the file contains the beam.

Another more complicated specification:


    example_uan:
        type: uvbeam
        file: example_uan.beamfits
        build_config:
            freq_start: 60000000
            freq_end: 88000000
            nfreq: 280
            values_file: example984.uan

This is the same as the previous specification but there is an extra `build_config` section. The `build_config` section is *only* used by one of the `process_*.py` scripts. It specifies how to create the UVBeam/FITS file from a text file supplied by Matlab or some other software. That is, how to create `file` from `values_file`.

A different example:

    rhino1_sparse: 
        type: sparse
        file: rhino1.beamfits
        params:
            nmax: 80
            mmodes:
                start: -45
                end: 46
            Nfeeds: 2

The beam will be loaded from the file but is to be converted to a sparse beam and used as such. The `params` section specify parameters necessary to configure a sparse beam.

### Loading a beam into Python

Use the function `load_beam` in file beam_show_ops.py. It will produce a beam of the appropriate type. See plot_beam.py for how to use it.

## Notebooks

There are plenty of notebooks here but two of them do the 21cm simulation and calibration and analysis, for a beam. These are the "main" notebooks to use. The notebooks are:

- Global_signal_simulation.ipynb
- Global_signal_calibration.ipynb OR Global_signal_calibration_new.ipynb

These are used when everything is run automatically as a batch system. To use them in Jupyter, you need to create copies of them that are set for a particular beam. Do the following:

1. Come up with a name for your beam, e.g. MYBEAM. Add a config for it in beams.yaml.
2. If necessary, edit the Makefile to change CAL_NOTEBOOK to the calibration notebook you want.
3. Run these commands in a shell (make sure conda is activated in the shell) to generate notebooks fixed to MYBEAM:
    ```
    make MYBEAM_sim.ipynb
    make MYBEAM_cal.ipynb
    ```
    If `make` says "up to date" then the files already exist.
4. You can now load those two notebooks into Jupyter and run them in Jupyter. Run the sim notebook and then the cal notebook.

Other notebooks can be run as they are.

## Batch system

The sim notebook and the calibration notebook can be run for all beams by running the command `make -k`. However you must have make commands in the Makefile and an entry in beams.yaml for every beam you want to run. See the existing Makefile for examples. Also add your beam to the "all:" make rule.

The notebooks are run *outside* Jupyter. This done using the papermill package, which must be installed into your conda environment using pip. For each beam called BEAM the result notebooks will be `BEAM_sim,ipynb` and `BEAM_cal.ipynb`.

If you are running on a cluster that has the SLURM batch system installed, then the notebooks will be run as SLURM jobs through the queueing system. If youre not running on a SLURM cluster, the notebooks will be run as Linux processes. In either case, there will be log files produced, with the extension `.log`. The file `makeflow.out` will contain a log of what the batch system is doing, and is updated as the batch system runs. It will also contian a report as to whether each job or process exited with success or failure.  The jobs/processes are given a unique number in makeflow.log and the name of the logfile for each job/process will contain that number.


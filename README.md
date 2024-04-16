
# rhino-sim

Pipeline for the RHINO 21cm global signal experiment. Basic operation is to run a simulation notebook to generate an observation of 21 cm using RHINO. Then run the simulation result through a calibration and analysis notebook.

There are a few entities in this system that need to be understood: beams, beam specifications, notebooks, batch system for running notebooks.

The system is complex because it is designed to be automated as a batch system from a single "make" command. If you just want to run  simulation and calibration in Jupyter, see the **Notebooks** section below.

## Beams

There are several types of beams known to the system:

- UVBeam. On disk they are stored in a FITS file, in a notebook they are a Python object created from the FITS file.
- Text file beam. These are files produced by external packages like Matlab or [XFdtd](https://www.remcom.com/applications/antenna-simulation-design-software). They cannot be used unless they are converted to a UVBeam/FITS, by one of the `process_*.py` scripts.
- sparse_beam. These do not exist in a file, they only exist as a Python object. In a Python script or notebook they are created from a UVBeam/FITS.
- sim_sparse_beam. These do not exist in a file, they only exist as a Python object. They are created from a sparse_beam. This sim_sparse_beam can only be used within a vis_cpu simulation.
- pyuvsim [AnalyticBeam](https://github.com/RadioAstronomySoftwareGroup/pyuvsim/blob/main/src/pyuvsim/analyticbeam.py). This is a simple beam type and it can be used for testing. You can have a Gaussian beam, airy beam, uniform beam. They are a Python object only.

UVBeams can be power beams or efield beams. Both beams can be used in simulations, but a sparse beam can only be created from a power beam. The type of UVBeam is indicated in the UVBeam Python object by the `beam_type` attribute. 

The purpose of the `sim_sparse_beam` is to do some pre-calculation and caching, which speeds up the use of the sparse beam. A sparse beam could be used by itself in a simulation, but it will be very slow. A sparse beam can be used directly in other notebooks and scripts that are not running simulations, for example the beam plotting notebook `plot_beam.ipynb`.

### HERA UVBeam/Fits

These are supplied, and their names being with NF_. They are not in this repository, they are on COSMA in /cosma8/data/dp270/dc-gars1. They would also be on NRAO lustre but I don't know where.

### Loading a beam into Python

Use the function `load_beam` in file beam_show_ops.py. It will produce a beam of the appropriate type. See plot_beam.py for how to use it.

## Beam specifications

All beams must be specified in `beams.yaml`. A specification for a beam can have several purposes: to point to a UVBeam/FITS file, to specify how to generate the UVBeam/FITS file, to specify parameters of the beam when it is used.

The simplest specification is like this:

    vivaldi: 
	    type: uvbeam
	    file: NF_HERA_Vivaldi_power_beam.fits

The name of the beam is `vivaldi`, and this is important because this is how beams are referred to everywhere, and is used to create unique file names. This beam is defined to be a UVBeam, and the FITS file contains the beam.

Another more complicated specification:


    example_uan:
        type: uvbeam
        file: example_uan.beamfits
        build_config:
            freq_start: 60000000
            freq_end: 88000000
            nfreq: 280
            values_file: example984.uan

This is the same as the previous specification but there is an extra `build_config` section. The `build_config` section is *only* used by one of the `process_*.py` scripts. It specifies how to create the UVBeam/FITS file from a text file supplied by Matlab or some other software (in this case from Xfdtd). That is, how to create `file` from `values_file`.

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

The beam will be loaded from the FITS file but is to be converted to a sparse beam and used as such. The `params` section specify parameters necessary to configure a sparse beam.


## Notebooks

There are two important notebooks, one to run a simulation, and another to run a calibration of the simulation. The notebooks exist as templates and shouldn't be used by themselves. The notebooks are:

- Global_signal_simulation.ipynb
- Global_signal_calibration.ipynb OR Global_signal_calibration_new.ipynb

From these templates, notebooks should be generated that are fixed to a particular beam. The batch system does this automatically but you can generate notebooks to use in Jupyter. Follow these steps:

1. Come up with a name for your beam, e.g. MYBEAM. Add a config for it in beams.yaml.
2. If necessary, edit the Makefile to change CAL_NOTEBOOK to the calibration notebook you want.
3. Run these commands in a shell (make sure conda is activated in the shell) to generate notebooks fixed to MYBEAM:
    ```
    make MYBEAM_sim.ipynb
    make MYBEAM_cal.ipynb
    ```
    If `make` says "up to date" then the files already exist.
4. You can now load the two notebooks from 3. into Jupyter and run them in Jupyter. Run the sim notebook and then the cal notebook.

Other notebooks can be run as they are.

## Batch system

A simulation and subsequent calibration can be run for all beams by running the command `make -k` in a shell (with conda activated). You must have commands in the Makefile and an entry in beams.yaml for every beam you want to run. See the existing Makefile for examples. Also add your beam to the "all:" make rule.

The notebooks are run *outside* Jupyter. This done using the papermill package, which must be installed into your conda environment using pip. For each beam called BEAM the result notebooks will be `BEAM_sim.ipynb` and `BEAM_cal.ipynb`.

If you are running on a cluster that has the SLURM batch system installed, then the notebooks will be run as SLURM jobs through the queueing system. If you're not running on a SLURM cluster, the notebooks will be run as Linux processes. In either case, there will be log files produced, with the extension `.log`. The file `makeflow.out` will contain a log of what the batch system is doing, and is updated as the batch system runs. It will also contain a report as to whether each job or process exited with success or failure.  The jobs/processes are given a unique number in makeflow.log and the name of the logfile for each job/process will contain that number.

A sparse_beam is always converted to a sim_sparse_beam when the simulation runs.


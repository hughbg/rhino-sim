BATCH = python run_and_wait.py 

SIM_NOTEBOOK = Global_signal_simulation.ipynb 
CAL_NOTEBOOK = Global_signal_calibration.ipynb

all: vivaldi_cal.milled vivaldi_efield_cal.milled rhino1_cal.milled airy_cal.milled vivaldi_sparse_cal.milled rhino1_sparse_cal.milled

# ------
# Make the notebooks for a particular beam. This doesn't automatically detect what beams
# are in beams.yaml so you have to add what you want.

%_sim.ipynb: $(SIM_NOTEBOOK)
	python check_beam.py $*
	sed s/BEAM_SPEC/$*/ $(SIM_NOTEBOOK) > $@

%_cal.ipynb: $(CAL_NOTEBOOK)
	python check_beam.py $*
	sed s/BEAM_SPEC/$*/ $(CAL_NOTEBOOK) > $@

# ------ Vivali beam

vivaldi_sim.milled: beams.yaml NF_HERA_Vivaldi_power_beam.fits vivaldi_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sim.ipynb

vivaldi_cal.milled: vivaldi_cal.ipynb vivaldi_sim.milled
	$(BATCH) papermill.sh vivaldi_cal.ipynb

# ------ Vivaldi efield beam

vivaldi_efield_sim.milled: beams.yaml NF_HERA_Vivaldi_efield_beam.fits vivaldi_efield_sim.ipynb
	$(BATCH) papermill.sh vivaldi_efield_sim.ipynb

vivaldi_efield_cal.milled: vivaldi_efield_cal.ipynb vivaldi_efield_sim.milled
	$(BATCH) papermill.sh vivaldi_efield_cal.ipynb


# ------ RHINO matlab beam, note that there is a command to build a UVBeam/FITS from the RHINO Matlab output

rhino1.beamfits: process_matlab.py matlab_horn_351MHz_rot.dat matlab_horn_351MHz_rot_az.dat matlab_horn_351MHz_rot_za.dat beams.yaml
	python process_matlab.py rhino1

rhino1_sim.milled: beams.yaml rhino1.beamfits rhino1_sim.ipynb
	$(BATCH) papermill.sh rhino1_sim.ipynb

rhino1_cal.milled: rhino1_cal.ipynb rhino1_sim.milled
	$(BATCH) papermill.sh rhino1_cal.ipynb

# ------ Airy beam. This is an Analytic beam defined in pyuvsim

airy_sim.milled: beams.yaml airy_sim.ipynb
	$(BATCH) papermill.sh airy_sim.ipynb

airy_cal.milled: airy_cal.ipynb airy_sim.milled
	$(BATCH) papermill.sh airy_cal.ipynb

# ------ Vivaldi sparse beam

vivaldi_sparse_sim.milled: beams.yaml NF_HERA_Vivaldi_power_beam.fits vivaldi_sparse_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sparse_sim.ipynb

vivaldi_sparse_cal.milled: vivaldi_sparse_cal.ipynb vivaldi_sparse_sim.milled
	$(BATCH) papermill.sh vivaldi_sparse_cal.ipynb


# ------ RHINO sparse beam


rhino1_sparse_sim.milled: beams.yaml rhino1.beamfits rhino1_sparse_sim.ipynb
	$(BATCH) papermill.sh rhino1_sparse_sim.ipynb

rhino1_sparse_cal.milled: rhino1_sparse_cal.ipynb rhino1_sparse_sim.milled
	$(BATCH) papermill.sh rhino1_sparse_cal.ipynb


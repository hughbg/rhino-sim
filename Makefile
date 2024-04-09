BATCH = python run_and_wait.py 

SIM_NOTEBOOK = Global_signal_simulation.ipynb 
CAL_NOTEBOOK = Global_signal_calibration.ipynb

all: vivaldi_cal.done vivaldi_efield_cal.done rhino1_cal.done airy_cal.done vivaldi_sparse_cal.done rhino1_sparse_cal.done

# ------
# Make the notebooks for a particular beam

%_sim.ipynb: $(SIM_NOTEBOOK)
	python check_beam.py $*
	sed s/BEAM_SPEC/$*/ $(SIM_NOTEBOOK) > $@

%_cal.ipynb: $(CAL_NOTEBOOK)
	python check_beam.py $*
	sed s/BEAM_SPEC/$*/ $(CAL_NOTEBOOK) > $@

# ------ Vivali beam

sim_global_vivaldi_power_refl.uvh5: beams.yaml NF_HERA_Vivaldi_power_beam.fits vivaldi_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sim.ipynb

vivaldi_cal.done: vivaldi_cal.ipynb sim_global_vivaldi_power_refl.uvh5
	$(BATCH) papermill.sh vivaldi_cal.ipynb

# ------ Vivaldi efield beam

sim_global_vivaldi_efield_refl.uvh5: beams.yaml NF_HERA_Vivaldi_efield_beam.fits vivaldi_efield_sim.ipynb
	$(BATCH) papermill.sh vivaldi_efield_sim.ipynb

vivaldi_efield_cal.done: vivaldi_efield_cal.ipynb sim_global_vivaldi_efield_refl.uvh5
	$(BATCH) papermill.sh vivaldi_efield_cal.ipynb


# ------ RHINO matlab beam, note that there is a command to build a UVBeam/FITS from the RHINO Matlab output

rhino1.beamfits: process_matlab.py matlab_horn_351MHz_rot.dat matlab_horn_351MHz_rot_az.dat matlab_horn_351MHz_rot_za.dat beams.yaml
	python process_matlab.py rhino1

sim_global_rhino1_refl.uvh5: beams.yaml rhino1.beamfits rhino1_sim.ipynb
	$(BATCH) papermill.sh rhino1_sim.ipynb

rhino1_cal.done: rhino1_cal.ipynb sim_global_rhino1_refl.uvh5
	$(BATCH) papermill.sh rhino1_cal.ipynb

# ------ Airy beam. This is an Analytic beam defined in pyuvsim

sim_global_airy_refl.uvh5: beams.yaml airy_sim.ipynb
	$(BATCH) papermill.sh airy_sim.ipynb

airy_cal.done: airy_cal.ipynb sim_global_airy_refl.uvh5
	$(BATCH) papermill.sh airy_cal.ipynb

# ------ Vivaldi sparse beam

sim_global_vivaldi_sparse_refl.uvh5: beams.yaml NF_HERA_Vivaldi_power_beam.fits vivaldi_sparse_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sparse_sim.ipynb

vivaldi_sparse_cal.done: vivaldi_sparse_cal.ipynb sim_global_vivaldi_sparse_refl.uvh5
	$(BATCH) papermill.sh vivaldi_sparse_cal.ipynb


# ------ RHINO sparse beam


sim_global_rhino1_sparse_refl.uvh5: beams.yaml rhino1.beamfits rhino1_sparse_sim.ipynb
	$(BATCH) papermill.sh rhino1_sparse_sim.ipynb

rhino1_sparse_cal.done: rhino1_sparse_cal.ipynb sim_global_rhino1_sparse_refl.uvh5
	$(BATCH) papermill.sh rhino1_sparse_cal.ipynb


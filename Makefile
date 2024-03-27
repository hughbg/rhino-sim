BATCH = python run_and_wait.py 

SIM_NOTEBOOK = Global_signal_simulation.ipynb 
CAL_NOTEBOOK = Global_signal_calibration.ipynb

all: vivaldi_calibration.ipynb rhino1_calibration.ipynb airy_calibration.ipynb vivaldi_sparse_calibration.ipynb rhino1_sparse_calibration.ipynb

# ------

sim_global_vivaldi_power.refl.uvh5: $(SIM_NOTEBOOK) beams.yaml NF_HERA_Vivaldi_power_beam.fits beams.yaml
	sed s/BEAM_SPEC/vivaldi/ $(SIM_NOTEBOOK) > vivaldi_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sim.ipynb

vivaldi_calibration.ipynb: $(CAL_NOTEBOOK) sim_global_vivaldi_power_refl.uvh5
	sed s/SIM_FILE/sim_global_vivaldi_power_refl.uvh5/ $(CAL_NOTEBOOK) > vivaldi_calibration.ipynb
	$(BATCH) papermill.sh vivaldi_calibration.ipynb

# ------
rhino_matlab1.beamfits: process_matlab.py matlab_horn_351MHz_rot.dat matlab_horn_351MHz_rot_az.dat matlab_horn_351MHz_rot_za.dat beams.yaml
	python process_matlab.py rhino1

sim_global_rhino1_refl.uvh5: $(SIM_NOTEBOOK) rhino_matlab1.beamfits beams.yaml
	sed s/BEAM_SPEC/rhino1/ $(SIM_NOTEBOOK) > rhino_sim.ipynb
	$(BATCH) papermill.sh rhino_sim.ipynb

rhino1_calibration.ipynb: $(CAL_NOTEBOOK) sim_global_rhino1_refl.uvh5
	sed s/SIM_FILE/sim_global_rhino1_refl.uvh5/ $(CAL_NOTEBOOK) > rhino1_calibration.ipynb
	$(BATCH) papermill.sh rhino1_calibration.ipynb

# ------

sim_global_airy_refl.uvh5: $(SIM_NOTEBOOK) beams.yaml beams.yaml
	sed s/BEAM_SPEC/airy/ $(SIM_NOTEBOOK) > airy_sim.ipynb
	$(BATCH) papermill.sh airy_sim.ipynb

airy_calibration.ipynb: $(CAL_NOTEBOOK) sim_global_airy_refl.uvh5
	sed s/SIM_FILE/sim_global_airy_refl.uvh5/ $(CAL_NOTEBOOK) > airy_calibration.ipynb
	$(BATCH) papermill.sh airy_calibration.ipynb

# ------

sim_global_vivaldi_sparse_refl.uvh5: $(SIM_NOTEBOOK) beams.yaml NF_HERA_Vivaldi_power_beam.fits beams.yaml
	sed s/BEAM_SPEC/vivaldi_sparse/ $(SIM_NOTEBOOK) > vivaldi_sparse_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sparse_sim.ipynb

vivaldi_sparse_calibration.ipynb: $(CAL_NOTEBOOK) sim_global_vivaldi_sparse_refl.uvh5
	sed s/SIM_FILE/sim_global_vivaldi_sparse_refl.uvh5/ $(CAL_NOTEBOOK) > vivaldi_sparse_calibration.ipynb
	$(BATCH) papermill.sh vivaldi_sparse_calibration.ipynb

# ------


sim_global_rhino1_sparse_refl.uvh5: $(SIM_NOTEBOOK) rhino_matlab1.beamfits beams.yaml
	sed s/BEAM_SPEC/rhino1_sparse/ $(SIM_NOTEBOOK) > rhino_sim_sparse.ipynb
	$(BATCH) papermill.sh rhino_sim_sparse.ipynb

rhino1_sparse_calibration.ipynb: $(CAL_NOTEBOOK) sim_global_rhino1_sparse_refl.uvh5
	sed s/SIM_FILE/sim_global_rhino1_sparse_refl.uvh5/ $(CAL_NOTEBOOK) > rhino1_sparse_calibration.ipynb
	$(BATCH) papermill.sh rhino1_sparse_calibration.ipynb


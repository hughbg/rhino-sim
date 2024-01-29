BATCH = python run_and_wait.py 

sim_global_vivaldi_refl.uvh5: Global_signal_simulation.ipynb beams.yaml NF_HERA_Vivaldi_power_beam.fits
	sed s/BEAM_FILE/NF_HERA_Vivaldi_power_beam.fits/ Global_signal_simulation.ipynb > vivaldi_sim.ipynb
	$(BATCH) papermill.sh vivaldi_sim.ipynb

calibration_vivaldi: Global_signal_calibration.ipynb sim_global_vivaldi_refl.uvh5
	sed s/SIM_FILE/sim_global_vivaldi_refl.uvh5/ Global_signal_calibration.ipynb > vivaldi_calibration.ipynb
	$(BATCH) papermill.sh vivaldi_calibration.ipynb

all: sim_global_vivaldi_refl.uvh5

import numpy as np
from pyuvdata import UVBeam
import yaml, sys


with open('beams.yaml', 'r') as file:
    beams = yaml.safe_load(file)

build_config = beams[sys.argv[1]]["build_config"]

with open(build_config["az_file"], "r") as f:
    az = np.deg2rad(np.array(f.readline()[:-1].split(","), dtype=float))

with open(build_config["za_file"], "r") as f:
    za = np.deg2rad(np.array(f.readline()[:-1].split(","), dtype=float))

values = np.loadtxt(build_config["values_file"])

assert values.shape[0] == za.size and values.shape[1] == az.size, "data and axes don't match in size"


# Now start filling the UVBeam

uvb = UVBeam()

uvb.Naxes_vec = 1
uvb.Nfreqs = build_config["nfreq"]
uvb.antenna_type = "simple"
uvb.bandpass_array = np.array([np.ones(build_config["nfreq"])])
uvb.beam_type = "power"
uvb.data_array = np.zeros((1, 1, 2, build_config["nfreq"], za.size, az.size))      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
for i in range(build_config["nfreq"]):
    uvb.data_array[0, 0, 0, i] = values
    uvb.data_array[0, 0, 1, i] = values
uvb.data_normalization = "physical"
uvb.feed_name = "RHINO"
uvb.feed_version = "1.0"
uvb.freq_array = np.array([np.linspace(build_config["freq_start"], build_config["freq_end"], build_config["nfreq"])])
uvb.future_array_shapes = False     # Had trouble with True, it still seems to expect the old shape for bandpass_array
uvb.history = "Created by make_power_beam.py "+sys.argv[1]
uvb.model_name = "Unknown"
uvb.model_version = "1.0"
uvb.pixel_coordinate_system = "az_za"
uvb.telescope_name = "RHINO"
# Non-requried
uvb.Naxes1 = az.size
uvb.Naxes2 = za.size
#uvb.Ncomponents_vec None         Only required for E-field beams.
#uvb.Nelements None               Only reuired for phased array
#uvb.Nfeeds None                  Not required if beam_type is “power”.
#uvb.Npixels None                 Only required if pixel_coordinate_system is ‘healpix’.
uvb.Npols = 2
uvb.Nspws = 1
uvb.axis1_array = az
uvb.axis2_array = za
#uvb.basis_vector_array None      Not required if beam_type is “power”.
#uvb.coupling_matrix None         Required if antenna_type = “phased_array”. 
#uvb.delay_array None             Required if antenna_type = “phased_array”. 
#uvb.element_coordinate_system None      Required if antenna_type = “phased_array”. 
#uvb.element_location_array None         Required if antenna_type = “phased_array”. 
uvb.extra_keywords = {}  
#uvb.feed_array None              Not required if beam_type is “power”.
uvb.filename = sys.argv[1]
uvb.freq_interp_kind = "linear"
#uvb.gain_array None              Required if antenna_type = “phased_array”. 
uvb.loss_array = None
uvb.mismatch_array = None
#uvb.nside None                   Only required if pixel_coordinate_system is ‘healpix’.
#uvb.ordering None                Only required if pixel_coordinate_system is “healpix”.
#uvb.pixel_array None             Only required if pixel_coordinate_system is “healpix”.
uvb.polarization_array = np.array([-5, -6])
uvb.receiver_temperature_array = None
uvb.reference_impedance = None
uvb.s_parameters = None
uvb.spw_array = [0]
uvb.x_orientation = "east"

# This one isn't in the docs but I can't run vis_cpu without it
uvb.interpolation_function = "az_za_simple"


uvb.write_beamfits(beams[sys.argv[1]]["file"], run_check=True, check_extra=True, run_check_acceptability=True, check_auto_power=True, clobber=True)

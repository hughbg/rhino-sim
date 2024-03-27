"""
To do:
Pay attention to the values in the header, at moment assumptions are made
"""

import numpy as np
from pyuvdata import UVBeam
import yaml, sys

def strip_header(fname):
    f = open(fname)
    line = f.readline()
    while "end_<parameters>" not in line:
        line = f.readline()

    return f


dB_to_lin = lambda db: 10**(db/10)

# amp in dB and phase in degree
polar_to_re_im = lambda amp, phase: dB_to_lin(amp)*(np.cos(np.deg2rad(phase))+np.sin(np.deg2rad(phase))*1j)

with open('beams.yaml', 'r') as file:
    beams = yaml.safe_load(file)

build_config = beams[sys.argv[1]]["build_config"]

values_file = np.loadtxt(strip_header(build_config["values_file"]))
im = np.argmax(values_file[:, 2])
print("Peak at", values_file[im])

# These are in steps of 5
za = np.deg2rad(np.sort(np.unique(values_file[:, 0])))
az = np.deg2rad(np.sort(np.unique(values_file[:, 1])))

# Build values for 2 Naxes_vec and 1 pol and 1 freq
values = np.zeros((2, 1, za.size, az.size), dtype=complex)      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
for i in range(values_file.shape[0]):
    _za = int(values_file[i, 0])
    _az = int(values_file[i, 1])
    E_za = polar_to_re_im(values_file[i, 2], values_file[i, 4])
    E_az = polar_to_re_im(values_file[i, 3], values_file[i, 5])
    values[0, 0, _za//5, _az//5] = E_az
    values[1, 0, _za//5, _az//5] = E_za


# Need a basis vector array
basis_vector_array = np.zeros((2, 2, za.size, az.size))
basis_vector_array[0, 0, :, :] = 1
basis_vector_array[0, 1, :, :] = 0
basis_vector_array[1, 0, :, :] = 0
basis_vector_array[1, 1, :, :] = 1


# Now start filling the UVBeam

uvb = UVBeam()

uvb.Naxes_vec = 2
uvb.Nfreqs = build_config["nfreq"]
uvb.antenna_type = "simple"
uvb.bandpass_array = np.array([np.ones(build_config["nfreq"])])
uvb.beam_type = "efield"
uvb.data_array = np.zeros((2, 1, 2, build_config["nfreq"], za.size, az.size), dtype=complex)      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
for f in range(build_config["nfreq"]):
    for p in range(2):    # pol
        uvb.data_array[:, :, p, f, :, :] = values
uvb.data_normalization = "physical"
uvb.feed_name = "EXAMPLE UAN"
uvb.feed_version = "1.0"
uvb.freq_array = np.array([np.linspace(build_config["freq_start"], build_config["freq_end"], build_config["nfreq"])])
uvb.future_array_shapes = False     # Had trouble with True, it still seems to expect the old shape for bandpass_array
uvb.history = "Created by make_power_beam.py "+sys.argv[1]
uvb.model_name = "Unknown"
uvb.model_version = "1.0"
uvb.pixel_coordinate_system = "az_za"
uvb.telescope_name = "JPL"
# Non-required
uvb.Naxes1 = az.size
uvb.Naxes2 = za.size
uvb.Ncomponents_vec = 2           # Only required for E-field beams.
#uvb.Nelements None               Only required for phased array
uvb.Nfeeds = 2                    # Not required if beam_type is “power”.
#uvb.Npixels None                 Only required if pixel_coordinate_system is ‘healpix’.
uvb.Npols = 2
uvb.Nspws = 1
uvb.axis1_array = az
uvb.axis2_array = za
uvb.basis_vector_array = basis_vector_array      # Not required if beam_type is “power”.
#uvb.coupling_matrix None         Required if antenna_type = “phased_array”. 
#uvb.delay_array None             Required if antenna_type = “phased_array”. 
#uvb.element_coordinate_system None      Required if antenna_type = “phased_array”. 
#uvb.element_location_array None         Required if antenna_type = “phased_array”. 
uvb.extra_keywords = {}  
uvb.feed_array = ['x', 'y']            #  Not required if beam_type is “power”.
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

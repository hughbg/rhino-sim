import numpy as np
import yaml, sys
from pyuvdata import UVBeam

def strip_header(fname):
    phi_inc = theta_inc = magnitude = None    # need these to process the file
    f = open(fname)
    line = f.readline()
    while "end_<parameters>" not in line:
        if "phi_inc" in line:
            phi_inc = int(line.split()[1])
        if "theta_inc" in line:
            theta_inc = int(line.split()[1])
        if "magnitude" in line:
            magnitude_unit = line.split()[1]

        line = f.readline()

    assert phi_inc is not None and theta_inc is not None and magnitude_unit is not None, "required headers missing"

    return f, phi_inc, theta_inc, magnitude_unit

# Functions

dB_to_lin = lambda vals: 10**(vals/10)
no_change = lambda vals: vals

 
# amp in dB and phase in degree
polar_to_re_im = lambda f, amp, phase: f(amp)*(np.cos(np.deg2rad(phase))+np.sin(np.deg2rad(phase))*1j)


# Script

# Some freq info for now
freq_info = {
    "freq_start": 60000000,
    "freq_end": 88000000,
    "nfreq": 280
}

# UAN loading
values_file, za_inc, az_inc, magnitude_type = strip_header(sys.argv[1])
uan_values = np.loadtxt(values_file)


za = np.sort(np.unique(uan_values[:, 0])).astype(int)       # these always int?
az = np.sort(np.unique(uan_values[:, 1])).astype(int)


scale = no_change
if magnitude_type == "dB":
    scale = dB_to_lin

values = np.zeros((2, za.size, az.size), dtype=complex)      
for i in range(uan_values.shape[0]):
    _za = int(uan_values[i, 0])
    _az = int(uan_values[i, 1])
    E_za = polar_to_re_im(scale, uan_values[i, 2], uan_values[i, 4])
    E_az = polar_to_re_im(scale, uan_values[i, 3], uan_values[i, 5])
    values[0, _za//za_inc, _az//az_inc] = E_az
    values[1, _za//za_inc, _az//az_inc] = E_za

to_power = lambda e_az, e_za : (e_az*np.conj(e_az)+e_za*np.conj(e_za)).real

np.savetxt("za.txt", za)
np.savetxt("az.txt", az) 
np.savetxt("values.txt", to_power(values[0], values[1]))

# Now start filling the UVBeam

uvb = UVBeam()

TYPE_WANTED = "efield"                 # parameterise

if TYPE_WANTED == "efield":              # to do: merge efield/power

    basis_vector_array = np.zeros((2, 2, za.size, az.size))
    basis_vector_array[0, 0, :, :] = 1
    basis_vector_array[0, 1, :, :] = 0
    basis_vector_array[1, 0, :, :] = 0
    basis_vector_array[1, 1, :, :] = 1

    uvb.Naxes_vec = 2
    uvb.Nfreqs = freq_info["nfreq"]
    uvb.antenna_type = "simple"
    uvb.bandpass_array = np.atleast_2d(np.ones(freq_info["nfreq"]))
    uvb.beam_type = "efield"
    uvb.data_array = np.zeros((2, 1, 2, freq_info["nfreq"], za.size, az.size), dtype=complex)      # (Naxes_vec, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    for f in range(freq_info["nfreq"]):
        for p in range(1):    # fake 2 pols
            uvb.data_array[:, 0, p, f, :, :] = values
    uvb.data_normalization = "physical"
    uvb.feed_name = "UAN"
    uvb.feed_version = "1.0"
    uvb.freq_array = np.atleast_2d(np.linspace(freq_info["freq_start"], freq_info["freq_end"], freq_info["nfreq"]))
    uvb.history = "Created by process_uan.py "+sys.argv[1]
    uvb.model_name = "Unknown"
    uvb.model_version = "1.0"
    uvb.pixel_coordinate_system = "az_za"
    uvb.telescope_name = "XFDTD"
    # Non-required
    uvb.Naxes1 = az.size
    uvb.Naxes2 = za.size
    uvb.Ncomponents_vec = 2           # Only required for E-field beams.
    #uvb.Nelements None               Only required for phased array
    uvb.Nfeeds = 2                    # Not required if beam_type is “power”.
    #uvb.Npixels None                 Only required if pixel_coordinate_system is ‘healpix’.
    uvb.Npols = 2
    uvb.Nspws = 1
    uvb.axis1_array = np.deg2rad(az.astype(float))
    uvb.axis2_array = np.deg2rad(za.astype(float))
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
    
    
else:      # power
    
    
    power_values = to_power(values[0], values[1])
        
    uvb.Naxes_vec = 1
    uvb.Nfreqs = freq_info["nfreq"]
    uvb.antenna_type = "simple"
    uvb.bandpass_array = np.ones(freq_info["nfreq"])
    uvb.beam_type = "power"
    uvb.data_array = np.zeros((1, 2, freq_info["nfreq"], za.size, az.size))      # (Naxes_vec, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    for i in range(freq_info["nfreq"]):
        uvb.data_array[0, 0, i] = power_values
        uvb.data_array[0, 1, i] = power_values
    uvb.data_normalization = "physical"
    uvb.feed_name = "MATLAB"
    uvb.feed_version = "1.0"
    uvb.freq_array = np.linspace(freq_info["freq_start"], freq_info["freq_end"], freq_info["nfreq"])
    uvb.history = "Created by process_uan.py "+sys.argv[1]
    uvb.model_name = "Unknown"
    uvb.model_version = "1.0"
    uvb.pixel_coordinate_system = "az_za"
    uvb.telescope_name = "XFDTD"
    # Non-requireed
    uvb.Naxes1 = az.size
    uvb.Naxes2 = za.size
    #uvb.Ncomponents_vec None         Only required for E-field beams.
    #uvb.Nelements None               Only reuired for phased array
    #uvb.Nfeeds None                  Not required if beam_type is “power”.
    #uvb.Npixels None                 Only required if pixel_coordinate_system is ‘healpix’.
    uvb.Npols = 2
    uvb.Nspws = 1
    uvb.axis1_array = np.deg2rad(az.astype(float))
    uvb.axis2_array = np.deg2rad(za.astype(float))
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
    
    
uvb.write_beamfits(sys.argv[1][:-4]+".beamfits", run_check=True, check_extra=True, run_check_acceptability=True, check_auto_power=True, clobber=True)

# Attempt read back in
uvb = UVBeam()
uvb.read_beamfits(sys.argv[1][:-4]+".beamfits")

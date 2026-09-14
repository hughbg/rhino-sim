import numpy as np
import yaml, sys, os
from pyuvdata import UVBeam

# Multiple files at different frequencies

# Functions

def strip_header(fname):
    phi_inc = theta_inc = magnitude = frequency = None    # need these to process the file
    f = open(fname)
    line = f.readline()
    while "end_<parameters>" not in line:
        if "phi_inc" in line:
            phi_inc = int(line.split()[1])
        if "theta_inc" in line:
            theta_inc = int(line.split()[1])
        if "magnitude" in line:
            magnitude_unit = line.split()[1]
        if "frequencyHz" in line:
            frequency = line.split()[1]

        line = f.readline()

    assert phi_inc is not None and theta_inc is not None and magnitude_unit is not None \
                and frequency is not None, "required headers missing"

    return f, phi_inc, theta_inc, frequency, magnitude_unit

dB_to_lin = lambda vals: 10**(vals/10)
no_change = lambda vals: vals

# See https://github.com/hughbg/GENETIS_RHINO/blob/master/UAN_process_beam.pdf
to_power = lambda e_az, e_za : np.abs(e_az)+np.abs(e_za)
 
# amp in dB and phase in degree
polar_to_re_im = lambda f, amp, phase: f(amp)*(np.cos(np.deg2rad(phase))+np.sin(np.deg2rad(phase))*1j)


# Script ----------------------

with open('beams.yaml', 'r') as file:
    beams = yaml.safe_load(file)

file_prefix = str(beams[sys.argv[1]]["build_config"]["file_prefix"])
output_file = beams[sys.argv[1]]["file"]

all_uans = []

# Load multiple UAN files
index = 1
while os.path.exists(file_prefix+"_"+str(index)+".uan"):
    fname = file_prefix+"_"+str(index)+".uan"
    print(fname)

    # UAN loading
    values_file, za_inc, az_inc, frequency, magnitude_type = strip_header(fname)
    uan_values = np.loadtxt(values_file)
    
    za = np.sort(np.unique(uan_values[:, 0])).astype(int)       # these always int?
    az = np.sort(np.unique(uan_values[:, 1])).astype(int) 
    
    scale = no_change        # default
    if magnitude_type == "dB":
        scale = dB_to_lin
    
    values = np.zeros((2, za.size, az.size), dtype=complex)      
    for i in range(uan_values.shape[0]):
        _za = int(uan_values[i, 0])
        _az = int(uan_values[i, 1])

        # convert magnitudes to linear if necessary
        E_za = polar_to_re_im(scale, uan_values[i, 2], uan_values[i, 4])
        E_az = polar_to_re_im(scale, uan_values[i, 3], uan_values[i, 5])
        values[0, _za//za_inc, _az//az_inc] = E_az
        values[1, _za//za_inc, _az//az_inc] = E_za

    loaded = {
        "values": values,
        "za": za,
        "az": az,
        "frequency": float(frequency),
    }
    all_uans.append(loaded)

    index += 1

# Check shape consistency across the frequencies
for i in range(1, len(all_uans)):
    assert (all_uans[0]["za"] == all_uans[i]["za"]).all(), "not same shape"
    assert (all_uans[0]["az"] == all_uans[i]["az"]).all(), "not same shape"
    assert all_uans[0]["values"].shape == all_uans[i]["values"].shape, "not same shape"
    assert all_uans[0]["frequency"] < all_uans[i]["frequency"], "frequencies not in order"

# Get frequency list
frequencies = [ d["frequency"] for d in all_uans ]
print("Frequencies", frequencies)


# Now start filling the UVBeam

uvb = UVBeam()

TYPE_WANTED = "efield"                # parameterise

# We have 2 polazarizations which will contain the same data. For the efield that will still mean there are 
# 2 axes in each polarization.

if TYPE_WANTED == "efield":              # to do: merge efield/power

    basis_vector_array = np.zeros((2, 2, za.size, az.size))
    basis_vector_array[0, 0, :, :] = 1
    basis_vector_array[0, 1, :, :] = 0
    basis_vector_array[1, 0, :, :] = 0
    basis_vector_array[1, 1, :, :] = 1

    uvb.Naxes_vec = 2
    uvb.Nfreqs = len(frequencies)
    uvb.antenna_type = "simple"
    uvb.bandpass_array = np.atleast_2d(np.ones(len(frequencies)))
    uvb.beam_type = "efield"
    uvb.data_array = np.zeros((2, 1, 2, len(frequencies), za.size, az.size), dtype=complex)      # (Naxes_vec, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    for f in range(len(frequencies)):
        for p in range(2):    # fake 2 pols
            uvb.data_array[:, 0, p, f, :, :] = all_uans[f]["values"]
    uvb.data_normalization = "physical"
    uvb.feed_name = "UAN"
    uvb.feed_version = "1.0"
    uvb.freq_array = np.atleast_2d(frequencies)
    uvb.history = "Created by process_uan.py "+" ".join(sys.argv[1:])
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
    
    
elif TYPE_WANTED == "power":      # power

    # There are 2 polarizations but only a real value for each. And will be made the same.
    
    uvb.Naxes_vec = 1
    uvb.Nfreqs = len(frequencies)
    uvb.antenna_type = "simple"
    uvb.bandpass_array = np.atleast_2d(np.ones(len(frequencies)))
    uvb.beam_type = "power"
    uvb.data_array = np.zeros((1, 1, 2, len(frequencies), za.size, az.size))      # (Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)
    for i in range(len(frequencies)):
        v = all_uans[i]["values"]
        uvb.data_array[0, 0, 0, i] = to_power(v[0], v[1])
        uvb.data_array[0, 0, 1, i] = to_power(v[0], v[1])   
        
    uvb.data_normalization = "physical"
    uvb.feed_name = "UAN"
    uvb.feed_version = "1.0"
    uvb.freq_array = np.atleast_2d(frequencies)
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

else:
    raise RuntimeError("Invalid beam type")
    
print("Running checks")
uvb.write_beamfits(output_file, run_check=True, check_extra=True, run_check_acceptability=True, check_auto_power=True, clobber=True)

# Attempt read back in
uvb = UVBeam()
uvb.read_beamfits(output_file)

# Check some values, are inserted right -----------------------

# Pick za, az to use

index_z = all_uans[0]["za"].size//2
use_z = all_uans[0]["za"][index_z]
index_a = all_uans[i]["az"].size//2
use_a = all_uans[0]["az"][index_a]

if TYPE_WANTED == "efield":

    # Check values read from file against values placed in uvb data array, should all be in expected place. 
    # In the uvb data array the shape is (Naxes_vec=2, 1, Nfeeds or Npols=2, Nfreqs, Naxes2, Naxes1), 
    for i in range(len(all_uans)):
        v = all_uans[i]["values"]
        for j in range(2):      # E by axis
            file_value = v[j, index_z, index_a]
            # Pol 0
            uvb_data_array_value = uvb.data_array[j, 0, 0, i, index_z, index_a]
            assert np.isclose(file_value, uvb_data_array_value), "Values inserted wrongly - not matching up"
            # Pol 1
            uvb_data_array_value = uvb.data_array[j, 0, 1, i, index_z, index_a]
            assert np.isclose(file_value, uvb_data_array_value), "Values inserted wrongly - not matching up"
    
    # Check values read from file against values interpolated from beam, should all be in expected place. 
    # In the uvb data array the shape is (Naxes_vec=2, 1, Nfeeds or Npols=2, Nfreqs, Naxes2, Naxes1), 
    
    interp_beam = uvb.interp(az_array=np.deg2rad([use_a]), za_array=np.deg2rad([use_z]), freq_array=np.array(frequencies))[0]
    
    # Interp beam shape is (Naxes_vec=2, 1, Nfeeds or Npols=2, Nfreqs, Nlocations),
    
    for i in range(len(all_uans)):
        v = all_uans[i]["values"]
        for j in range(2):      # E by axis
            file_value = v[j, index_z, index_a]
            # Pol 0.. Note the 2 pols have the same values
            uvb_interp_value = interp_beam[j, 0, 0, i, 0]
    
            assert np.isclose(file_value, uvb_interp_value), "Values inserted wrongly - not matching up " \
                        +str(file_value)+" "+str(uvb_interp_value)
            # Pol 1
            uvb_data_array_value = interp_beam[j, 0, 1, i, 0]
            assert np.isclose(file_value, uvb_data_array_value), "Values inserted wrongly - not matching up " \
                        +str(file_value)+" "+str(uvb_interp_value)

elif TYPE_WANTED == "power":

    # Check values read from file against values placed in uvb data array, should all be in expected place. 
    # In the uvb data array the shape is (Naxes_vec=1, 1, Nfeeds or Npols=2, Nfreqs, Naxes2, Naxes1), 
    for i in range(len(all_uans)):
        v = all_uans[i]["values"]
        file_value = to_power(v[0, index_z, index_a], v[1, index_z, index_a])
        # Pol 0
        uvb_data_array_value = uvb.data_array[0, 0, 0, i, index_z, index_a]
        assert np.isclose(file_value, uvb_data_array_value), "Values inserted wrongly - not matching up"
        # Pol 1
        uvb_data_array_value = uvb.data_array[0, 0, 1, i, index_z, index_a]
        assert np.isclose(file_value, uvb_data_array_value), "Values inserted wrongly - not matching up"

    interp_beam = uvb.interp(az_array=np.deg2rad([use_a]), za_array=np.deg2rad([use_z]), freq_array=np.array(frequencies))[0]

    # Interp beam shape is (Naxes_vec=1, 1, Nfeeds or Npols=2, Nfreqs, Nlocations),
    
    for i in range(len(all_uans)):
        v = all_uans[i]["values"]

        file_value = to_power(v[0, index_z, index_a], v[1, index_z, index_a])
        # Pol 0.. Note the 2 pols have the same values
        uvb_interp_value = interp_beam[0, 0, 0, i, 0]

        assert np.isclose(file_value, uvb_interp_value), "Values inserted wrongly - not matching up " \
                    +str(file_value)+" "+str(uvb_interp_value)
        # Pol 1
        uvb_interp_value = interp_beam[0, 0, 1, i, 0]
        assert np.isclose(file_value, uvb_interp_value), "Values inserted wrongly - not matching up " \
                    +str(file_value)+" "+str(uvb_interp_value)

print("Checks ok")

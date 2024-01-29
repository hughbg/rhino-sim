from pyuvdata import UVBeam
import sys

uvb = UVBeam()
uvb.read(sys.argv[1])

print("Naxes_vec", uvb.Naxes_vec)
print("Nfreqs", uvb.Nfreqs)
print("antenna_type", uvb.antenna_type)
print("bandpass_array shape", uvb.bandpass_array.shape)
print("bandpass_array", uvb.bandpass_array)
print("beam_type", uvb.beam_type)
print("data_array shape", uvb.data_array.shape)
if uvb.future_array_shapes:
    if uvb.beam_type == "healpix":
        print("\t(Naxes_vec, Nfeeds or Npols, Nfreqs, Npixels)")
    else:
        print("\t(Naxes_vec, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)")
else:
    if uvb.beam_type == "healpix":
        print("\t(Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Npixels)")
    else:
        print("\t(Naxes_vec, 1, Nfeeds or Npols, Nfreqs, Naxes2, Naxes1)")
   

print("data_normalization", uvb.data_normalization)
print("feed_name", uvb.feed_name)
print("feed_version", uvb.feed_version)
print("freq_array", uvb.freq_array)
print("future_array_shapes", uvb.future_array_shapes)
print("history", uvb.history)
print("model_name", uvb.model_name)
print("model_version", uvb.model_version)
print("pixel_coordinate_system", uvb.pixel_coordinate_system)
print("telescope_name", uvb.telescope_name)

print("------------------------------------------------------")


print("Naxes1", uvb.Naxes1)
print("Naxes2", uvb.Naxes2)
print("Ncomponents_vec", uvb.Ncomponents_vec)
print("Nelements", uvb.Nelements)
print("Nfeeds", uvb.Nfeeds)
print("Npixels", uvb.Npixels)
print("Npols", uvb.Npols)
print("Nspws", uvb.Nspws)
print("axis1_array shape", uvb.axis1_array.shape)
print("axis2_array shape", uvb.axis2_array.shape)
print("basis_vector_array shape ", uvb.basis_vector_array.shape)
if uvb.beam_type == "healpix":
    print("\t(Naxes_vec, Ncomponents_vec, Npixels)")
else:
    print("\t(Naxes_vec, Ncomponents_vec, Naxes2, Naxes1)")
print("coupling_matrix", uvb.coupling_matrix)
print("delay_array", uvb.delay_array)
print("element_coordinate_system", uvb.element_coordinate_system)
print("element_location_array", uvb.element_location_array)
print("extra_keywords", uvb.extra_keywords)
print("feed_array", uvb.feed_array)
print("filename", uvb.filename)
print("freq_interp_kind", uvb.freq_interp_kind)
print("gain_array", uvb.gain_array)
print("loss_array", uvb.loss_array)
print("mismatch_array", uvb.mismatch_array)
print("nside", uvb.nside)
print("ordering", uvb.ordering)
print("pixel_array", uvb.pixel_array)
print("polarization_array", uvb.polarization_array)
print("receiver_temperature_array", uvb.receiver_temperature_array)
print("reference_impedance", uvb.reference_impedance)
print("s_parameters", uvb.s_parameters)
print("spw_array", uvb.spw_array)
print("x_orientation", uvb.x_orientation)

import numpy as np
from pyuvdata import UVBeam
import pyuvsim
from sparse_beam import sparse_beam, sim_sparse_beam
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def print_UVBeam(uvb):
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
    if uvb.basis_vector_array is not None: 
        print("basis_vector_array shape ", uvb.basis_vector_array.shape)
        if uvb.beam_type == "healpix":
            print("\t(Naxes_vec, Ncomponents_vec, Npixels)")
        else:
            print("\t(Naxes_vec, Ncomponents_vec, Naxes2, Naxes1)")
    else: print("uvb.basis_vector_array None")
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
    print("-----------------------------------")


def plot_beam(beam, freq, name, feed=0, save_to=None):
    # How to plot an efield beam is debatable. I'm combining the two axes by squaring and summing, which
    # is what is done in vis_cpu when autocorrelations are calculated (although source intensity is multiplied in).
    # For the Vivaldi beam this comes out the same as the power beam but I don't know if that generalizes.
    # I think it applies to efield beams with 2 orthogonal axes.
    
    to_power = lambda b : b[0]*np.conj(b[0])+b[1]*np.conj(b[1])
               
    beam.peak_normalize()
    
    _az = np.linspace(0., 2.*np.pi, 360)
    _za = np.linspace(0., 0.5*np.pi, 90)
    az, za = np.meshgrid(_az, _za)
    az = az.flatten()
    za = za.flatten()

    interp_beam = beam.interp(az_array=az, za_array=za, freq_array=np.array([freq]))[0]
    if beam.beam_type == "efield": 
            interp_beam = interp_beam[:, 0, feed, 0]
            interp_beam = to_power(interp_beam)
    else:
        interp_beam = interp_beam[0, 0, feed, 0]
        
    #print(interp_beam[10000:10003])
    
    r = np.sin(za)
    x = r*np.sin(az)
    y = r*np.cos(az)  

    grid_dim = 64

    # convert the x/y points to a grid location. x/y can be -1 to 1
    gi = np.round(np.interp(x, [-1, 1], [0, grid_dim-1])).astype(int)
    gj = np.round(np.interp(y, [-1, 1], [0, grid_dim-1])).astype(int)

    # Insert beam values into grid and weight
    grid = np.zeros((grid_dim, grid_dim), dtype=complex)
    np.add.at(grid, (gi, gj), interp_beam)
    weights = np.zeros((grid_dim, grid_dim))
    np.add.at(weights, (gi, gj), 1)

    grid /= weights

    #ax = plt.axes()
    #ax.remove()           # Causes problem in subplots

    im=plt.imshow(np.abs(grid), interpolation="quadric", norm=LogNorm(), cmap="rainbow")
    plt.xticks([])
    plt.yticks([])


    if True:
        points = np.arange(0, 2*np.pi, 0.01)
        for deg in [ 30, 45, 60, 75 ]:
            r = np.cos(deg*np.pi/180)
            x = r*np.cos(points)
            y = r*np.sin(points)
            plt.text(r/np.sqrt(2), -r/np.sqrt(2), "    $"+str(deg)+"^\circ$", c="w")
            plt.scatter(x, y, s=0.01, c='w', marker='o')
    

    plt.colorbar(im,fraction=0.04, pad=0.04)
    plt.title(name)

    for pos in ['right', 'top', 'bottom', 'left']: 
        plt.gca().spines[pos].set_visible(False) 
    

    if save_to is not None:
        plt.savefig(save_to)
         
def load_beam(beam_cfg, convert_sparse_to_sim_sparse=False, interp_freq_array=None, interp_freq_chunk=None):
    # beam_spec is a dictionary for a beam, from beams.yaml
    # convert_sparse_to_sim_sparse. ONLY use this for a beam being used by vis_cpu

    if beam_cfg["type"] == "uvbeam": 
        uvb = UVBeam()
        uvb.read_beamfits(beam_cfg["file"])
        if uvb.interpolation_function is None:
            uvb.interpolation_function = "az_za_simple"
        if uvb.freq_interp_kind is None:
            uvb.freq_interp_kind = "linear"
        return uvb
    elif beam_cfg["type"] == "sparse":

        if convert_sparse_to_sim_sparse:
            return sim_sparse_beam(beam_cfg["file"], beam_cfg["params"]["nmax"], mmodes=np.arange(beam_cfg["params"]["mmodes"]["start"], beam_cfg["params"]["mmodes"]["end"]), 
                         Nfeeds=beam_cfg["params"]["Nfeeds"], interp_freq_array=interp_freq_array, interp_freq_chunk=interp_freq_chunk)
        else:
            return sparse_beam(beam_cfg["file"], beam_cfg["params"]["nmax"], mmodes=np.arange(beam_cfg["params"]["mmodes"]["start"], beam_cfg["params"]["mmodes"]["end"]), 
                         Nfeeds=beam_cfg["params"]["Nfeeds"])

    elif beam_cfg["type"] == "analytic":
        if "params" in beam_cfg:
            return pyuvsim.AnalyticBeam(beam_cfg["name"], **beam_cfg["params"])
        else:
            return pyuvsim.AnalyticBeam(beam_cfg["name"])
    
    
        

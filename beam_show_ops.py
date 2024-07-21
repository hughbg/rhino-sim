import numpy as np
from pyuvdata import UVBeam
import pyuvsim
from sparse_beam import sparse_beam, sim_sparse_beam
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def print_UVBeam(uvb):
    """
    Print all the header information from a UVBeam.
    
    Parameters
    ----------
    uvb : UVBeam
        The beam to print

    """
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
    """
    Plot a beam. The plot is the beam projected flat onto the ground when looking from above.
    Colors indicate the beam values.

    Parameters
    ----------
    beam : Must be a beam that has an interp() function like UVBeam.
        The beam to plot.
    freq : float
        The frequency at which to plot the beam.
    feed: int
        See the UVBeam manual. Effectively the same as pols, i.e. X, Y, so can be 0, 1
    save_to: str
        A file name to save the plot to.

    """
    
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
        
def plot_horn(params, save_to=None):
    
    def line(start, end, colour="blue"):
        x = [start[0], end[0]]
        y = [start[1], end[1]]
        z = [start[2], end[2]]

        ax.plot3D(x, y, z, colour)

    def rect_around_y_axis(x_len, z_len, y_loc, colour="blue"):
        line((-x_len/2, y_loc, -z_len/2), (-x_len/2, y_loc, z_len/2), colour=colour)
        line((-x_len/2, y_loc, z_len/2), (x_len/2, y_loc, z_len/2), colour=colour)
        line((x_len/2, y_loc, z_len/2), (x_len/2, y_loc, -z_len/2), colour=colour)
        line((-x_len/2, y_loc, -z_len/2), (x_len/2, y_loc, -z_len/2), colour=colour)

    def line_parallel_y_axis(x, z, y_start, length, colour="blue"):
        line((x, y_start, z), (x, y_start+length, z))

    def waveguide(_w2, _h2, y_left, y_right):
        rect_around_y_axis(_w2, _h2, y_left, colour="blue")
        rect_around_y_axis(_w2, _h2, y_right, colour="blue")
        for x in [ -_w2/2, _w2/2 ]:
            for z in [ -_h2/2, _h2/2 ]:
                line_parallel_y_axis(x, z, y_left, y_right-y_left, colour="blue")

    def flare(_w1, _h1, _w2, _h2, y_left, y_right):
        rect_around_y_axis(_w1, _h1, y_right, colour="red")
        for x_dir in [ -1, 1]:
            for z_dir in [ -1, 1]:
                line((x_dir*_w2/2, y_left, z_dir*_h2/2), (x_dir*_w1/2, y_right, z_dir*_h1/2), colour="red")

    def feed(_h2, _l2, _f0, _f1, _h3, _y_start):
        # Where is the 0 of the feed?
        # is the feed offset [ x, y ]? or [y, x]?
        # is the feedwidth in the x or y direction? or both?
        # Ignoring feedwidth and assuming offset is x, y and is in centre of l2

        feed_x = _f0
        feed_y = _y_start+_l2/2+_f1
        line((feed_x, feed_y, -_h2/2), (feed_x, feed_y, -_h2/2+_h3), colour="black")
        
        

    # Convert to lengths, widths etc. (l1, w1, ...) on the diagram at 
    # https://uk.mathworks.com/help/antenna/ref/horn.html
    l1 = params["FlareLength"]
    h1 = params["FlareHeight"]
    w1 = params["FlareWidth"]
    l2 = params["Length"]
    h2 = params["Height"]
    w2 = params["Width"]
    h3 = params["FeedHeight"]
    w3 = params["FeedWidth"]
    f0 = params["FeedOffset0"]
    f1 = params["FeedOffset1"]

    # The midpoint of the horn along the y axis is placed at y=0
    # See https://uk.mathworks.com/help/antenna/ref/horn.html for
    # orientation of axes
    y_start = -(l1+l2)/2       
    #print("y start", y_start)


    max_x_offset = max(max(w1/2, w2/2), f0)
    max_y_offset = max(abs(y_start), y_start+l2/2+f1)
    max_z_offset = max(max(h1/2, h2/2), -h2/2+h3)
    
    max_offset = max(max_x_offset, max(max_y_offset, max_z_offset))


    plt.figure(figsize=(12, 12))
    ax = plt.axes(projection='3d')
    ax.set_aspect('equal', adjustable='box')

    #ax.view_init(10, 20)

    waveguide(w2, h2, y_start, y_start+l2)
    flare(w1, h1, w2, h2, y_start+l2, y_start+l2+l1)
    feed(h2, l2, f0, f1, h3, y_start)
    

    # Make the axes all the same. Should be done better. The point is that
    # the axes have to have the same scale. If the horn is 10m long
    # and only 2m high then how to make sure that it LOOKS like that in 3d plot.
    ax.set_xlim(-max_offset, max_offset)
    ax.set_ylim(-max_offset, max_offset)
    ax.set_zlim(-max_offset, max_offset)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if save_to is not None:
        plt.savefig(save_to)
        
        
def plot_mat_beam(data_file, az_file, za_file, name, log=True, save_to=None):
    """
    Plot a beam. The plot is the beam projected flat onto the ground when looking from above.
    Colors indicate the beam values.

    Parameters
    ----------
    data_file : matlab beam output za by az
        The beam to plot.
    az_file : matlab az file
    za_file: matlab za file
    save_to: str
        A file name to save the plot to.

    """
       
    
    az = np.deg2rad(np.loadtxt(az_file, delimiter=","))
    za = np.deg2rad(np.loadtxt(za_file, delimiter=","))  
    if za[0] != 0: 
        print("Warning, za needs to be 0 to 180. Adjusting to HG convention")
        za = np.deg2rad(np.arange(181, dtype=int)) # Trickey going on here, setup by the Tilt so that the first row in the data is za = 0

    data = np.loadtxt(data_file, delimiter=",")
    data = 10**(data/10)    # dB to power
    
    assert data.shape[0] == za.size and data.shape[1] == az.size
    
    za_coord = np.repeat(za, az.size)
    az_coord = np.tile(az, za.size)
    data = data.ravel()
    
    r = np.sin(za_coord)
    x = r*np.sin(az_coord)
    y = r*np.cos(az_coord)  

    grid_dim = 64

    # convert the x/y points to a grid location. x/y can be -1 to 1
    gi = np.round(np.interp(x, [-1, 1], [0, grid_dim-1])).astype(int)
    gj = np.round(np.interp(y, [-1, 1], [0, grid_dim-1])).astype(int)

    # Insert beam values into grid and weight
    grid = np.zeros((grid_dim, grid_dim), dtype=complex)
    np.add.at(grid, (gi, gj), data)
    weights = np.zeros((grid_dim, grid_dim))
    np.add.at(weights, (gi, gj), 1)

    grid /= weights

    #ax = plt.axes()
    #ax.remove()           # Causes problem in subplots

    plt.clf()
    if log: 
        im=plt.imshow(np.abs(grid), interpolation="quadric", norm=LogNorm(), cmap="rainbow")
    else:
        im=plt.imshow(np.abs(grid), interpolation="quadric", cmap="rainbow")
    plt.xticks([])
    plt.yticks([])


    if False:
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
    # shape.custom3d()
    #show()
    # or just show(h2) replace with image dump

    if save_to is not None:
        plt.savefig(save_to)
        
def plot_mat_beam_grid(data_file, az_file, za_file, name, log=True, vmin=None, save_to=None):
    az = np.loadtxt(az_file, delimiter=",", dtype=int)
    za = np.loadtxt(za_file, delimiter=",", dtype=int)  
    if za[0] != 0: 
        print("Warning, za needs to be 0 to 180. Adjusting to HG convention")
        za = np.arange(181, dtype=int) # Trickey going on here, setup by the Tilt so that the first row in the data is za = 0 regardless of what el says

    data = np.loadtxt(data_file, delimiter=",")
    data = 10**(data/10)    # dB to power
    
    assert data.shape[0] == za.size and data.shape[1] == az.size

    if vmin is not None:
        data = np.where(data<vmin, vmin, data)

    if log:
        plt.imshow(data.T, extent=[np.min(za), np.max(za), np.max(az), np.min(az)], 
                   norm=LogNorm())
    else:
        plt.imshow(data.T, extent=[np.min(za), np.max(za), np.max(az), np.min(az)], 
                   )
    plt.ylabel("az [deg]")
    plt.xlabel("za [deg]")
    plt.title("Power")
    plt.colorbar()

    plt.savefig(name)
        
def plot_cst_beam(pattern_file, name, log=True, save_to=None):
    """
    Plot a beam. The plot is the beam projected flat onto the ground when looking from above.
    Colors indicate the beam values.


    """
    
    cst = np.loadtxt(pattern_file, skiprows=2)
    data = 10**(cst[:, 2]/10)    # from dB
    
    az = cst[:, 0]
    el = cst[:, 1]
    
    data = data[el>0]
    az = az[el>0]
    el = el[el>0]
    
    peak_i = np.argmax(data)
    print("Max at az=", az[peak_i], "el=", el[peak_i])
    min_i = np.argmin(data)
    print("Min at az=", az[min_i], "el=", el[min_i])
    
    az = np.deg2rad(az)
    el = np.deg2rad(el)
    
    r = np.cos(el)
    x = r*np.sin(az)
    y = r*np.cos(az)  

    grid_dim = 64

    # convert the x/y points to a grid location. x/y can be -1 to 1
    gi = np.round(np.interp(x, [-1, 1], [0, grid_dim-1])).astype(int)
    gj = np.round(np.interp(y, [-1, 1], [0, grid_dim-1])).astype(int)

    # Insert beam values into grid and weight
    grid = np.zeros((grid_dim, grid_dim), dtype=complex)
    np.add.at(grid, (gi, gj), data)
    weights = np.zeros((grid_dim, grid_dim))
    np.add.at(weights, (gi, gj), 1)

    grid /= weights

    #ax = plt.axes()
    #ax.remove()           # Causes problem in subplots

    plt.clf()
    if log: 
        im=plt.imshow(np.abs(grid), interpolation="quadric", norm=LogNorm(), cmap="rainbow")
    else:
        im=plt.imshow(np.abs(grid), interpolation="quadric", cmap="rainbow")
    plt.xticks([])
    plt.yticks([])


    if False:
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
    # shape.custom3d()
    #show()
    # or just show(h2) replace with image dump

    if save_to is not None:
        plt.savefig(save_to)
        
def plot_cst_beam_grid(pattern_file, name, log=True, save_to=None):
    cst = np.loadtxt(pattern_file, skiprows=2)
    data = 10**(cst[:, 2]/10)    # from dB
    
    az = cst[:, 0].astype(int)
    el = cst[:, 1].astype(int)
    
    # az goes from -180 to 179. el goes from -90 to 90 
    
    grid = np.zeros((360, 181))
    for i in range(az.size):
        grid[az[i]+180, el[i]+90] = data[i]
 
    plt.matshow(grid, extent=[-90, 91, -180, 180])
    plt.ylabel("Theta [deg]")
    plt.xlabel("Phi [deg]")
    plt.title("Power")
    plt.savefig(name)


         
def load_beam(beam_cfg, convert_sparse_to_sim_sparse=False, interp_freq_array=None, interp_freq_chunk=None):
    """
    Load a beam, as per its specification in beams.yaml

    Parameters
    ----------
    beam_cfg : dict
        A dictionary containing beam information such as file name, beam type, parameters etc. Obtained from
        beams.yaml for a particular beam. 
    convert_sparse_to_sim_sparse : bool
        If True, and the beam is a sparse_beam, then convert it to a sim sparse_beam
    interp_freq_array: ndarray
        Must be supplied if the beam is a sparse_beam and convert_sparse_to_sim_sparse is True.
        An array that contains all the frequencies that will be used within a vis_cpu simulation using
        this beam. In order.
    interp_freq_chunk: int
        Must be supplied if the beam is a sparse_beam and convert_sparse_to_sim_sparse is True.
        Specifies how many frequencies the sim_sparse_beam should pre-calculate and cache when the beam
        is used in a vis_cpu simulation.

    Returns
    -------
    A beam object of the appropriate type
    """
  
    if beam_cfg["type"] == "uvbeam": 
        uvb = UVBeam()
        uvb.read_beamfits(beam_cfg["file"])
        try:
            if uvb.interpolation_function is None:
                uvb.interpolation_function = "az_za_simple"
        except:
            uvb.interpolation_function = "az_za_simple"
        try:
            if uvb.freq_interp_kind is None:
                uvb.freq_interp_kind = "linear"
        except:
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
        
    
def rotate_beam(beam, angle=-90, axis="x"):
    """
    Rotate a UVBeam and return a new one. The angle is in degrees and the
    rotation axis has to be specified as x, y, z, how this relates to az/za
    is undefined, you need to experiment.
    """

    def rotate(X, theta, axis='x'):
      '''Rotate multidimensional array `X` `theta` radians around axis `axis`'''
      c, s = np.cos(theta), np.sin(theta)
      if axis == 'x': return np.dot(X, np.array([
        [1.,  0,  0],
        [0 ,  c, -s],
        [0 ,  s,  c]
      ]))
      elif axis == 'y': return np.dot(X, np.array([
        [c,  0,  -s],
        [0,  1,   0],
        [s,  0,   c]
      ]))
      elif axis == 'z': return np.dot(X, np.array([
        [c, -s,  0 ],
        [s,  c,  0 ],
        [0,  0,  1.],
      ]))

    def to_xyz(az, za):

        r = np.sin(za)
        x = r*np.sin(az)
        y = r*np.cos(az)  
        z = np.cos(za)

        return [x, y, z]

    def to_az_za(xyz):
        za = np.arccos(xyz[2])
        az = np.arctan2(xyz[0], xyz[1])
        return [az, za]

    def find_index(a, val):
        assert np.min(a) <= val and val <= np.max(a), str(val)+" outside of range "+str(np.min(a))+" - "+str(np.max(a))
        return np.argmin(np.abs(a-val))

    uvb = beam.copy()
    az = uvb.axis1_array
    za = uvb.axis2_array
    assert np.min(az) >= 0, "Rotate won't work if there are negative az"
    assert np.min(za) >= 0, "Rotate won't work if there are negative za"


    all_az = []
    all_za = []
    all_values = []
    # Convert the values
    rotate_theta = np.deg2rad(angle)
    for i in range(len(az)):
        for j in range(len(za)):

            v = to_xyz(az[i], za[j])
            v = rotate(v, rotate_theta, axis=axis)
            az_za = to_az_za(v)

            # Wrap angles so they are positive
            new_az = az_za[0]
            if new_az < 0: new_az += 2*np.pi
            new_za = az_za[1] 
            if new_za < 0: new_za += 2*np.pi

            all_az.append(new_az)
            all_za.append(new_za)

            all_values.append(uvb.data_array[0, 0, 0, 0, j, i])
            
    rot_az_za = np.column_stack((np.array(all_az), np.array(all_za)))
            
    xi_az = np.tile(az, za.size)
    xi_za = np.repeat(za, az.size)
    xi = np.column_stack((xi_az, xi_za))

    interpolated_power = griddata(rot_az_za, np.array(all_values), xi, method="nearest")

    # Put into data_array
    uvb_data_array = np.zeros_like(uvb.data_array)
    weights = np.zeros_like(uvb.data_array)
    for i in range(xi.shape[0]):
        az_index = find_index(az, xi[i, 0])
        za_index = find_index(za, xi[i, 1])
        uvb_data_array[0, 0, 0, 0, za_index, az_index] += interpolated_power[i]
        weights[0, 0, 0, 0, za_index, az_index] += 1

    uvb_data_array = np.where(weights==0, uvb_data_array, uvb_data_array/weights)
    uvb.data_array = uvb_data_array
    
    return uvb
    
        

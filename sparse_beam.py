from pyuvdata import UVBeam
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.linalg import solve, lstsq
from scipy.interpolate import interp1d
import hashlib


class sparse_beam(UVBeam):
    
    def __init__(self, filename, nmax, mmodes, za_range=(0, 90), save_fn='', 
                 load=False, bound="Dirichlet", Nfeeds=None,
                 alpha=np.sqrt(1 - np.cos(46 * np.pi / 90)), **kwargs):
        """
        Construct the sparse_beam instance, which is a subclass of UVBeam

        Parameters:
            filename (str): 
                The filename to for the UVBeam compatible file.
            nmax (int): 
                Maximum number of radial (Bessel) modes.
            mmodes (array of int): 
                Which azimuthal (Fourier) modes to include.
            za_range (tuple): 
                Minimum and maximum zenith angle to read in.
            save_fn (str): 
                filepath to save a numpy array of coefficients once fitting is
                complete.
            load (bool): 
                Whether to load coefficients from save_fn rather than fitting 
                anew.
            bound (str): 
                Options are 'Dirichlet' or 'Neumann'. Refers to the boundary
                conditions for the Laplace equation in cylindrical coordinates
                i.e. it determines whether to use 0th order or 1st order
                Bessel functions.
            alpha (float):
                A constant to adjust where the boundary condition is satisfied
                on the disk. Default is slightly underneath the horizon.
        """
        super().__init__()
        self.bound = bound
        self.read_beamfits(filename, za_range=za_range, **kwargs)
        self.peak_normalize()
        
        self.az_array = self.axis1_array
        self.alpha = alpha
        self.rad_array = self.get_rad_array()
        
        self.az_grid, self.rad_grid = np.meshgrid(self.az_array, self.rad_array)
        self.ncoord = self.az_grid.size
        
        self.nmax = nmax
        self.mmodes = mmodes
        self.ncoeff_bess = self.nmax * len(self.mmodes)
        
        self.bess_matr, self.trig_matr = self.get_dmatr()

        
        self.daz = self.axis1_array[1] - self.axis1_array[0]
        self.drad = self.rad_array[1] - self.rad_array[0]
        self.dA = self.rad_grid * self.drad * self.daz
        
        self.save_fn = save_fn
        self.bess_fits, self.bess_beam = self.get_fits(load=load)
        self.bess_ps = np.abs(self.bess_fits)**2

        self.az_array_dict = {}
        self.za_array_dict = {}
        self.trig_matr_interp_dict = {}
        self.bess_matr_interp_dict = {}
        self.bt_matr_interp_dict = {}

        if Nfeeds is not None:         # power beam may not have the Nfeeds set
            assert self.Nfeeds is None, "Nfeeds already set on the beam"
            self.Nfeeds = Nfeeds

    def get_rad_array(self, za_array=None):
        """
        Get the radial coordinates corresponding to the zenith angles in 
        za_array, calculated according to the formula in Hydra Beam Paper I.

        Parameters:
            za_array (array):
                The zenith angles in question.

        Returns:
            rad_array (array):
                The radial coordinates corresponding to the zenith angles.
        """
        if za_array is None:
            za_array = self.axis2_array
        rad_array = np.sqrt(1 - np.cos(za_array)) / self.alpha
        
        return rad_array


    def get_bzeros(self):
        """
        Get the zeros of the appropriate Bessel function based on the
        desired basis specified by the 'bound' attribute, along with the 
        associated normalization.

        Returns:
            zeros (array): 
                The zeros of the appropriate Bessel function
            norm (array): 
                The normalization for the Bessel functions so that their L2 norm
                on the unit disc is 1.

        """
        if self.bound == "Dirichlet":
            zeros = jn_zeros(0, self.nmax)
            norm = jn(1, zeros)
        else:
            zeros = jn_zeros(1, self.nmax - 1)
            norm = jn(2, zeros)
            
            zeros = np.append(0, zeros)
            norm = np.append(1, norm)
        norm = norm / np.sqrt(2)

        return zeros, norm 
        
        
    def get_dmatr(self):
        """
        Compute the factored design matrix that maps from Fourier-Bessel 
        coefficients to pixel centers on the sky. Assumes az/za coordinates,
        AND uniform sampling in azimuth. Full design matrix is the tensor 
        product of these two factors.
                
        Returns:
            bess_matr (array, complex):
                Has shape (Nza, Nn). Contains the radial information of the
                design matrix.
            trig_matr (array, complex):
                Has shape (Naz, Nm). Contains the azimuthal information of the
                design matrix.
        """  
        zeros, norm = self.get_bzeros()
        
        Naz = len(self.az_array)
                
        bess_matr = jn(0, zeros[np.newaxis] * self.rad_array[:, np.newaxis]) / norm
        # Assume a regular az spacing and just make a unitary DFT matrix; better for fitting later
        trig_matr = np.exp(1.0j * np.array(self.mmodes)[np.newaxis] * self.az_array[:, np.newaxis]) / np.sqrt(Naz)
        
        return bess_matr, trig_matr
    
    def get_dmatr_interp(self, az_array, za_array):
        """
        Get a design matrix specialized for interpolation rather than fitting.

        Parameters:
            az_array (array):
                Azimuth angles to evaluate bassis functions at. Does not have to 
                be on a uniform grid, unlike the fitting design matrix. Should
                be 1-dimensional (i.e. flattened).
            za_array (array):
                Zenith angles to evaluate basis functions at. Should be 
                1-dimensional (i.e. flattened)

        Returns:
            bess_matr (array):
                The Bessel part of the design matrix.
            trig_matr (array, complex):
                The Fourier part of the design matrix.
        """
        rad_array = self.get_rad_array(za_array)
        zeros, norm = self.get_bzeros()
        Naz = len(self.az_array)

        bess_matr = jn(0, zeros[np.newaxis] * rad_array[:, np.newaxis]) / norm
        # Need to use the same normalization as in the dmatr used for fitting
        trig_matr = np.exp(1.j *  np.array(self.mmodes)[np.newaxis] * az_array[:, np.newaxis]) / np.sqrt(Naz)

        return bess_matr, trig_matr
    
    
    def get_fits(self, load=False):
        """
        Compute Fourier-Bessel fits up to nmax and for all m-modes.

        Parameters:
            load (bool): 
                Whether to load precomputed solutions

        Returns:
            fit_coeffs (array, complex):
                The coefficients for the Fourier-Bessel fit. Has shape
                (nmax, len(mmodes), Naxes_vec, 1, Nfeeds, Nfreqs)
            fit_beam (array, complex):
                The fit beam in sky coordinates. Has shape 
                (Naxes_vec, 1, Nfeeds, Nfreqs, Nza, Naz)
        """
        
        if load:
            fit_coeffs = np.load(f"{self.save_fn}_bess_fit_coeffs.npy")
            fit_beam = np.load(f"{self.save_fn}_bess_fit_beam.npy")
        else:
            # az_modes are discretely orthonormal so just project onto the basis
            # Saves loads of memory and time
            az_fit = self.data_array @ self.trig_matr.conj() # Naxes_vec, 1, Nfeeds, Nfreq, Nza, Nm

            BtB = self.bess_matr.T @ self.bess_matr
            Baz = self.bess_matr.T @ az_fit # Naxes_vec, 1, Nfeeds, Nfreq, Nn, Nm
            Baz = Baz.transpose(4, 5, 0, 1, 2, 3) # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq


            fit_coeffs = solve(BtB, Baz, assume_a="sym") # Nn, Nm, Naxes_vec, 1, Nfeeds, Nfreq

            # Apply design matrices to get fit beams
            fit_beam_az = np.tensordot(self.trig_matr, fit_coeffs, axes=((1,), (1,))) # Naz, Nn, Naxes_vec, 1, Nfeeds, Nfreq
            fit_beam = np.tensordot(self.bess_matr, fit_beam_az, axes=((1,), (1,))) # Nza, Naz, Naxes_vec, 1, Nfeeds, Nfreq
            fit_beam = fit_beam.transpose(2, 3, 4, 5, 0, 1)

            np.save(f"{self.save_fn}_bess_fit_coeffs.npy", fit_coeffs)
            np.save(f"{self.save_fn}_bess_fit_beam.npy", fit_beam)        

        return fit_coeffs, fit_beam
    
    def get_comp_inds(self, num_modes=64):
        """
        Get the indices for the num_modes most significant modes for each
        feed, polarization, and frequency.

        Parameters:
            num_modes (int): 
                The number of modes to use for the compressed fit.

        Returns:
            nmodes_comp (array, int):
                The radial mode numbers corresponding to the top num_modes 
                Fourier-Bessl modes, in descending order of significance. Has 
                shape (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes).
            mmodes_comp (array, int):    
                The azimuthal modes numbers corresponding to the top num_modes 
                Fourier-Bessel modes, in descending order of significance. Has
                shape (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes).
        """

        ps_sort_inds = np.argsort(self.bess_ps.reshape((self.ncoeff_bess, 
                                                        self.Naxes_vec, 1, 
                                                        self.Nfeeds, 
                                                        self.Nfreqs)),
                                  axis=0)
        # Highest modes start from the end
        sort_inds_flip = np.flip(ps_sort_inds, axis=0)[:num_modes]
        nmodes_comp, mmodes_comp = np.unravel_index(sort_inds_flip, 
                                                    (self.nmax, 
                                                     len(self.mmodes)))
        
        return nmodes_comp, mmodes_comp
    
    def get_comp_fits(self, num_modes=64):
        """
        Get the beam fit coefficients and fit beams in a compressed basis using
        num_modes modes for each polarization, feed, and frequency.

        Parameters:
            num_modes (int):
                The number of Fourier-Bessel modes to use for the compresed fit.
            
        Returns:
            fit_coeffs (array, complex):
                The coefficients for the Fourier-Bessel fit. Has shape
                (Naxes_vec, 1, Nfeeds, Nfreqs, num_modes)
            fit_beam (array, complex):
                The fit beam in sky coordinates. Has shape 
                (Naxes_vec, 1, Nfeeds, Nfreqs, Nza, Naz)
        """
        nmodes_comp, mmodes_comp = self.get_comp_inds(num_modes=num_modes)
        num_modes = nmodes_comp.shape[0]

        
        fit_coeffs, fit_beam = self.sparse_fit_loop(num_modes, nmodes_comp, mmodes_comp)
        
        return fit_coeffs, fit_beam

    def sparse_fit_loop(self, num_modes, nmodes_comp, mmodes_comp, 
                        fit_coeffs=None, do_fit=True, bess_matr=None,
                        trig_matr=None):
        """
        Do a loop over all the axes and fit/evaluate fit in position space.

        Parameters:
            num_modes (int): 
                Number of modes in the sparse fit.
            nmodes_comp (array of int):
                Which nmodes are being used in the sparse fit 
                (output of get_comp_inds method).
            mmodes_comp (array_of_int):
                Which mmodes are being used in the sparse fit 
                (output of get_comp_inds method).
            fit_coeffs (array, complex):
                Precomputed fit coefficients (if just evaluating).
            do_fit (bool):
                Whether to do the fit (set to False if fit_coeffs supplied).
            bess_matr (array):
                Bessel part of design matrix.
            trig_matr (array, complex):
                Fourier part of design matrix.
        
        Returns:
            fit_coeffs (array, complex; if do_fit is True):
                The newly calculated fit coefficients in the sparse basis.
            fit_beam (array, complex):
                The sparsely fit beam evaluated in position space.
        """
        # nmodes might vary from pol to pol, freq to freq. The fit is fast, just do a big for loop.
        interp_kwargs = [bess_matr, trig_matr, fit_coeffs]
        if do_fit:
            fit_coeffs = np.zeros([self.Naxes_vec, 1, self.Nfeeds, self.Nfreqs, 
                                  num_modes], dtype=complex)
            beam_shape = self.data_array.shape
            bess_matr = self.bess_matr
            trig_matr = self.trig_matr
        elif any([item is None for item in interp_kwargs]):
            raise ValueError("Must supply fit_coeffs, bess_matr, and trig_matr "
                             "if not doing fit.")
        else:
            Npos = bess_matr.shape[0]
            beam_shape = (self.Naxes_vec, 1, self.Nfeeds, self.Nfreqs, Npos)
        fit_beam = np.zeros(beam_shape, dtype=complex)
        
        for vec_ind in range(self.Naxes_vec):
            for feed_ind in range(self.Nfeeds):
                for freq_ind in range(self.Nfreqs):
                    if do_fit:
                        dat_iter = self.data_array[vec_ind, 0, feed_ind, freq_ind]
                    nmodes_iter = nmodes_comp[:, vec_ind, 0, feed_ind, freq_ind]
                    mmodes_iter = mmodes_comp[:, vec_ind, 0, feed_ind, freq_ind]
                    unique_mmodes_iter = np.unique(mmodes_iter)
       
                    for mmode in unique_mmodes_iter:
                        mmode_inds = mmodes_iter == mmode
                        # Get the nmodes that this mmode is used for
                        nmodes_mmode = nmodes_iter[mmode_inds] 

                        bess_matr_mmode = bess_matr[:, nmodes_mmode]
                        trig_mode = trig_matr[:, mmode]

                        if do_fit:
                            az_fit_mmode = dat_iter @ trig_mode.conj() # Nza

                            fit_coeffs_mmode = lstsq(bess_matr_mmode, az_fit_mmode)[0]
                            fit_coeffs[vec_ind, 0, feed_ind, freq_ind, mmode_inds] = fit_coeffs_mmode
                            fit_beam[vec_ind, 0, feed_ind, freq_ind] += np.outer(bess_matr_mmode @ fit_coeffs_mmode, trig_mode)
                        else:
                            fit_coeffs_mmode = fit_coeffs[vec_ind, 0, feed_ind, freq_ind, mmode_inds]
                            fit_beam[vec_ind, 0, feed_ind, freq_ind] += (bess_matr_mmode @ fit_coeffs_mmode) * trig_mode
                        
        if do_fit:
            return fit_coeffs,fit_beam
        else:
            return fit_beam
    
    def interp(self, sparse_fit=False, fit_coeffs=None, az_array=None, 
               za_array=None, reuse_spline=False, freq_array=None,
               freq_interp_kind="cubic",
               **kwargs):
        """
        A very paired down override of UVBeam.interp that more resembles
        pyuvsim.AnalyticBeam.interp. Any kwarg for UVBeam.interp that is not
        explicitly listed in this version of interp will do nothing.

        Parameters:
            sparse_fit (bool): 
                Whether a sparse fit is being supplied. If False (default), just
                uses the full fit specified at instantiation.
            fit_coeffs (bool):
                The sparse fit coefficients being supplied if sparse_fit is 
                True.
            az_array (array): 
                Flattened azimuth angles to interpolate to.
            za_array (array):
                Flattened zenith angles to interpolate to.
            reuse_spline (bool):
                Whether to reuse the spatial design matrix for a particular 
                az_array and za_array (named to keep consistency with UVBeam).
            freq_array (array):
                Frequencies to interpolate to. If None (default), just computes
                the beam at all frequencies in self.freq_array.
            freq_interp_kind (str or int):
                Type of frequency interpolation function to use. Default is a
                cubic spline. See scipy.interpolate.interp1d 'kind' keyword
                documentation for other options.

        Returns:
            beam_vals (array, complex):
                The values of the beam at the interpolated 
                frequencies/spatial positions. Has shape 
                (Naxes_vec, 1, Npols, Nfreqs, Npos).
        """
        if az_array is None and za_array is None and freq_array is not None:
            new_beam = super().interp(freq_array=freq_array, new_object=True, run_check=False)
            new_beam.bess_matr, new_beam.trig_matr = new_beam.get_dmatr()
            new_beam.bess_fits, new_beam.bess_beam = new_beam.get_fits()
            new_beam.bess_ps = np.abs(new_beam.bess_fits)**2
            new_beam.az_array_dict = {}
            new_beam.za_array_dict = {}
            new_beam.trig_matr_interp_dict = {}
            new_beam.bess_matr_interp_dict = {}
            new_beam.bt_matr_interp_dict = {}
            return new_beam

        
        if az_array is None:
            raise ValueError("Must specify an azimuth array.")
        if za_array is None:
            raise ValueError("Must specify a zenith-angle array.")
        
        if reuse_spline:
            az_hash = hashlib.sha1(az_array).hexdigest()
            za_hash = hashlib.sha1(za_array).hexdigest()
            if (az_hash in self.az_array_dict) and (za_hash in self.za_array_dict):
                trig_matr = self.trig_matr_interp_dict[az_hash]
                bess_matr = self.bess_matr_interp_dict[za_hash]
                bt_matr = self.bt_matr_interp_dict[(az_hash, za_hash)]
            else:
                self.az_array_dict[az_hash] = az_array
                self.za_array_dict[za_hash] = za_array
    
                bess_matr, trig_matr = self.get_dmatr_interp(az_array, za_array)
                bt_matr = trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis]
                
                self.trig_matr_interp_dict[az_hash] = trig_matr
                self.bess_matr_interp_dict[za_hash] = bess_matr
                self.bt_matr_interp_dict[(az_hash, za_hash)] = bt_matr
        else:
            bess_matr, trig_matr = self.get_dmatr_interp(az_array, za_array)
            bt_matr = trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis]
        
        if freq_array is None:
            bess_fits = self.bess_fits
        else:
            freq_array = np.atleast_1d(freq_array)
            assert freq_array.ndim == 1, "Freq array for interp must be exactly 1d"
            
            # FIXME: More explicit and complete future_array_shapes compatibility throughout code base desired
            if self.freq_array.ndim > 1:
                freq_array_knots = self.freq_array[0]
            else:
                freq_array_knots = self.freq_array
            bess_fits_interp = interp1d(freq_array_knots, self.bess_fits, axis=5,
                                        kind=freq_interp_kind)
            bess_fits = bess_fits_interp(freq_array)
        
        if sparse_fit:
            if freq_array is not None:
                raise NotImplementedError("Frequency interpolation is not "
                                          "implemented for sparse_fit=True.")
            num_modes = fit_coeffs.shape[-1]
            nmodes_comp, mmodes_comp = self.get_comp_inds(num_modes)
            beam_vals = self.sparse_fit_loop(num_modes, nmodes_comp, 
                                             mmodes_comp, fit_coeffs=fit_coeffs,
                                             do_fit=False, bess_matr=bess_matr,
                                             trig_matr=trig_matr)
        else:
            beam_vals = np.tensordot(bt_matr, bess_fits, axes=2).transpose(1, 2, 3, 4, 0)
        if self.beam_type == "power":
            # FIXME: This assumes you are reading in a power beam and is just to get rid of the imaginary component
            beam_vals = np.abs(beam_vals)
            
        return beam_vals, None
    
    def efield_to_power(*args, **kwargs):
        raise NotImplementedError("efield_to_power is not implemented yet.")
    
    def efield_to_pstokes(*args, **kwargs):
        raise NotImplementedError("efield_to_pstokes is not implemented yet.")

class FreqTimeCache:
    def __init__(self, all_freq, freq_chunk_size, master_beam):
        self.all_freq = all_freq
        self.freq_chunk_size = freq_chunk_size
        self.master_beam = master_beam
        self.reset()

    def reset(self):
        self.times = []              # At each time, there are several frequencies passed to interp(). This is the results - the data.
        self.beam = None
        

    def fast_interp(self, az_array, za_array):
        # Use the functions in sparse beam
        bess_matr, trig_matr = self.beam.get_dmatr_interp(az_array, za_array)
        bt_matr = trig_matr[:, np.newaxis] * bess_matr[:, :, np.newaxis]
        beam_vals = np.tensordot(bt_matr, self.beam.bess_fits, axes=2).transpose(1, 2, 3, 4, 0)
        if self.beam.beam_type == "power":
            beam_vals = np.abs(beam_vals)
        return beam_vals

    def fetch(self, freq, time, **kwargs):      # UVBeam interp has no *args, only **kwargs
        
        def find(a, val):         
            index = (np.abs(a-val)).argmin()

            if np.isclose(a[index], val):
                return index
            else:
                return -1

        if self.beam is None:
            # Fetch a chunk of frequencies. Since nothing is here, time must be 0. Check it.
            assert time == 0, "Strange to initialize everything with a time != 0"
            assert len(self.times) == 0, "Supposed to be no freqs present but there is data"

            # Extract the frequencies to use, being with the requested freq, and then a bunch more
            index = find(self.all_freq, freq)
            assert index != -1, "freq is not in the list supplied at the start"
           

            # Now actually get the data. Not just this specific frequency but a lot more we will cache
            self.beam = self.master_beam.interp(freq_array=self.all_freq[index:index+self.freq_chunk_size], new_object=True, run_check=False)
            #print("Load1", len(self.beam.freq_array[0]), "freq", self.beam.freq_array[0][0], "-", self.beam.freq_array[0][-1], "Time", time)

            # Now reset some things done in init, for new beam
            self.beam.bess_matr, self.beam.trig_matr = self.beam.get_dmatr()
            self.beam.bess_fits, self.beam.bess_beam = self.beam.get_fits()
            self.beam.bess_ps = np.abs(self.beam.bess_fits)**2
    
            self.beam.az_array_dict = {}
            self.beam.za_array_dict = {}
            self.beam.trig_matr_interp_dict = {}
            self.beam.bess_matr_interp_dict = {}
            self.beam.bt_matr_interp_dict = {}

            interp_vals = self.fast_interp(kwargs["az_array"], kwargs["za_array"])

            self.times.append(interp_vals)               # Contains lots of frequencies

            return interp_vals[:, :, :, 0:1, :], None   # Return just requested frequency

        else:
            freq_index = find(self.beam.freq_array[0], freq)     # Remember beam has multiple freqs according to chunk
            if freq_index >= 0:
                # We have the frequency, but check the time
                if time < len(self.times):
                    
                    # We have the time, so return the data

                    # This can only occur when we have gone through all the times for self.freq_present[0]. Check it.
                    assert find(self.beam.freq_array[0], freq) != -1, "something wrong with the sequencing"
                    #print("return", freq, time, "from cache")
                    return self.times[time][:, :, :, freq_index:freq_index+1, :], None

                else:
                    # We have the frequency, but not the time. This can only occur when we are scanning through
                    # the times for self.freq_present[0]+chunk and we need the next time. Check.
                    assert freq == self.beam.freq_array[0, 0] and time == len(self.times), "something wrong with the sequencing"

                    #print("Load2", len(self.beam.freq_array[0]), "freq", self.beam.freq_array[0][0], "-", self.beam.freq_array[0][-1], "Time", time)
                    interp_vals = self.fast_interp(kwargs["az_array"], kwargs["za_array"])
        
                    self.times.append(interp_vals)               # Contains lots of frequencies
        
                    return interp_vals[:, :, :, 0:1, :], None   # Return just requested frequency
            else:
                # There are frequencies present but not the one we want. This means we must have consumed all the cached frequencies and times.
                # Should be requesting the next frequency at time 0. Check.
                index_next = find(self.all_freq, freq)
                assert index_next != -1 and time == 0, "something wrong with the sequencing"

                # Reset some things and then go around to get the cache reloaded
                self.reset()          

                return self.fetch(freq, time, **kwargs)

    def report(self):
        if self.freq_present is None:
            assert len(self.times) == 0, "Supposed to be no freqs present but there is data"
            print("Freqs present: none")
        else:
            print(len(self.freq_present), "freq present:", self.freq_present, "Times present:", len(self.times))
                


class sim_sparse_beam:
    def __init__(self, *args, **kwargs):


        self.in_sim = False

        # Extract the params we want and strip them out of the kwargs
        assert "interp_freq_array" in kwargs, "must pass the whole frequency list to sim_sparse_beam"
        self.interp_freq_array = kwargs["interp_freq_array"]
        kwargs.pop("interp_freq_array")

        if "interp_freq_chunk" in kwargs:
            self.interp_freq_chunk = kwargs["interp_freq_chunk"]
            kwargs.pop("interp_freq_chunk")
        else:
            self.interp_freq_chunk = len(self.interp_freq_array)

        # Create sparse_beam 
        self.master_beam = sparse_beam(*args, **kwargs)
        self.master_beam.interpolation_function = "az_za_simple"
        self.master_beam.freq_interp_kind = "linear"
        self.beam_type = self.master_beam.beam_type
        self.polarization_array = self.master_beam.polarization_array

        self.reset()
        
    def reset(self):
        self.prev_freq = -1
        self.time_index = -1
        self.cache = FreqTimeCache(self.interp_freq_array, self.interp_freq_chunk, self.master_beam)

    def sim_start(self):
        self.reset()
        self.in_sim = True    
        
    def sim_end(self):
        self.cache = None
        self.in_sim = False        

    def interp(self, *args, **kwargs):
        
        if "az_array" not in kwargs and "za_array" not in kwargs and "freq_array" in kwargs:
            
            # Case where requested to create a new beam with frequencies interpolated. Here we do nothing.
            assert len(kwargs["freq_array"]) == 1, "Expecting 1 freq"
            return self

        if not self.in_sim:
            # Not running vis_cpu, we just want to do an interp()
            return self.master_beam.interp(*args, **kwargs)

        # Now we are being called by vis_cpu
        freq = kwargs["freq_array"]
        assert len(freq) == 1, "sim_sparse_beam expects 1 frequency"  

        # Only keep track of freq and time
        if freq != self.prev_freq:
            # Scanning times of a new frequency
            self.time_index = 0
            self.prev_freq = freq
        else:
            # Scanning times of current frequency
            self.time_index += 1

        return self.cache.fetch(freq, self.time_index, **kwargs)

    def peak_normalize(self):
        # sparse beam is peak normalized when initialized
        pass
        
        
if __name__ == '__main__': 
    from resource import getrusage, RUSAGE_SELF
    import time

    Nfreqs = 208
    Ntimes = 10
    Nsrc = 3000
    freqs = np.linspace(60., 88., Nfreqs)*1e6

    az = np.random.random(size=(Ntimes, Nsrc))*2*np.pi
    za = np.random.random(size=(Ntimes, Nsrc))*np.pi    

    uvb = UVBeam()
    uvb.read_beamfits("NF_HERA_Vivaldi_power_beam.fits")
    uvb.interpolation_function = "az_za_simple"
    uvb.freq_interp_kind = "linear"
    kw =  {
                "reuse_spline": True,
                "check_azza_domain": False,
            }

    # Run what UVBeam does for some freq and some times
    start = time.time()
    for f in freqs:
        bm = uvb.interp(freq_array=np.array([f]), new_object=True, run_check=False)

        for t in range(Ntimes):
            bm.interp(az_array=az[0], za_array=za[0], freq_array=np.atleast_1d(f), **kw)
            
    print("uvbeam time", time.time()-start)

    # Run what sim_sparse_beam does for some freq and some times
   
    for chunk in [ 1, 2, 4, 8, 13, 16, 26, 52, 104, 208 ]:
        if chunk > Nfreqs: continue
        sb = sim_sparse_beam("NF_HERA_Vivaldi_power_beam.fits", 80, np.arange(-45, 46), interp_freq_array=freqs, interp_freq_chunk=chunk)
        sb.sim_start()
        start = time.time()
        for f in freqs:
            for t in range(Ntimes):
                sb.interp(az_array=az[0], za_array=za[0], freq_array=np.array([f]))
                
        print("sim_sparse_beam time using freq chunk size", chunk, time.time()-start)

    exit()

    # Compare values from sparse_beam with sim_sparse_beam

    # Uncached

    sb = sparse_beam("NF_HERA_Vivaldi_power_beam.fits", 80, np.arange(-45, 46))
    usage = getrusage(RUSAGE_SELF)
    print("MEM", usage.ru_maxrss/1000.0/1000, "GB");     # Usage in GB


    save_results_uncache = np.zeros((Nfreqs*Ntimes, 1, 1, 2, 1, Nsrc))
    start = time.time()
    index = 0
    for i, nf in enumerate(freqs):
        for j in range(Ntimes):
            b, _ = sb.interp(az_array=az[j], za_array=za[j], freq_array=np.array([nf*1e6]))
            save_results_uncache[index] = b
            index += 1
    usage = getrusage(RUSAGE_SELF)
    print("MEM", usage.ru_maxrss/1000.0/1000, "GB");     # Usage in GB

    print("Time", time.time()-start)

    
    sb = sim_sparse_beam("NF_HERA_Vivaldi_power_beam.fits", 80, np.arange(-45, 46), interp_freq_array=freqs*1e6, interp_freq_chunk=10)

    usage = getrusage(RUSAGE_SELF)
    print("MEM", usage.ru_maxrss/1000.0/1000, "GB");     # Usage in GB

    sb.sim_start()

    save_results_cache = np.zeros((Nfreqs*Ntimes, 1, 1, 2, 1, Nsrc))
    start = time.time()
    index = 0
    for i, nf in enumerate(freqs):
        fstart = time.time()
        for j in range(Ntimes):
            b, _ = sb.interp(az_array=az[j], za_array=za[j], freq_array=np.array([nf*1e6]))
            save_results_cache[index] = b

            #print(np.allclose(save_results_uncache[index], save_results_cache[index]))
         
            index += 1
        print("freq", i, "took", time.time()-fstart)
            
    usage = getrusage(RUSAGE_SELF)
    print("MEM", usage.ru_maxrss/1000.0/1000, "GB");     # Usage in GB
    print("Time", time.time()-start)
    exit()

    



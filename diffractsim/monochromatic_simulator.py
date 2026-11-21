from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import (
    angular_spectrum_method, two_steps_fresnel_method,
    zoom_fresnel_method, apply_transfer_function,
)

import numpy as np
from .util.backend_functions import backend as bd
from .util.bluestein_FFT import bluestein_fft2


"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


class MonochromaticField:
    """
    Represent a monochromatic complex field on a transverse rectangular grid.

    Lengths are in micrometres; use constants such as nm, um, mm and m.
    The squared field amplitude is irradiance in W/m². Integrated powers
    are in picowatts; divide by W to express them in watts. Optical operations
    return a new field, allowing callbacks to use F = F.add(...) followed
    by F = F.propagate(...) or F = F.zoom_propagate(...).
    """
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Create a uniform plane wave on a rectangular grid.

        Parameters
        ----------
        wavelength: wavelength of the plane wave, in micrometres.
        extent_x: width of the grid, in micrometres.
        extent_y: height of the grid, in micrometres.
        Nx: number of horizontal samples.
        Ny: number of vertical samples.
        intensity: irradiance in W/m², default 0.1.
        """
        global bd
        global backend_name
        from .util.backend_functions import backend as bd
        from .util.backend_functions import backend_name
        
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x/Nx
        self.dy = extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add(self, optical_element):
        """
        Apply an optical element and return the resulting field.

        Parameters
        ----------
        optical_element: element providing get_E(E, xx, yy, wavelength).

        Returns
        -------
        field: new MonochromaticField on the same grid and at the same distance.
        """
        F = self._spawn_like()
        F.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)
        F.z = self.z
        if hasattr(optical_element, "get_angular_irradiance"):
            F.periodic_gaussian_pupil = optical_element
        return F

    def propagate(self, z, scale_factor=1, model="ASM"):
        """Propagate to a grid with the same sample counts as the input.

        ``model="ASM"`` uses the angular-spectrum kernel, including evanescent
        decay for positive z. ``model="Fresnel"`` uses the two-step Fresnel
        method and its paraxial approximation. Both return a new field; the
        input field is unchanged.

        Parameters
        ----------
        z: propagation distance in micrometres; nonzero for Fresnel.
        scale_factor: multiplier for the output width and height, default 1.
            It remains the second argument for existing positional calls.
        model: "ASM" (default) or "Fresnel". Keep this choice fixed under JIT.

        Returns
        -------
        field: new MonochromaticField at self.z + z.

        Notes
        -----
        Use zoom_propagate to choose the monitor bounds and sample counts
        independently. Nonuniform pupil quadrature needs its angular sampling.

        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456 (angular spectrum and two-step Fresnel).
        """
        if model == "ASM":
            if hasattr(self, "pupil_quadrature"):
                raise ValueError('Nonuniform pupil quadrature requires zoom_propagate with model="ASM", sampling="angular".')
            F = self._spawn_like()
            F.E = angular_spectrum_method(F, self.E, z, self.λ, scale_factor=scale_factor)
            F.z = self.z+z
            return F
        if model == "Fresnel":
            x_new, y_new, E_new = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor)
            return self._from_arrays(x_new, y_new, E_new, self.z+z)
        raise ValueError('model must be "ASM" or "Fresnel".')




    def zoom_propagate(self, z, x_interval, y_interval, model="Fresnel",
                       sampling="fft", padding=1, Nx_mon=None, Ny_mon=None):
        """Propagate to a rectangular monitor with independently chosen sample counts.

        Parameters
        ----------
        z: propagation distance, in micrometres; positive for angular sampling.
        x_interval: first and last horizontal monitor coordinates, in micrometres.
        y_interval: first and last vertical monitor coordinates, in micrometres.
        model: "Fresnel" for the paraxial Fresnel approximation, or "ASM" for the full
            angular-spectrum kernel.
        sampling: "fft" uses FFT/Bluestein propagation; "angular" evaluates
            the pupil response in direction-cosine coordinates before mapping
            it to the Cartesian monitor.
        padding: positive integer refining ASM FFT frequency spacing; must be 1
            for Fresnel or angular sampling.
        Nx_mon: horizontal monitor count, at least 2; None keeps self.Nx.
        Ny_mon: vertical monitor count, at least 2; None keeps self.Ny.

        Returns
        -------
        field: new MonochromaticField at self.z + z, on endpoint-inclusive axes.

        Notes
        -----
        Fresnel/fft is the Fresnel-Bluestein method. Fresnel/angular uses a pupil
        chirp and Fourier transform with exact screen geometry and spherical
        phase; it remains a Fresnel approximation. ASM/angular directly
        integrates a finite pupil, useful for wide, coarsely sampled monitors.
        Both angular methods preserve the complex field phase.

        ASM/fft bandwidth is limited by the coarser input/output pitch; padding
        refines frequency spacing but cannot recover omitted bandwidth.
        Keep model, sampling, padding and array sizes fixed under JAX jit.

        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
        diffraction using the Bluestein method," Light: Science & Applications 9,
        119 (2020), doi:10.1038/s41377-020-00362-z.
        H. Weyl, Annalen der Physik 365, 481–500 (1919),
        doi:10.1002/andp.19193652104 (ASM angular quadrature; see its module bibliography).
        """
        if sampling not in ("fft", "angular"):
            raise ValueError('sampling must be "fft" or "angular".')
        if model == "Fresnel":
            if padding != 1:
                raise ValueError('padding requires model="ASM", sampling="fft".')
            if sampling == "angular":
                from .propagation_methods.angular_fresnel_method import angular_fresnel_method
                x_new, y_new, E_new = angular_fresnel_method(self, z, x_interval, y_interval, Nx_mon, Ny_mon)
            else:
                x_new, y_new, E_new = zoom_fresnel_method(self, self.E, z, self.λ, x_interval, y_interval, Nx_mon, Ny_mon)
        elif model == "ASM":
            from .propagation_methods.zoom_angular_spectrum_method import zoom_angular_spectrum_method
            x_new, y_new, E_new = zoom_angular_spectrum_method(
                self, z, x_interval, y_interval, sampling=sampling, padding=padding,
                Nx_mon=Nx_mon, Ny_mon=Ny_mon)
        else:
            raise ValueError('model must be "Fresnel" or "ASM".')
        return self._from_arrays(x_new, y_new, E_new, self.z+z)


    def angular_propagate(self, z, angular_samples=1024, angular_limit=2.0,
                          model="ASM", *, alpha=None, beta=None):
        """Evaluate the coherent intensity response in direction-cosine coordinates.

        Omit alpha and beta to return a uniform grid ``(a, K)``. Supply both
        to return ``K`` only at those paired directions. This avoids constructing
        a full angular image when an incoherent convolution needs a small region.
        The result is an intensity kernel, not a complex field for further optics.

        Parameters
        ----------
        z: positive pupil-to-monitor distance in micrometres.
        angular_samples: samples per axis for the uniform grid, at least 8.
        angular_limit: half-width of that grid, at least 2.
        model: "ASM" uses direct quadrature of the full angular-spectrum
            integral. "Fresnel" uses the paraxial pupil chirp with exact
            direction-cosine mapping. Keep the model fixed under JIT.
        alpha, beta: optional arrays of paired direction cosines with the same
            shape. Both must be supplied; the grid settings are then unused.

        Returns
        -------
        a, K: shared angular axis and square intensity array when no pairs are given.
        K: intensity array with the coordinate shape when pairs are supplied.

        Notes
        -----
        Inside alpha² + beta² < 1, K = z² |E(z*alpha/c, z*beta/c)|² / c⁴,
        where c = sqrt(1-alpha²-beta²). Outside the disk it continues as the
        pupil Fourier intensity divided by wavelength². This continuation is
        used for shifted-source convolution, not as a physical screen field.
        The existing dense and paired quadrature rules remain separate so their
        accuracy, memory use and cached fixed-geometry weights are preserved.

        References
        ----------
        H. Weyl, Annalen der Physik 365, 481–500 (1919),
        doi:10.1002/andp.19193652104 (angular-spectrum representation).
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456 (Fresnel propagation).
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript
        (response normalization and incoherent convolution).
        """
        if (alpha is None) != (beta is None):
            raise ValueError("Supply both alpha and beta, or omit both for a uniform grid.")
        if model == "Fresnel":
            from .propagation_methods.angular_fresnel_method import (
                angular_fresnel_response, angular_fresnel_response_at)
            if alpha is None:
                return angular_fresnel_response(self, z, angular_samples, angular_limit)
            return angular_fresnel_response_at(self, z, alpha, beta)
        if model == "ASM":
            from .propagation_methods.direct_quadrature_angular_spectrum_method import (
                direct_quadrature_angular_spectrum_method)
            return direct_quadrature_angular_spectrum_method(
                self, z, angular_samples, angular_limit, alpha=alpha, beta=beta)
        raise ValueError('model must be "ASM" or "Fresnel".')



    def get_modes_power(self, a1=None, a2=None):
        """
        Compute scalar diffraction-order powers from one sampled periodic cell.

        Parameters
        ----------
        a1: optional two-component lattice vector along the array columns,
            in micrometres; defaults to (extent_x, 0).
        a2: optional two-component lattice vector along the array rows,
            in micrometres; defaults to (0, extent_y).

        Returns
        -------
        m: centred horizontal integer order grid, with shape (Ny, Nx).
        n: centred vertical integer order grid, with shape (Ny, Nx).
        power: power per order and unit cell, in picowatts, with shape (Ny, Nx).
            Divide by W for watts. Evanescent orders are set to zero.

        Notes
        -----
        The field must uniformly sample one cell without repeating the endpoint;
        a1 and a2 may describe an oblique cell. Power is
        cell_area * |FFT(E)/(Nx*Ny)|², since |E|² already represents irradiance.
        All array calculations use the active backend and retain JAX gradients.
        """
        if hasattr(self, "pupil_quadrature"):
            raise ValueError("Mode powers require a uniformly sampled periodic cell.")
        a1 = bd.asarray((self.extent_x, 0) if a1 is None else a1)
        a2 = bd.asarray((0, self.extent_y) if a2 is None else a2)
        determinant = a1[0]*a2[1]-a1[1]*a2[0]
        m, n = bd.meshgrid(bd.arange(self.Nx)-self.Nx//2,
                          bd.arange(self.Ny)-self.Ny//2)
        alpha = self.λ*(m*a2[1]-n*a1[1])/determinant
        beta = self.λ*(n*a1[0]-m*a2[0])/determinant
        coefficients = bd.fft.fftshift(bd.fft.fft2(self.E))/self.E.size
        power = bd.abs(determinant)*bd.abs(coefficients)**2
        return m, n, bd.where(alpha**2+beta**2 <= 1, power, 0)


    def get_farfield(self, alpha_interval = None, beta_interval = None):

        """
        Compute the angular far field S = (dP/dΩ)/cos(theta).

        Returns the direction-cosine axes alpha, beta and S in picowatts
        per steradian. Divide S by W for W/sr. The far-field plotting
        methods make this conversion automatically.

        Parameters
        ----------
        alpha_interval : list with the region [α0, αf] over which to evaluate the radiant intensity per cosine.
        beta_interval : list with the region [β0, βf] over which to evaluate the radiant intensity per cosine.


        Reference: 
        James E. Harvey, Cynthia L. Vernold, Andrey Krywonos, and Patrick L. Thompson, 
        "Diffracted radiance: a fundamental quantity in nonparaxial scalar diffraction theory," Appl. Opt. 38, 6469-6481 (1999)

        """
        if hasattr(self, "pupil_quadrature"):
            if alpha_interval is None:
                fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
                fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
            else:
                fx = bd.linspace(alpha_interval[0]/self.λ, alpha_interval[1]/self.λ, self.Nx)
                fy = bd.linspace(beta_interval[0]/self.λ, beta_interval[1]/self.λ, self.Ny)
            spectrum = self.pupil_quadrature.fourier_transform(self.E, fx, fy)
            return fx*self.λ, fy*self.λ, bd.abs(spectrum/self.λ)**2

        # Compute the sampling intervals in x and y directions
        # If no specific output region is provided, use the natural FFT frequency grid
        if alpha_interval == None: 
            fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
            fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
            U_f = bd.fft.fftshift(bd.fft.fft2(self.E))
        else: # Use Bluestein's method to compute the Fourier transform
            fx0, fx1 = bd.array(alpha_interval) / self.λ 
            fy0, fy1 = bd.array(beta_interval) / self.λ 
            fx = bd.linspace(fx0, fx1, self.Nx) 
            fy = bd.linspace(fy0, fy1, self.Ny)
            U_f = bluestein_fft2(self.E, fx0, fx1, 1/self.dx, fy0, fy1, 1/self.dy)
        
        fxx, fyy = bd.meshgrid(fx, fy) #create 2d arrayswith the frequencies
        # Calculate a FT phase translation correction and scaling factor  
        ft_factor = self.dx * self.dy * bd.exp(-1j * 2 * bd.pi * self.x[0] * fxx - 1j * 2 * bd.pi * self.y[0] * fyy)
        U_f = U_f * ft_factor
        α = fx * self.λ 
        β = fy * self.λ

        # Return the angular coordinates and the computed radiant_intensity distribution (Eq.2.1.32)
        radiant_intensity_percos = bd.real(U_f * bd.conj(U_f)) / (self.λ**2)
        return α, β, radiant_intensity_percos


    def propagate_to_image_plane(self, pupil, M, zi, z0, scale_factor = 1):
        """
        Diffraction-limited imaging via the exit pupil, producing the field at the image plane.
        Returns a NEW MonochromaticField.
        """
        # copy of current arrays (not modifying self)
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)
        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.x[1]-self.x[0]))
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.y[1]-self.y[0]))
        fxx, fyy = bd.meshgrid(fx, fy)

        H = pupil.get_amplitude_transfer_function(fxx/bd.abs(M), fyy/bd.abs(M), zi, self.λ)

        # start from self, then apply transfer function (re-using provided helper)
        E_tmp = apply_transfer_function(self, self.E / bd.abs(M) * (bd.flip(self.E*0+1) if M < 0 else 1), self.λ, H, scale_factor)

        # scale coordinates by |M|
        M_abs = bd.abs(M)
        x_new = M_abs * self.x
        y_new = M_abs * self.y
        E_new = E_tmp

        return self._from_arrays(x_new, y_new, E_new, self.z + zi + z0)


    def propagate_to_lens_focal_plane(self, focal_length, x_interval, y_interval):
        """
        Fresnel lens and propagation to its focal plane.
        Returns a NEW MonochromaticField.
        """
        from .util.backend_functions import backend as bd
        from .util.bluestein_FFT import bluestein_fft2, bluestein_fftfreq

        C = bluestein_fft2(self.E,
                           x_interval[0] / (focal_length*self.λ), x_interval[1] / (focal_length*self.λ), 1/self.dx,
                           y_interval[0] / (focal_length*self.λ), y_interval[1] / (focal_length*self.λ), 1/self.dy)

        dfx = 1/(self.Nx*self.dx); dfy = 1/(self.Ny*self.dy)
        fx_zfft = bluestein_fftfreq(x_interval[0]/(focal_length*self.λ), x_interval[1]/(focal_length*self.λ), self.Nx)
        fy_zfft = bluestein_fftfreq(y_interval[0]/(focal_length*self.λ), y_interval[1]/(focal_length*self.λ), self.Ny)
        dfx_zfft = fx_zfft[1]-fx_zfft[0]; dfy_zfft = fy_zfft[1]-fy_zfft[0]
        nn, mm = bd.meshgrid((bd.linspace(0,(self.Nx-1),self.Nx)*dfx_zfft/dfx ),
                              (bd.linspace(0,(self.Ny-1),self.Ny)*dfy_zfft/dfy ))
        ft_factor = (self.dx*self.dy* bd.exp(bd.pi*1j * (nn + mm)))

        x_new = fx_zfft*(focal_length*self.λ)
        y_new = fy_zfft*(focal_length*self.λ)
        xx_new, yy_new = bd.meshgrid(x_new, y_new)
        E_new = C*ft_factor * bd.exp(1j*bd.pi/(self.λ*focal_length)  * (xx_new**2 + yy_new**2)  +   1j*2*bd.pi/self.λ * focal_length ) / (1j*focal_length*self.λ)

        return self._from_arrays(x_new, y_new, E_new, self.z + focal_length)


    def get_colors(self):
        """compute RGB colors of the cross-section profile at the current distance"""

        # compute Field Intensity
        I = bd.real(self.E * bd.conjugate(self.E))  

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def get_field(self):
        """get field of the cross-section profile at the current distance"""

        return self.E


    def get_intensity(self):
        """compute field intensity of the cross-section profile at the current distance"""

        return bd.real(self.E * bd.conjugate(self.E))  



    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb

    def interpolate(self, Nx, Ny):
        """Interpolate the field to the new shape (Nx,Ny)"""
        from scipy.interpolate import RectBivariateSpline


        if backend_name == 'cupy':
            self.E = self.E.get()

        fun_real = RectBivariateSpline(
                    self.dx*(np.arange(self.Nx)-self.Nx//2),
                    self.dy*(np.arange(self.Ny)-self.Ny//2),
                    np.real(self.E))

        fun_imag = RectBivariateSpline(
                    self.dx*(np.arange(self.Nx)-self.Nx//2),
                    self.dy*(np.arange(self.Ny)-self.Ny//2),
                    np.imag(self.E))


        self.Nx = Nx
        self.Ny = Ny

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.E = bd.array(fun_real(self.dx*(np.arange(Nx)-Nx//2), self.dy*(np.arange(Ny)-Ny//2))  +  fun_imag(self.dx*(np.arange(Nx)-Nx//2), self.dy*(np.arange(Ny)-Ny//2))*1j)


        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)




    def get_longitudinal_profile(self, start_distance, end_distance, steps, scale_factor = 1):
        """
        Propagates the field at `steps` positions between start_distance and end_distance
        and returns (RGB image stack, field slice stack, extent) **without mutating self**.
        """

        z = bd.linspace(start_distance, end_distance, steps)

        longitudinal_profile_rgb = bd.zeros((steps, self.Nx, 3))
        longitudinal_profile_E   = bd.zeros((steps, self.Nx), dtype=complex)
        t0 = time.time()

        bar = progressbar.ProgressBar()
        for i in bar(range(steps)):
            if scale_factor == 1:
                F = self.propagate(z[i])                       # returns a NEW MonochromaticField
            else:
                F = self.propagate(z[i], scale_factor, model="Fresnel")   # returns a NEW MonochromaticField

            rgb = F.get_colors()

            # take the center row along y for the x–z profile
            if backend_name == 'jax':
                longitudinal_profile_rgb = longitudinal_profile_rgb.at[i, :, :].set(rgb[F.Ny // 2, :, :])
                longitudinal_profile_E   = longitudinal_profile_E.at[i, :].set(F.E[F.Ny // 2, :])
            else:
                longitudinal_profile_rgb[i, :, :] = rgb[F.Ny // 2, :, :]
                longitudinal_profile_E[i, :]      = F.E[F.Ny // 2, :]

        print("Took", time.time() - t0)

        # Keep the same convention as before for the plotted x-extent:
        extent = [self.x[0] * scale_factor, self.x[-1] * scale_factor, start_distance, end_distance]
        return longitudinal_profile_rgb, longitudinal_profile_E, extent


    def _spawn_like(self):
        """Create a new MonochromaticField with same sampling/extent/wavelength.
        The returned instance's E/z will be overwritten by the caller."""
        F = MonochromaticField(self.λ, self.extent_x, self.extent_y, self.Nx, self.Ny)
        # copy colour system so rendering stays consistent
        F.cs = self.cs
        # Preserve shifted monitor coordinates when adding or propagating optics.
        F.x, F.y, F.xx, F.yy = self.x, self.y, self.xx, self.yy
        F.dx, F.dy = self.dx, self.dy
        if hasattr(self, "pupil_quadrature"):
            F.pupil_quadrature = self.pupil_quadrature
        return F

    def _from_arrays(self, x, y, E, z):
        """Create a new MonochromaticField built from explicit x,y,E and absolute z."""
        F = MonochromaticField(self.λ, x[-1]-x[0]+(x[1]-x[0]), y[-1]-y[0]+(y[1]-y[0]), len(x), len(y))
        F.x = x
        F.y = y
        F.xx, F.yy = bd.meshgrid(x, y)
        F.dx = x[1]-x[0]
        F.dy = y[1]-y[0]
        F.extent_x = F.Nx * F.dx
        F.extent_y = F.Ny * F.dy
        F.E = E
        F.z = z
        F.cs = self.cs
        return F





    def __add__(self, Field):
        """
        Interfere MonochromaticField with another MonochromaticField instance. 
        The wavelength, dimensions and sampling of the interfering fields must be identical
        """

        if ((self.extent_x == Field.extent_x) and (self.extent_y == Field.extent_y) and (self.Nx == Field.Nx) and (self.Ny == Field.Ny) and (self.λ == Field.λ )):
            mixed_field = MonochromaticField(self.λ, self.extent_x, self.extent_y, self.Nx, self.Ny)
            mixed_field.E = self.E + Field.E
            return mixed_field

        else:
            raise ValueError(
            "The wavelength, dimensions and sampling of the interfering fields must be identical")


    from .visualization import plot_colors, plot_phase, plot_intensity, plot_longitudinal_profile_colors, plot_longitudinal_profile_intensity, plot_farfield, plot_farfield_spherical_coordinates, plot_grating_modes



#MonochromaticField = Field

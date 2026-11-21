"""Record a callback's final propagation and sample its coherent response.

Earlier steps are evaluated before further optics. The final step waits
until the incoherent convolution supplies its angular coordinates. Both
the recording and the response sampling live in DeferredPropagation.

References
----------
D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
SPIE (2011), doi:10.1117/3.858456.
R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
propagation of partially coherent light (2026), accompanying manuscript.
"""
import numpy as np
from ..util import backend_functions


class DeferredPropagation:
    """Record the final coherent propagation and evaluate it where needed.
    
    IncoherentLightSimulation supplies this wrapper to nonparaxial callbacks.
    The callback can add optics and propagate as usual. Earlier propagation
    steps are evaluated before adding more optics or recording another step;
    only the final step waits for the response sampling coordinates.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """

    def __init__(self, pupil, method=None, distance=None, **options):
        """Wrap a coherent pupil with an optional pending propagation step.
        
        Parameters
        ----------
        pupil: MonochromaticField immediately before the pending propagation.
        method: "propagate" or "zoom_propagate"; None means no step is pending.
        distance: propagation distance in micrometres, or None when method is None.
        **options: keyword arguments for the selected MonochromaticField method.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        self.pupil, self.method = pupil, method
        self.distance, self.options = distance, options
        self.model = options.get("model", "ASM" if method == "propagate" else "Fresnel")
        self.sampling = options.get("sampling", "fft")
        self._field = None
        self._spectrum_limit = 2.5

    def field(self):
        """Evaluate and cache the pending step on its rectangular monitor.
        
        Returns
        -------
        field: propagated MonochromaticField, or the input pupil when no step
            is pending. Repeated calls reuse the evaluated field.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        if self.method is None:
            return self.pupil
        if self._field is None:
            self._field = getattr(self.pupil, self.method)(self.distance, **self.options)
        return self._field

    def __getattr__(self, name):
        # Coordinate/field inspection and other optical methods still work.
        return getattr(self.field(), name)

    @property
    def E(self):
        """Complex field on the rectangular monitor; reading evaluates the pending step.
        
        Assigning E first evaluates that step, then replaces the resulting field
        values and clears the pending propagation.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        return self.field().E

    @E.setter
    def E(self, value):
        self.pupil = self.field()
        self.pupil.E = value
        self.method, self._field = None, None

    def add(self, optical_element):
        """Evaluate any pending propagation, then apply an optical element.
        
        Parameters
        ----------
        optical_element: element accepted by MonochromaticField.add.
        
        Returns
        -------
        response: DeferredPropagation wrapping the updated field, with no pending step.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        return DeferredPropagation(self.field().add(optical_element))

    def propagate(self, z, scale_factor=1, model="ASM"):
        """Record the final propagation after evaluating any earlier step.

        ASM uses the full angular spectrum. Fresnel uses the two-step method
        and its paraxial approximation. Arguments match MonochromaticField.propagate.
        Reference: D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        """
        if model not in ("ASM", "Fresnel"):
            raise ValueError('model must be "ASM" or "Fresnel".')
        return DeferredPropagation(self.field(), "propagate", z,
                                scale_factor=scale_factor, model=model)

    def zoom_propagate(self, z, x_interval, y_interval, model="Fresnel",
                       sampling="fft", padding=1, Nx_mon=None, Ny_mon=None):
        """Record propagation to a chosen rectangular monitor.
        
        Parameters
        ----------
        z: propagation distance, in micrometres; positive for angular sampling.
        x_interval: first and last horizontal monitor coordinates, in micrometres.
        y_interval: first and last vertical monitor coordinates, in micrometres.
        model: "Fresnel" for the Fresnel approximation or "ASM" for angular spectrum.
        sampling: "fft" uses the propagated screen; "angular" samples the final
            coherent response directly at direction cosines when requested.
        padding: positive integer refining ASM FFT frequency sampling; must be 1
            for Fresnel or angular sampling.
        Nx_mon: horizontal monitor count, at least 2; None keeps the pupil count.
        Ny_mon: vertical monitor count, at least 2; None keeps the pupil count.
        
        Returns
        -------
        response: DeferredPropagation with this final step pending.
        
        Notes
        -----
        Intervals include both endpoints. Arguments have the same meaning as in
        MonochromaticField.zoom_propagate; any earlier pending step is evaluated.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        if model not in ("Fresnel", "ASM"):
            raise ValueError('model must be "Fresnel" or "ASM".')
        if sampling not in ("fft", "angular"):
            raise ValueError('sampling must be "fft" or "angular".')
        if not isinstance(padding, (int, np.integer)) or padding < 1:
            raise ValueError("padding must be a positive integer.")
        if model == "Fresnel" and padding != 1:
            raise ValueError('padding requires model="ASM", sampling="fft".')
        if sampling == "angular" and padding != 1:
            raise ValueError('padding applies only to sampling="fft".')
        for n in (Nx_mon, Ny_mon):
            if n is not None and (isinstance(n, (bool, np.bool_)) or
                    not isinstance(n, (int, np.integer)) or n < 2):
                raise ValueError("Monitor counts must be integers >= 2.")
        return DeferredPropagation(self.field(), "zoom_propagate", z,
            x_interval=x_interval, y_interval=y_interval, model=model,
            sampling=sampling, padding=padding, Nx_mon=Nx_mon, Ny_mon=Ny_mon)


    def screen_axes(self, *, host=False):
        """Return the monitor axes without running the propagation.
        
        Parameters
        ----------
        host: if True, return NumPy arrays for calculations with fixed geometry.
            If False, use the selected backend so the coordinates can remain
            part of a differentiable calculation.
        
        Returns
        -------
        x: one-dimensional horizontal monitor axis, in micrometres.
        y: one-dimensional vertical monitor axis, in micrometres.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        bd = backend_functions.backend
        to_host = bd.asnumpy if backend_functions.backend_name == "cupy" else np.asarray
        if self.method == "zoom_propagate":
            nx, ny = self.options["Nx_mon"], self.options["Ny_mon"]
            x_interval, y_interval = self.options["x_interval"], self.options["y_interval"]
            if host:
                x_interval, y_interval = to_host(x_interval), to_host(y_interval)
            xp = np if host else bd
            return (xp.linspace(*x_interval, self.pupil.Nx if nx is None else nx),
                    xp.linspace(*y_interval, self.pupil.Ny if ny is None else ny))
        scale = self.options.get("scale_factor", 1)
        x, y = self.pupil.x*scale, self.pupil.y*scale
        return (to_host(x), to_host(y)) if host else (x, y)

    def angular_propagate(self, z, angular_samples=1024, angular_limit=2.0,
                          *, alpha=None, beta=None):
        """Sample the recorded coherent response on a grid or at paired directions.

        With no alpha or beta, return (a, K) on the shared uniform angular axis.
        With both supplied, return K at those pairs. Grid settings are then unused.
        The recorded propagation selects the model and sampling method.

        Angular sampling evaluates the pupil response directly. FFT sampling
        evaluates the recorded monitor once and interpolates its intensity.
        The normalization is K = z² |E(z*alpha/c, z*beta/c)|² / c⁴ inside
        the unit disk, c = sqrt(1-alpha²-beta²). Outside it, use the pupil's
        Fourier intensity divided by wavelength² for source convolution.

        Parameters
        ----------
        z: positive final propagation distance in micrometres.
        angular_samples: samples per angular axis for grid output.
        angular_limit: angular half-width for grid output, normally at least 2.
        alpha, beta: optional matching arrays of paired direction cosines.

        References
        ----------
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        """
        if (alpha is None) != (beta is None):
            raise ValueError("Supply both alpha and beta, or omit both for a uniform grid.")
        if self.sampling == "angular":
            return self.pupil.angular_propagate(z, angular_samples, angular_limit,
                model=self.model, alpha=alpha, beta=beta)
        if alpha is not None:
            return self._screen_response_at(z, alpha, beta)
        bd = backend_functions.backend
        self._spectrum_limit = max(2.5, angular_limit+0.5)
        a = -angular_limit+bd.arange(angular_samples)*(2*angular_limit/angular_samples)
        alpha, beta = bd.meshgrid(a, a)
        return a, self._screen_response_at(z, alpha, beta)



    def _screen_response_at(self, z, alpha, beta):
        """Sample the recorded FFT monitor intensity at paired directions.

        Linear interpolation maps the resolved Cartesian monitor to direction
        cosines, then applies K = z² |E|² / c⁴. Fixed NumPy coordinates entirely
        inside the unit disk skip the unused Fourier-continuation calculation.
        Angular quadrature bypasses this helper through angular_propagate.

        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        if hasattr(self.pupil, "pupil_quadrature"):
            raise ValueError('Physical-pixel quadrature requires sampling="angular".')
        bd = backend_functions.backend
        if backend_functions.backend_name == "jax":
            from jax.scipy.ndimage import map_coordinates
        elif backend_functions.backend_name == "cupy":
            from cupyx.scipy.ndimage import map_coordinates
        else:
            from scipy.ndimage import map_coordinates

        F = self.field()
        inside = alpha*alpha+beta*beta < 1
        c = bd.sqrt(bd.where(inside, 1-alpha*alpha-beta*beta, 1))
        H = map_coordinates(bd.abs(F.E)**2,
            bd.array([(z*beta/c-F.y[0])/F.dy, (z*alpha/c-F.x[0])/F.dx]),
            order=1, mode="constant", cval=0)
        response = z*z/c**4*H
        # Compact convolution supplies fixed host directions. If every one is
        # propagating, no Fourier continuation is needed. where() would still
        # evaluate that expensive branch before selecting its values.
        if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
            if np.all(alpha*alpha+beta*beta < 1):
                return response
        return bd.where(inside, response, self._fourier_intensity(alpha, beta))

    def _fourier_intensity(self, alpha, beta):
        """Pupil Fourier continuation outside the direction-cosine disk.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
        propagation of partially coherent light (2026), accompanying manuscript.
        """
        from ..util.bluestein_FFT import bluestein_fft
        from .direct_quadrature_angular_spectrum_method import sample_analytic_spectrum
        bd, F = backend_functions.backend, self.pupil
        # A fixed grid with an interpolation margin keeps JAX shapes static.
        limit = self._spectrum_limit
        n = max(64, int(np.ceil(24*limit)),
                int(np.ceil(8*limit*max(F.extent_x, F.extent_y)/F.λ)))
        step = 2*limit/n
        a = -limit+bd.arange(n)*step
        A = bluestein_fft(F.E, 1, a[0]/F.λ, a[-1]/F.λ, 1/F.dx, n)
        A = bluestein_fft(A, 0, a[0]/F.λ, a[-1]/F.λ, 1/F.dy, n)
        A *= F.dx*F.dy/F.λ
        A *= bd.exp(-2j*bd.pi*F.x[0]*a/F.λ)[None, :]
        A *= bd.exp(-2j*bd.pi*F.y[0]*a/F.λ)[:, None]
        values = sample_analytic_spectrum(A, bd.asarray(alpha).ravel(),
            bd.asarray(beta).ravel(), a[0], step)
        return bd.abs(values.reshape(alpha.shape))**2


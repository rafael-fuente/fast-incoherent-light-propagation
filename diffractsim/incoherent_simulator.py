"""Incoherent propagation shared by monochromatic and polychromatic examples.

Radiance functions supply the source. A coherent response and an intensity
convolution then give the output, including compact collection regions.
"""
from .monochromatic_simulator import MonochromaticField
from .propagation_methods.coherent_response import DeferredPropagation
from .util.convolution import crop_convolution
from .util.constants import W, m, mm, cm, um, nm
import matplotlib.pyplot as plt
import numpy as np

class IncoherentLightSimulation:
    """
    Simulate diffraction from a spatially incoherent source at one wavelength.

    Configure the source radiance, then supply a callback that adds the optics
    and performs the propagation. The source contributions are combined by
    intensity convolution, using the selected NumPy, CuPy or JAX backend.
    """

    def __init__(self, λ, extent_x, extent_y, Nx, Ny):
        """
        Initialize the rectangular grid on which the pupil is sampled.

        Parameters
        ----------
        λ: wavelength of the light, in micrometres.
        extent_x: width of the pupil grid, in micrometres.
        extent_y: height of the pupil grid, in micrometres.
        Nx: number of horizontal pupil samples.
        Ny: number of vertical pupil samples.

        Notes
        -----
        The callback defines the monitor grid independently of the pupil grid.
        For nonparaxial calculations, x_screen and y_screen can override its
        monitor coordinates; see get_irradiance.
        """
        # Resolve the selected backend when constructed, after set_backend(...).
        global bd, backend_name, map_coordinates, jax
        from .util.backend_functions import backend as bd, backend_name
        if backend_name == "jax":
            import jax
            from jax.scipy.ndimage import map_coordinates
        elif backend_name == "cupy":
            from cupyx.scipy.ndimage import map_coordinates
        else:
            from scipy.ndimage import map_coordinates

        # Coordinates where the pupil transmission is sampled.
        self.λ = λ
        self.Nx, self.Ny = Nx, Ny
        self.extent_x, self.extent_y = extent_x, extent_y

        self.dx = extent_x / Nx
        self.dy = extent_y / Ny
        self.x = self.dx * (bd.arange(Nx) - Nx // 2)
        self.y = self.dy * (bd.arange(Ny) - Ny // 2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        # The source is configured separately from propagation.
        self._source_plane_illumination = None  # Original plane source for plotting.
        self.plane_illumination = None
        self.angular_illumination = None
        self.z1 = bd.inf  # ∞ means angular specification
        self.paraxial = True

        # Optional detector coordinates; None uses the callback's monitor.
        self.x_screen = self.y_screen = None

        # A propagation call stores the source coherence for plotting.
        self._last_gamma = None  # mutual coherence magnitude for plotting

    def set_pupil_quadrature(self, radius, order=2):
        """
        Resolve phase pixels and the boundary of a circular pupil by quadrature.

        Parameters
        ----------
        radius: radius of the circular pupil, in micrometres; it must fit in the grid.
        order: number of Gauss integration points per axis within each pixel.

        Notes
        -----
        Call before sampling the aperture: this replaces x, y, xx, yy, Nx and Ny
        with the quadrature grid. The original pixels retain their physical size.
        Finite-distance propagation uses sampling="angular" with ASM or Fresnel.
        The quadrature geometry is fixed; pupil values remain differentiable.
        """
        from diffractsim.util.pupil_quadrature import PupilQuadrature
        previous = getattr(self, "pupil_quadrature", None)
        nx, ny = (self.Nx, self.Ny) if previous is None else (previous.nx, previous.ny)
        self.pupil_quadrature = PupilQuadrature(self.extent_x, self.extent_y,
                                               nx, ny, radius, order)
        self.x, self.y = self.pupil_quadrature.x, self.pupil_quadrature.y
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

    # --------------------- illumination ---------------------

    def set_illumination_from_plane(self, illum_func, distance, paraxial=True,
                                    angular_samples=1024, angular_limit=2.0,
                                    source_subpixels=1, source_bounds=None):
        """
        Set source-plane radiance and map it to incident directions.

        Parameters
        ----------
        illum_func: function illum_func(x, y) returning radiance in W m^-2 sr^-1;
            x and y are source-plane coordinates in micrometres, supplied as arrays.
        distance: positive source-to-pupil distance, in micrometres.
        paraxial: use small-angle geometry if True; direction cosines if False.
        angular_samples: number of samples per axis of the finite-distance
            angular convolution grid; integer >= 8.
        angular_limit: half-width of that grid in direction-cosine coordinates.
        source_subpixels: samples per axis within each angular source cell;
            1 samples directly, larger values average the source within the cell.
        source_bounds: optional (alpha_min, alpha_max, beta_min, beta_max)
            enclosing the source in angular coordinates, not lengths.

        Notes
        -----
        The distance also supplies the common incident quadratic phase.
        The source must be zero outside source_bounds when bounds are supplied.
        See set_angular_illumination for grid extent and compact-monitor rules.
        """
        traced_distance = backend_name == "jax" and isinstance(distance, jax.core.Tracer)
        if not traced_distance and (not np.isfinite(distance) or distance <= 0):
            raise ValueError("Source distance must be finite and positive.")
        def angular_from_plane(alpha, beta):
            """Evaluate the plane-source radiance along each incident direction."""
            c = 1 if paraxial else bd.sqrt(bd.maximum(1-alpha**2-beta**2, 1e-15))
            radiance = illum_func(distance*alpha/c, distance*beta/c)
            return radiance if paraxial else bd.where(alpha**2+beta**2 < 1, radiance, 0)

        self.set_angular_illumination(angular_from_plane, paraxial=paraxial,
                                      angular_samples=angular_samples,
                                      angular_limit=angular_limit,
                                      source_subpixels=source_subpixels, source_bounds=source_bounds)
        self.z1 = distance
        self.plane_illumination = self._source_plane_illumination = illum_func

    def set_angular_illumination(self, illum_func, paraxial=False,
                                 angular_samples=1024, angular_limit=2.0,
                                 source_subpixels=1, source_bounds=None):
        """
        Set source radiance as a function of incident direction cosines.

        Parameters
        ----------
        illum_func: function illum_func(alpha, beta) returning radiance in
            W m^-2 sr^-1; it must accept broadcastable coordinate arrays.
        paraxial: use the paraxial convolution if True; the nonparaxial model
            if False. This does not select the callback's coherent propagator.
        angular_samples: samples per axis of the nonparaxial finite-distance
            convolution grid; integer >= 8, default 1024.
        angular_limit: half-width of that grid; default 2 covers all sums of
            physical source and output direction cosines for general pupils.
        source_subpixels: positive number of samples per axis within a source
            cell; use values > 1 to resolve source edges by averaging.
        source_bounds: optional (alpha_min, alpha_max, beta_min, beta_max)
            outside which the radiance is zero; enables compact convolutions
            for fixed angular-Fresnel or FFT rectangular monitors, paired regions,
            and large-pupil angular ASM zooms that exceed the global-grid threshold.

        Notes
        -----
        This setter removes any source-plane curvature. General full-grid
        angular responses require angular_limit >= 2. PeriodicGaussianPupil
        instead requires a grid covering its selected diffraction orders.
        Compact convolutions require fixed source bounds and monitor geometry;
        source and pupil values remain differentiable. get_farfield uses the
        grid supplied by its callback, rather than these sampling settings.
        """
        if not isinstance(angular_samples, (int, np.integer)) or angular_samples < 8:
            raise ValueError("angular_samples must be an integer >= 8.")
        if not np.isfinite(angular_limit) or angular_limit <= 0:
            raise ValueError("angular_limit must be finite and positive.")
        if not isinstance(source_subpixels, (int, np.integer)) or source_subpixels < 1:
            raise ValueError("source_subpixels must be a positive integer.")
        self.plane_illumination = self._source_plane_illumination = None
        self.angular_illumination = illum_func
        self.z1, self.paraxial = np.inf, paraxial
        self.angular_samples = angular_samples
        self.angular_limit = angular_limit
        self.source_subpixels = source_subpixels
        if source_bounds is not None:
            if len(source_bounds) != 4 or not np.all(np.isfinite(source_bounds)):
                raise ValueError("source_bounds must contain four finite angular limits.")
            if source_bounds[0] >= source_bounds[1] or source_bounds[2] >= source_bounds[3]:
                raise ValueError("source_bounds must have increasing limits.")
            source_bounds = tuple(source_bounds)
        self.source_bounds = source_bounds

    def get_incident_irradiance(self):
        """
        Integrate the configured angular radiance over its compact support.

        Returns
        -------
        irradiance: scalar incident pupil irradiance, in W/m².

        Notes
        -----
        Configure illumination with source_bounds first. The integral uses
        source_subpixels and the angular grid step; multiplying by pupil area
        gives the incident power used to normalize collection efficiency.
        """
        if self.angular_illumination is None:
            raise ValueError("Configure illumination first.")
        if self.source_bounds is None:
            raise ValueError("Set source_bounds to integrate the source support.")
        from .util.compact_source import source_kernel
        _, _, mass = source_kernel(self.angular_illumination, self.source_bounds,
            2*self.angular_limit/self.angular_samples, self.source_subpixels)
        return bd.sum(mass)

    # --------------------- core computations ---------------------

    def get_irradiance_rs(self, finite_distance_model, z, *, source_order=3,
                          monitor_order=7, angular_samples=None, source_bounds=None,
                          Nx_mon=None, Ny_mon=None):
        """
        Check a single-pupil simulation with the paper's full RS integral.

        Parameters
        ----------
        finite_distance_model: the same callback used by get_irradiance. It
            applies the pupil and ends with a free-space propagation by z.
            Its final model and sampling select no approximation here: the
            reference always evaluates the full Rayleigh-Sommerfeld kernel.
        z: positive pupil-to-monitor distance, in micrometres.
        source_order: interpolation nodes per source-slope axis, default 3.
            Used only for a source at finite distance.
        monitor_order: interpolation nodes per monitor-slope axis, default 7.
        angular_samples: optional angular resolution for this reference;
            None uses the configured source grid spacing. It does not change
            the simulator or the number of returned monitor samples.
        source_bounds: optional (alpha_min, alpha_max, beta_min, beta_max).
            None uses the configured bounds, or finds the nonzero source cells.
            For a finite source, this rectangle must fit inside the unit disk.
        Nx_mon, Ny_mon: optional monitor counts for a zoom callback. None
            retains its counts. Smaller counts can make an image check cheaper.

        Returns
        -------
        x, y, E: monitor coordinates in micrometres and irradiance in W/m²,
            with the same axis and array conventions as get_irradiance.

        Notes
        -----
        For a plane source, Eq. (25) retains each emitter's spherical incident
        field and the outgoing RS kernel. An angular source supplies plane
        waves instead. Intensities add with the configured radiance; the
        paraxial flag and Eq. (24)'s common curvature are not used.

        As in the paper's whole-image reference, interpolation of the smooth
        kernel factors avoids propagating thousands of emitters separately.
        Both kernels are evaluated in full at the interpolation nodes.
        Increase source_order, monitor_order and angular_samples separately
        to check numerical convergence. source_subpixels still resolves edges
        of the source; the existing pupil grid supplies the pupil quadrature.

        This reference supports one sampled pupil followed by free space.
        Geometry and orders must be fixed. Phase gradients are supported;
        pass fixed source_bounds to differentiate source parameters under JIT.
        No pupil values or source values are cached between calls.
        """
        from .propagation_methods.rayleigh_sommerfeld import rayleigh_sommerfeld_irradiance
        if self.angular_illumination is None:
            raise ValueError("Configure illumination before propagation.")
        if not np.isscalar(z) or not np.isfinite(z) or z <= 0:
            raise ValueError("The RS reference requires a fixed, positive distance.")
        for name, order in (("source_order", source_order), ("monitor_order", monitor_order)):
            if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or order < 2:
                raise ValueError(f"{name} must be an integer >= 2.")
        samples = self.angular_samples if angular_samples is None else angular_samples
        if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 8:
            raise ValueError("angular_samples must be an integer >= 8.")

        F = MonochromaticField(self.λ, self.extent_x, self.extent_y, self.Nx, self.Ny, intensity=1)
        if hasattr(self, "pupil_quadrature"):
            F.pupil_quadrature = self.pupil_quadrature
            F.x, F.y, F.xx, F.yy = self.x, self.y, self.xx, self.yy
        response = finite_distance_model(DeferredPropagation(F), z)
        if not isinstance(response, DeferredPropagation) or response.method is None:
            raise ValueError("The callback must end with a propagation by z.")
        if not np.isclose(response.distance, z, rtol=1e-12, atol=0):
            raise ValueError("The callback's final propagation distance must equal z.")
        pupil = response.pupil
        if (pupil.z != 0 or pupil.E.shape != F.E.shape or
                pupil.extent_x != F.extent_x or pupil.extent_y != F.extent_y):
            raise ValueError("The RS reference requires one pupil before the final free-space step.")
        if hasattr(pupil, "periodic_gaussian_pupil"):
            raise ValueError("Sample the physical Gaussian pupil before using the RS reference.")
        if (self.x_screen is None) != (self.y_screen is None):
            raise ValueError("Set both x_screen and y_screen, or leave both as None.")
        if Nx_mon is not None or Ny_mon is not None:
            if response.method != "zoom_propagate" or self.x_screen is not None:
                raise ValueError("Monitor count overrides require a zoom callback without x_screen/y_screen.")
            options = dict(response.options)
            for name, count in (("Nx_mon", Nx_mon), ("Ny_mon", Ny_mon)):
                if count is not None:
                    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
                        raise ValueError(f"{name} must be an integer >= 2.")
                    options[name] = count
            response = DeferredPropagation(pupil, response.method, response.distance, **options)
        to_host = bd.asnumpy if backend_name == "cupy" else np.asarray
        x, y = (response.screen_axes(host=True) if self.x_screen is None
                else (to_host(self.x_screen), to_host(self.y_screen)))
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.size(x) or not np.size(y):
            raise ValueError("Monitor coordinates must be finite and nonempty.")

        illumination = self.angular_illumination
        if self.plane_illumination is not None:
            if not np.isscalar(self.z1) or not np.isfinite(self.z1):
                raise ValueError("The RS reference requires a fixed source distance.")
            def illumination(a, b):
                c = bd.sqrt(bd.maximum(1-a*a-b*b, 1e-30))
                return bd.where(a*a+b*b < 1, self.plane_illumination(self.z1*a/c, self.z1*b/c), 0)
        bounds = self.source_bounds if source_bounds is None else source_bounds
        E = rayleigh_sommerfeld_irradiance(pupil, z, x, y, illumination,
            self.z1, 2*self.angular_limit/samples, self.source_subpixels,
            bounds, source_order, monitor_order)
        if backend_name != "jax" or not isinstance(E, jax.core.Tracer):
            self.z = z
            self._irradiance_coordinates = (bd.asarray(x), bd.asarray(y))
        return bd.asarray(x), bd.asarray(y), E

    def get_irradiance(self, finite_distance_model, z):
        """
        Compute irradiance on a finite monitor for the configured source.

        Parameters
        ----------
        finite_distance_model: callback finite_distance_model(F, z) that adds
            optical elements, propagates and returns F. Use F = F.add(...)
            and F = F.zoom_propagate(...), or another supported propagation
            method. In the nonparaxial case, the final step must propagate by z.
        z: positive distance from the final pupil to the monitor, in micrometres.

        Returns
        -------
        x: horizontal monitor coordinates, in micrometres.
        y: vertical monitor coordinates, in micrometres.
        E: irradiance in W/m²; shape (len(y), len(x)) for one-dimensional axes,
            or the broadcast shape of paired coordinate arrays.

        Notes
        -----
        The illumination's paraxial setting selects Eq. (8) or Eq. (24).
        Your callback chooses the coherent propagation model and monitor.
        For nonparaxial zooms, sampling="angular" evaluates the response at
        direction cosines; sampling="fft" uses the coherent monitor intensity.

        With source_bounds and fixed geometry, angular Fresnel and FFT zooms
        evaluate only the region needed by the source convolution. Angular
        ASM does this for ordinary unpropagated pupils when the estimated
        full spectrum exceeds 8192 samples per axis. It evaluates that region
        in blocks; the region including source shifts must lie inside the
        propagating disk. FFT zooms extend the coherent monitor at its original
        spacing when all required directions lie inside that disk.

        To override the nonparaxial monitor, set both x_screen and y_screen.
        One-dimensional arrays give rectangular axes; paired arrays give
        individual points. With source_bounds, each row of paired coordinates
        is a separate region and must remain fixed. Omit bounds to differentiate
        those coordinates. Phase gradients remain supported with fixed geometry.
        PeriodicGaussianPupil uses its own response and requires monitor axes.
        """
        traced_distance = backend_name == "jax" and isinstance(z, jax.core.Tracer)
        if not traced_distance and (not np.isfinite(z) or z <= 0):
            raise ValueError("z must be positive.")
        if self.angular_illumination is None:
            raise ValueError("Configure illumination before propagation.")
        F = MonochromaticField(self.λ, self.extent_x, self.extent_y,
                               self.Nx, self.Ny, intensity=1)
        if hasattr(self, "pupil_quadrature"):
            F.pupil_quadrature = self.pupil_quadrature
            F.x, F.y, F.xx, F.yy = self.x, self.y, self.xx, self.yy
        # Add the common curvature of a source at finite distance.
        if self.plane_illumination is not None:
            F.E = F.E * bd.exp(1j*bd.pi*(F.xx**2+F.yy**2)/(self.λ*self.z1))
        # Record the final step so Eq. (24) samples it only where it is needed.
        if not self.paraxial:
            F = DeferredPropagation(F)
        F = finite_distance_model(F, z)

        if self.paraxial:
            # Eq. (8): source directions translate the coherent screen image.
            x, y = F.x, F.y
            xx, yy = bd.meshgrid(x, y)
            J = self.angular_illumination(-xx/z, -yy/z)
            H = bd.abs(F.E)**2
            shape = (2*len(y), 2*len(x))
            C = bd.fft.irfft2(bd.fft.rfft2(J, s=shape)*bd.fft.rfft2(H, s=shape), s=shape)
            # Align the summed origins; fractional offsets blend neighboring crops.
            E = crop_convolution(C, x, y)
            # Plotting data are cached only in ordinary, untraced evaluations.
            if backend_name != "jax" or not isinstance(E, jax.core.Tracer):
                self.z = z
                self._last_gamma = bd.abs(bd.fft.fftshift(bd.fft.ifft2(bd.fft.ifftshift(J))))*J.size*(x[1]-x[0])*(y[1]-y[0])/z**2
                self._coherence_x = self.λ*z*bd.fft.fftshift(bd.fft.fftfreq(len(x), d=x[1]-x[0]))
                self._coherence_y = self.λ*z*bd.fft.fftshift(bd.fft.fftfreq(len(y), d=y[1]-y[0]))
                self._irradiance_coordinates = (x, y)
            return x, y, bd.maximum(E, 0)*(x[1]-x[0])*(y[1]-y[0])/z**2

        if not isinstance(F, DeferredPropagation) or F.method is None:
            raise ValueError("The model must return F after propagate or zoom_propagate.")
        traced_step = backend_name == "jax" and isinstance(F.distance, jax.core.Tracer)
        if not (traced_distance or traced_step) and not np.isclose(F.distance, z, rtol=1e-12, atol=0):
            raise ValueError("The model's final propagation distance must equal z.")
        if (self.x_screen is None) != (self.y_screen is None):
            raise ValueError("Set both x_screen and y_screen, or leave both as None.")
        # A large sampled pupil can make the global ASM spectrum impractical.
        # Only this previously unbounded case selects the tiled quadrature;
        # existing small-pupil and explicitly paired-region paths are retained.
        large_angular_asm = False
        if (F.model == "ASM" and F.sampling == "angular" and
                F.method == "zoom_propagate" and self.source_bounds is not None and
                self.x_screen is None and not hasattr(F.pupil, "pupil_quadrature") and
                np.isscalar(F.pupil.z) and F.pupil.z == 0):
            geometry = (F.pupil.extent_x, F.pupil.extent_y, F.pupil.λ)
            if all(np.isscalar(v) for v in geometry):
                factor = max(2, int(np.ceil(8*self.angular_limit*max(geometry[:2])/
                                           (geometry[2]*self.angular_samples))))
                large_angular_asm = factor*self.angular_samples > 8192
        # Keep fixed zoom geometry on the host, even while JAX traces the pupil.
        automatic_compact = not traced_step and (
            F.model == "Fresnel" and F.sampling == "angular" or
            F.sampling == "fft" and F.method == "zoom_propagate" or large_angular_asm)
        fixed_zoom = (self.source_bounds is not None and automatic_compact
                      and F.method == "zoom_propagate" and not traced_distance)
        if fixed_zoom and backend_name == "jax":
            fixed_zoom = not any(isinstance(v, jax.core.Tracer)
                                 for v in jax.tree_util.tree_leaves(F.options))
        x_screen, y_screen = (F.screen_axes(host=fixed_zoom) if self.x_screen is None
                              else (self.x_screen, self.y_screen))

        if hasattr(F.pupil, "periodic_gaussian_pupil"):
            # Evaluate the selected Gaussian-order response for the VCSEL pupil.
            if F.model not in ("ASM", "Fresnel") or F.sampling != "angular":
                raise ValueError('PeriodicGaussianPupil requires model="ASM" or "Fresnel" '
                                 'with sampling="angular".')
            if self.angular_illumination is None or self.plane_illumination is not None:
                raise ValueError("Use angular illumination for a periodic Gaussian pupil.")
            a, S = F.pupil.periodic_gaussian_pupil.get_angular_irradiance(
                self.λ, z, self.angular_illumination, self.angular_samples,
                self.angular_limit, self.source_subpixels, model=F.model)
            x, y = bd.asarray(x_screen), bd.asarray(y_screen)
            xx, yy = bd.meshgrid(x, y)
            rho = bd.sqrt(z*z+xx*xx+yy*yy)
            da = a[1]-a[0]
            E = z*z/rho**4*map_coordinates(S,
                bd.array([(yy/rho-a[0])/da, (xx/rho-a[0])/da]),
                order=1, mode="constant", cval=0)
            if backend_name != "jax" or not isinstance(E, jax.core.Tracer):
                self.z = z
                self._irradiance_coordinates = (x, y)
                aa, bb = bd.meshgrid(a, a)
                J = self.angular_illumination(-aa, -bb)
                self._last_gamma = bd.abs(bd.fft.fftshift(bd.fft.ifft2(bd.fft.ifftshift(J))))*J.size*da**2
                self._coherence_x = self.λ*bd.fft.fftshift(bd.fft.fftfreq(len(a), d=da))
                self._coherence_y = self._coherence_x
            return x, y, E

        # A rectangle is one compact region; explicit paired rows retain their
        # existing grouping. The public result keeps the requested monitor shape.
        rectangular = np.ndim(x_screen) == np.ndim(y_screen) == 1
        paired_rows = np.ndim(x_screen) == np.ndim(y_screen) == 2
        traced_geometry = traced_distance
        if backend_name == "jax":
            traced_geometry = traced_geometry or any(isinstance(v, jax.core.Tracer)
                for v in jax.tree_util.tree_leaves((x_screen, y_screen)))
        if self.source_bounds is not None and (paired_rows or
                rectangular and automatic_compact and not traced_geometry):
            if traced_geometry:
                raise ValueError("Compact regions require fixed monitor geometry; omit source_bounds to differentiate it.")
            from .util.compact_source import compact_irradiance
            if rectangular:
                to_host = bd.asnumpy if backend_name == "cupy" else np.asarray
                xx, yy = np.meshgrid(to_host(x_screen), to_host(y_screen))
                points_x, points_y = xx.reshape(1, -1), yy.reshape(1, -1)
            else:
                points_x, points_y = x_screen, y_screen
            E = compact_irradiance(F, z, points_x, points_y,
                self.angular_illumination, self.source_bounds,
                2*self.angular_limit/self.angular_samples, self.source_subpixels,
                tiled_asm=large_angular_asm and rectangular)
            if rectangular:
                E = E.reshape(len(y_screen), len(x_screen))
            if backend_name != "jax" or not isinstance(E, jax.core.Tracer):
                self.z = z
                self._irradiance_coordinates = (bd.asarray(x_screen), bd.asarray(y_screen))
            return bd.asarray(x_screen), bd.asarray(y_screen), E

        source_subpixels = self.source_subpixels
        x, y = bd.asarray(x_screen), bd.asarray(y_screen)
        # K is the coherent response expressed in direction-cosine coordinates.
        a, K = F.angular_propagate(z, self.angular_samples, self.angular_limit)
        da = a[1]-a[0]
        aa, bb = bd.meshgrid(a, a)

        # Evaluate L_in(-a,-b) once, or average within each angular cell.
        if source_subpixels == 1:
            J = bd.where(aa*aa+bb*bb < 1, self.angular_illumination(-aa, -bb), 0)
        else:
            J = bd.zeros_like(aa)
            offsets = ((np.arange(source_subpixels)+0.5)/source_subpixels-0.5)*da
            for du in offsets:
                for dv in offsets:
                    u, v = -aa-du, -bb-dv
                    valid = u*u+v*v < 1
                    value = self.angular_illumination(u, v)
                    J = J+bd.where(valid, value, 0)/source_subpixels**2

        # Zero-padded linear convolution: no periodic images from FFT wraparound.
        shape = (2*len(a),)*2
        S = bd.fft.irfft2(bd.fft.rfft2(J, s=shape)*bd.fft.rfft2(K, s=shape), s=shape)*da**2
        xx, yy = (bd.meshgrid(x, y) if x.ndim == y.ndim == 1
                  else bd.broadcast_arrays(x, y))
        rho = bd.sqrt(z*z+xx*xx+yy*yy)
        alpha, beta = xx/rho, yy/rho
        S = map_coordinates(S, bd.array([(beta-2*a[0])/da, (alpha-2*a[0])/da]),
                            order=1, mode="constant", cval=0)
        # Convert the angular result to irradiance on the flat monitor.
        E = z*z/rho**4*S
        if backend_name != "jax" or not isinstance(E, jax.core.Tracer):
            self.z = z
            self._last_gamma = bd.abs(bd.fft.fftshift(bd.fft.ifft2(bd.fft.ifftshift(J))))*J.size*da**2
            self._coherence_x = self.λ*bd.fft.fftshift(bd.fft.fftfreq(len(a), d=da))
            self._coherence_y = self._coherence_x
            self._last_angular_kernel = (a, K)
            self._irradiance_coordinates = (x, y)
        return x, y, bd.maximum(E, 0)

    def get_farfield(self, far_field_model):
        """
        Compute the angular far field for the configured incoherent source.

        Parameters
        ----------
        far_field_model: callback far_field_model(F) that adds the optics and
            returns F.get_farfield(...), including its two angular axes.

        Returns
        -------
        alpha: one-dimensional horizontal direction-cosine axis.
        beta: one-dimensional vertical direction-cosine axis.
        S: radiant intensity divided by cos(theta), in picowatts/sr, with shape
            (len(beta), len(alpha)); zero outside alpha² + beta² < 1.

        Notes
        -----
        The callback grid must cover sums of output and source directions;
        [-2, 2] on each axis covers all physical pairs. Keep the full coherent
        spectrum until source contributions have been combined. Convert S to
        radiant intensity with to_radiant_intensity(alpha, beta, S), then divide
        by W for W/sr. plot_farfield converts its input to W/sr for display.
        """
        if self.angular_illumination is None:
            raise ValueError("Configure illumination before propagation.")
        F = MonochromaticField(self.λ, self.extent_x, self.extent_y,
                               self.Nx, self.Ny, intensity=1)
        if hasattr(self, "pupil_quadrature"):
            F.pupil_quadrature = self.pupil_quadrature
            F.x, F.y, F.xx, F.yy = self.x, self.y, self.xx, self.yy
        # Add the common curvature of a source at finite distance.
        if self.plane_illumination is not None:
            F.E = F.E*bd.exp(1j*bd.pi*(F.xx**2+F.yy**2)/(self.λ*self.z1))
        a, b, H = far_field_model(F)
        aa, bb = bd.meshgrid(a, b)
        # Reverse the source coordinates to write the angular sum as a convolution.
        J = self.angular_illumination(-aa, -bb)
        J = bd.where(aa*aa+bb*bb < 1, J, 0)
        da, db = a[1]-a[0], b[1]-b[0]
        shape = (2*len(b), 2*len(a))
        S = bd.fft.irfft2(bd.fft.rfft2(J, s=shape)*bd.fft.rfft2(H, s=shape), s=shape)*da*db
        S = crop_convolution(S, a, b)
        if backend_name != "jax" or not isinstance(S, jax.core.Tracer):
            self._last_gamma = bd.abs(bd.fft.fftshift(bd.fft.ifft2(bd.fft.ifftshift(J))))*J.size*da*db
            self._coherence_x = self.λ*bd.fft.fftshift(bd.fft.fftfreq(len(a), d=da))
            self._coherence_y = self.λ*bd.fft.fftshift(bd.fft.fftfreq(len(b), d=db))
        return a, b, bd.where(aa*aa+bb*bb < 1, bd.maximum(S, 0), 0)

    # --------------------- visualization helpers ---------------------

    @staticmethod
    def to_radiant_intensity(alpha, beta, S):
        """
        Convert the angular output S to radiant intensity dP/dΩ.

        Parameters
        ----------
        alpha: one-dimensional horizontal direction-cosine axis.
        beta: one-dimensional vertical direction-cosine axis.
        S: output of get_farfield, with shape (len(beta), len(alpha)).

        Returns
        -------
        intensity: S * sqrt(1 - alpha² - beta²), in the same units as S.
            Divide by W for W/sr when S comes from get_farfield. Values
            outside the propagating disk are zero. JAX gradients are retained.
        """
        from .util.backend_functions import backend as bd
        alpha, beta = bd.asarray(alpha), bd.asarray(beta)
        cosine = bd.sqrt(bd.clip(1-alpha[None, :]**2-beta[:, None]**2, 0, 1))
        return cosine*bd.asarray(S)

    @staticmethod
    def plot_map(x, y, values, *, ax=None, title="", labels=("x", "y"), colorbar_label="",
                    units=1, xlim=None, ylim=None, figsize=(5, 4),
                    cmap="inferno", vmin=None, vmax=None):
        """
        Plot a two-dimensional array with physical axes and a colorbar.

        Parameters
        ----------
        x: ascending one-dimensional horizontal sample coordinates.
        y: ascending one-dimensional vertical sample coordinates.
        values: array of shape (len(y), len(x)), displayed without normalization.
        ax: Matplotlib axes to draw on; None creates and shows a figure.
        title: plot title.
        labels: horizontal and vertical axis labels, including display units.
        colorbar_label: label describing the supplied values and their units.
        units: positive divisor for displayed coordinates; e.g. mm to display millimetres.
        xlim: optional horizontal limits in the original coordinate units.
        ylim: optional vertical limits in the original coordinate units.
        figsize: figure width and height in inches, when ax is None.
        cmap: Matplotlib colormap name or object.
        vmin: lower color limit; None uses the data minimum.
        vmax: upper color limit; None uses the data maximum.

        Returns
        -------
        ax: Matplotlib axes containing the map.

        Notes
        -----
        Nonuniform coordinates are resampled only for display. Plotting copies
        arrays to the CPU and should be called outside JAX transformations.
        """
        import numpy as np
        from .util.backend_functions import backend_name
        if backend_name == "cupy":
            import cupy as cp
            to_numpy = cp.asnumpy
        else:
            to_numpy = np.asarray

        x, y, values = map(to_numpy, (x, y, values))
        if not (np.allclose(np.diff(x), x[1]-x[0], rtol=1e-6, atol=0) and
                np.allclose(np.diff(y), y[1]-y[0], rtol=1e-6, atol=0)):
            # Resample only for display; physical quadrature nodes are nonuniform.
            if xlim is not None:
                lo, hi = max(0, np.searchsorted(x, xlim[0])-1), min(len(x), np.searchsorted(x, xlim[1])+1)
                x, values = x[lo:hi], values[:, lo:hi]
            if ylim is not None:
                lo, hi = max(0, np.searchsorted(y, ylim[0])-1), min(len(y), np.searchsorted(y, ylim[1])+1)
                y, values = y[lo:hi], values[lo:hi]
            xx, yy = np.linspace(x[0], x[-1], 1024), np.linspace(y[0], y[-1], 1024)
            ix, iy = abs(x[:, None]-xx).argmin(axis=0), abs(y[:, None]-yy).argmin(axis=0)
            values = values[iy[:, None], ix[None, :]]
            x, y = xx, yy
        dx, dy = x[1] - x[0], y[1] - y[0]
        extent = [(x[0]-dx/2)/units, (x[-1]+dx/2)/units,
                  (y[0]-dy/2)/units, (y[-1]+dy/2)/units]
        owns_figure = ax is None
        if owns_figure:
            _, ax = plt.subplots(figsize=figsize, layout="constrained")
        im = ax.imshow(values, origin="lower", extent=extent, cmap=cmap,
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set(title=title, xlabel=labels[0], ylabel=labels[1], aspect="equal")
        if xlim is not None:
            ax.set_xlim(np.asarray(xlim)/units)
        if ylim is not None:
            ax.set_ylim(np.asarray(ylim)/units)
        ax.figure.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)
        if owns_figure:
            plt.show()
        return ax

    def plot_source_radiance(self, *, xlim=None, ylim=None, title=None,
                             angular=None, units=None, samples=256, ax=None,
                             figsize=(5, 4), cmap="inferno"):
        """
        Plot the source radiance on its original plane or in angular coordinates.

        Parameters
        ----------
        xlim: horizontal limits in micrometres for a plane source, or direction cosines.
        ylim: vertical limits in the same coordinate system as xlim.
        title: plot title; None selects a title for the source coordinates.
        angular: True for angular coordinates, False for source-plane coordinates;
            None uses the original source plane when available.
        units: coordinate display divisor; defaults to mm for a plane and 1 for angles.
        samples: number of plotting samples per axis, at least 2; does not change
            the propagation grid.
        ax: Matplotlib axes; None creates and shows a figure.
        figsize: figure width and height in inches, when ax is None.
        cmap: Matplotlib colormap name or object.

        Returns
        -------
        ax: Matplotlib axes with a radiance colorbar in W m^-2 sr^-1.

        Notes
        -----
        Default limits span the pupil extents for a plane source and [-1, 1]
        for angular coordinates. Set limits to cover the actual emitter array.
        """
        plane_source = self._source_plane_illumination
        if angular is None:
            angular = plane_source is None
        if angular:
            source = self.angular_illumination
            xlim = (-1, 1) if xlim is None else xlim
            ylim = (-1, 1) if ylim is None else ylim
            units = 1 if units is None else units
            labels = ("α′", "β′") if units == 1 else (f"α′ / {units:g}", f"β′ / {units:g}")
            title = "Angular source radiance" if title is None else title
        else:
            source = plane_source
            xlim = (-self.extent_x/2, self.extent_x/2) if xlim is None else xlim
            ylim = (-self.extent_y/2, self.extent_y/2) if ylim is None else ylim
            units = mm if units is None else units
            unit_name = {m: "m", cm: "cm", mm: "mm", um: "µm", nm: "nm"}.get(units, f"{units/m:g} m")
            labels = (f"x′ [{unit_name}]", f"y′ [{unit_name}]")
            title = "Source-plane radiance" if title is None else title
        if source is None:
            raise ValueError("No source in these coordinates. Configure illumination first.")
        if samples < 2 or units <= 0 or xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
            raise ValueError("Use samples >= 2, positive units, and increasing limits.")

        x = xlim[0] + (bd.arange(samples)+0.5)*(xlim[1]-xlim[0])/samples
        y = ylim[0] + (bd.arange(samples)+0.5)*(ylim[1]-ylim[0])/samples
        xx, yy = bd.meshgrid(x, y)
        radiance = bd.broadcast_to(source(xx, yy), xx.shape)
        if angular:
            radiance = bd.where(xx**2+yy**2 < 1, radiance, 0)
        return self.plot_map(x, y, radiance, ax=ax, title=title, labels=labels,
                                colorbar_label=r"Radiance [W m$^{-2}$ sr$^{-1}$]",
                                units=units, xlim=xlim, ylim=ylim,
                                figsize=figsize, cmap=cmap, vmin=0)

    def plot_aperture(self, aperture, *, xlim=None, ylim=None, title=None,
                      units=mm, phase=False, ax=None, figsize=(5, 4), cmap=None):
        """
        Plot pupil amplitude transmission or wrapped phase.

        Parameters
        ----------
        aperture: optical element with get_transmittance, function (x, y, wavelength),
            or complex transmission array broadcastable to (Ny, Nx).
        xlim: optional horizontal limits, in micrometres.
        ylim: optional vertical limits, in micrometres.
        title: plot title; None selects an amplitude or phase title.
        units: positive coordinate display divisor, default mm.
        phase: True plots phase in [0, 2*pi); False plots amplitude |P|.
        ax: Matplotlib axes; None creates and shows a figure.
        figsize: figure width and height in inches, when ax is None.
        cmap: Matplotlib colormap; None uses gray for amplitude and twilight for phase.

        Returns
        -------
        ax: Matplotlib axes containing the pupil plot.

        Notes
        -----
        Zero-transmission regions have no displayed phase. A configured circular
        pupil quadrature masks transmission outside its radius.
        """
        if hasattr(aperture, "get_transmittance"):
            transmission = aperture.get_transmittance(self.xx, self.yy, self.λ)
        elif callable(aperture):
            transmission = aperture(self.xx, self.yy, self.λ)
        else:
            transmission = aperture
        transmission = bd.broadcast_to(transmission, (self.Ny, self.Nx))
        if hasattr(self, "pupil_quadrature"):
            transmission = bd.where(self.xx**2+self.yy**2 <= self.pupil_quadrature.radius**2,
                                     transmission, 0)
        if units <= 0:
            raise ValueError("units must be positive.")
        unit_name = {m: "m", cm: "cm", mm: "mm", um: "µm", nm: "nm"}.get(units, f"{units/m:g} m")
        labels = (f"x [{unit_name}]", f"y [{unit_name}]")
        if phase:
            values = bd.where(bd.abs(transmission) > 0, bd.angle(transmission) % (2*bd.pi), bd.nan)
            title = "Aperture phase (wrapped)" if title is None else title
            label, default_cmap, vmax = "Phase [rad]", "twilight", 2*bd.pi
        else:
            values = bd.abs(transmission)
            title = "Aperture amplitude" if title is None else title
            label, default_cmap, vmax = r"Amplitude transmission $|P|$", "gray", None
        return self.plot_map(self.x, self.y, values, ax=ax, title=title,
                                labels=labels, colorbar_label=label, units=units,
                                xlim=xlim, ylim=ylim, figsize=figsize,
                                cmap=default_cmap if cmap is None else cmap,
                                vmin=0, vmax=vmax)

    def plot_illumination_plane(self, extent_x=None, extent_y=None, figsize=(7, 6), **kwargs):
        """
        Plot the source in its original plane coordinates.

        Parameters
        ----------
        extent_x: displayed width centred at zero, in micrometres; None uses default limits.
        extent_y: displayed height centred at zero, in micrometres; None uses default limits.
        figsize: figure width and height in inches.
        **kwargs: plot_source_radiance options, such as title, units, samples and ax.
            Explicit xlim and ylim override the corresponding extents.

        Returns
        -------
        ax: Matplotlib axes containing the source radiance.
        """
        if extent_x is not None:
            kwargs.setdefault("xlim", (-extent_x/2, extent_x/2))
        if extent_y is not None:
            kwargs.setdefault("ylim", (-extent_y/2, extent_y/2))
        return self.plot_source_radiance(angular=False, figsize=figsize, **kwargs)

    def plot_angular_illumination(self, source_extent_α=2.0, source_extent_β=2.0,
                                 figsize=(7, 6), **kwargs):
        """
        Plot source radiance versus incident direction cosines.

        Parameters
        ----------
        source_extent_α: displayed horizontal angular width centred at zero.
        source_extent_β: displayed vertical angular width centred at zero.
        figsize: figure width and height in inches.
        **kwargs: plot_source_radiance options, such as title, samples, cmap and ax.
            Explicit xlim and ylim override the corresponding angular widths.

        Returns
        -------
        ax: Matplotlib axes containing the angular source radiance.
        """
        kwargs.setdefault("xlim", (-source_extent_α/2, source_extent_α/2))
        kwargs.setdefault("ylim", (-source_extent_β/2, source_extent_β/2))
        return self.plot_source_radiance(angular=True, figsize=figsize, **kwargs)


    def plot_mutual_coherence(self):
        """
        Plot the source mutual-coherence magnitude from the last calculation.

        Returns
        -------
        ax: Matplotlib axes; separations are in micrometres and |Gamma| in W/m².

        Notes
        -----
        Requires a previous get_farfield or full-grid get_irradiance call outside
        JAX tracing. Compact-monitor calculations do not update this cached map.
        """
        if self._last_gamma is None:
            raise ValueError("Run a simulation before plotting mutual coherence.")
        return self.plot_map(self._coherence_x, self._coherence_y,
            self._last_gamma, ax=None, title="Mutual coherence magnitude",
            labels=("Separation x [µm]", "Separation y [µm]"), units=um,
            colorbar_label=r"$|\Gamma|$ [W m$^{-2}$]", vmin=0)

    def plot_irradiance(self, E, **kwargs):
        """
        Plot irradiance using the monitor coordinates from the last calculation.

        Parameters
        ----------
        E: irradiance array in W/m² on the latest rectangular monitor.
        **kwargs: plot_intensity options, such as units, cmap, text, xlim, ylim,
            slice_y_pos and slice_x_pos. irradiance_units=1e-3 displays mW/m².

        Returns
        -------
        ax: Matplotlib axes containing the irradiance map.

        Notes
        -----
        Call get_irradiance outside JAX tracing first to store the monitor axes.
        Color limits refer to displayed values, including any square root.
        """
        if not hasattr(self, "_irradiance_coordinates"):
            raise ValueError("Call get_irradiance before plot_irradiance.")
        x, y = self._irradiance_coordinates
        return self.plot_intensity(x, y, E, **kwargs)

    def plot_intensity(self, x, y, I, square_root=False, figsize=(7, 6),
                       xlim=None, ylim=None, grid=False, text=None, units=mm,
                       slice_y_pos=None, slice_x_pos=None, dark_background=False,
                       cmap="inferno", vmin=0, vmax=None, irradiance_units=1):
        """
        Plot irradiance on explicit monitor axes, with optional cross-sections.

        Parameters
        ----------
        x: one-dimensional horizontal monitor coordinates, in micrometres.
        y: one-dimensional vertical monitor coordinates, in micrometres.
        I: irradiance in W/m², with shape (len(y), len(x)).
        square_root: display the square root of the irradiance if True.
        figsize: figure width and height in inches.
        xlim: optional horizontal display limits, in micrometres.
        ylim: optional vertical display limits, in micrometres.
        grid: show grid lines if True.
        text: plot title; None uses the last propagation distance.
        units: positive divisor for displayed spatial coordinates, default mm.
        slice_y_pos: optional y position of a horizontal cross-section, in micrometres;
            uses the nearest sampled row.
        slice_x_pos: optional x position of a vertical cross-section, in micrometres;
            uses the nearest sampled column.
        dark_background: use Matplotlib's dark background if True.
        cmap: Matplotlib colormap name or object.
        vmin: lower color limit in displayed units, after any square root.
        vmax: upper color limit in displayed units; None uses the data maximum.
        irradiance_units: positive power-unit divisor; 1 displays W/m²,
            1e-3 displays mW/m² and 1e-6 displays µW/m².

        Returns
        -------
        ax: Matplotlib axes containing the map; cross-sections use a second panel.
        """
        if not np.isfinite(irradiance_units) or irradiance_units <= 0:
            raise ValueError("irradiance_units must be finite and positive.")
        plt.style.use("dark_background" if dark_background else "default")
        to_numpy = (lambda a: a.get()) if backend_name == "cupy" else np.asarray
        x, y, I = map(to_numpy, (x, y, I))
        values = I/irradiance_units
        power_unit = {1: "W", 1e-3: "mW", 1e-6: "µW"}.get(irradiance_units, f"{irradiance_units:g} W")
        label = f"Irradiance [{power_unit} m$^{{-2}}$]"
        if square_root:
            values = np.sqrt(values)
            label = rf"$\sqrt{{E}}$ [({power_unit})$^{{1/2}}$ m$^{{-1}}$]"
        unit = {m: "m", cm: "cm", mm: "mm", um: "µm", nm: "nm"}.get(units, f"{units/m:g} m")
        has_slice = slice_y_pos is not None or slice_x_pos is not None
        fig, axes = plt.subplots(1, 2 if has_slice else 1, figsize=figsize, squeeze=False,
                                 layout="constrained")
        ax = axes[0, 0]
        self.plot_map(x, y, values, ax=ax, title=text or f"Monitor at z = {self.z/mm:g} mm",
                         labels=(f"x [{unit}]", f"y [{unit}]"), colorbar_label=label,
                         units=units, xlim=xlim, ylim=ylim, cmap=cmap, vmin=vmin, vmax=vmax)
        if grid:
            ax.grid(alpha=0.2)
        if has_slice:
            cut = axes[0, 1]
            if slice_y_pos is not None:
                cut.plot(x/units, values[np.argmin(abs(y-slice_y_pos)), :])
                cut.set(xlabel=f"x [{unit}]", ylabel=label)
            if slice_x_pos is not None:
                cut.plot(y/units, values[:, np.argmin(abs(x-slice_x_pos))])
                cut.set(xlabel=f"y [{unit}]", ylabel=label)
        plt.show()
        return ax

    def plot_farfield(self, alpha, beta, S, figsize=(7, 6), alpha_lim=None,
                      beta_lim=None, grid=False, text=None, dark_background=False):
        """
        Plot the angular output S returned by get_farfield.

        Parameters
        ----------
        alpha: one-dimensional horizontal direction-cosine axis.
        beta: one-dimensional vertical direction-cosine axis.
        S: radiant intensity divided by cos(theta), in picowatts/sr, with shape
            (len(beta), len(alpha)).
        figsize: figure width and height in inches.
        alpha_lim: optional horizontal limits in direction-cosine coordinates.
        beta_lim: optional vertical limits in direction-cosine coordinates.
        grid: show grid lines if True.
        text: plot title; None uses "Angular far field".
        dark_background: use Matplotlib's dark background if True.

        Returns
        -------
        ax: Matplotlib axes containing the angular map.

        Notes
        -----
        The display converts S to W/sr. Use to_radiant_intensity to obtain dP/dΩ.
        """
        plt.style.use("dark_background" if dark_background else "default")
        ax = self.plot_map(alpha, beta, S/W, ax=None, title=text or "Angular far field",
            labels=(r"$\alpha$", r"$\beta$"), colorbar_label=r"$S=(dP/d\Omega)/c$ [W sr$^{-1}$]",
            xlim=alpha_lim, ylim=beta_lim, figsize=figsize, vmin=0)
        if grid:
            ax.grid(alpha=0.2)
        return ax

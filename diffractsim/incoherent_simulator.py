"""Simulate diffraction from an extended, spatially incoherent source.

The fast calculation presented in this implementation uses this idea to sum light from a whole source [1].
It computes one reference diffraction pattern, shifts it for each source
direction and weights each copy by the radiance in that direction.
A convolution evaluates the sum. At small angles, the shifts are measured
in distances on the detector. At wide angles, they are measured in
direction cosines.

References
----------
[1] R. de la Fuente Herrezuelo, “Efficient convolution-based nonparaxial
    propagation of partially coherent light” (2026). Equation and section
    numbers below refer to this manuscript. 
"""
from .monochromatic_simulator import MonochromaticField
from .propagation_methods.coherent_response import DeferredPropagation
from .util.convolution import crop_convolution
from .util.constants import W, m, mm, cm, um, nm
import matplotlib.pyplot as plt
import numpy as np

class IncoherentLightSimulation:
    """Predict the light reaching a detector at one wavelength from extended
    incoherent sources.

    Start by describing the source. Use set_illumination_from_plane for
    emitters at known positions and distance, or set_angular_illumination
    when you know the distribution of incoming directions.

    Next describe the optics. The pupil / far_field_model / finite_distance_model 
    specifies how the aperture changes the amplitude and phase of the light across its opening. 
    Pass a function that applies your optics to get_irradiance for a detector at a known
    distance, or to get_farfield for the power sent in each direction.

    Lengths are in micrometres. Use constants such as nm and mm when a
    different unit is more natural. Supply source radiance in W m⁻² sr⁻¹.
    Monitor irradiance is returned in W/m². Section 3.7 of [1] connects
    these two stages of a simulation to the propagation equations. Select
    the NumPy, CuPy or JAX backend before creating the simulation.
    """

    def __init__(self, λ, extent_x, extent_y, Nx, Ny):
        """Set the wavelength and the grid on which the pupil is sampled.

        Parameters
        ----------
        λ : float
            Wavelength in micrometres, for example 940*nm.
        extent_x, extent_y : float
            Width and height of the pupil grid in micrometres.
        Nx, Ny : int
            Number of pupil samples along x and y.

        Notes
        -----
        Choose the pupil spacing fine enough to describe the features in your
        optics. The detector can have a different size and spacing. For example,
        zoom_propagate in the optical callback chooses the part of the detector
        you want to see. See get_irradiance for other choices of monitor points.
        """
        # Use the array library selected before this simulation was created.
        global bd, backend_name, map_coordinates, jax
        from .util.backend_functions import backend as bd, backend_name
        if backend_name == "jax":
            import jax
            from jax.scipy.ndimage import map_coordinates
        elif backend_name == "cupy":
            from cupyx.scipy.ndimage import map_coordinates
        else:
            from scipy.ndimage import map_coordinates

        # The pupil is sampled on a grid with the optical axis at x = y = 0.
        self.λ = λ
        self.Nx, self.Ny = Nx, Ny
        self.extent_x, self.extent_y = extent_x, extent_y

        self.dx = extent_x / Nx
        self.dy = extent_y / Ny
        self.x = self.dx * (bd.arange(Nx) - Nx // 2)
        self.y = self.dy * (bd.arange(Ny) - Ny // 2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        # Keep the original plane source so it can also be plotted in length units.
        self._source_plane_illumination = None
        self.plane_illumination = None
        self.angular_illumination = None
        self.z1 = bd.inf  # An angular source supplies plane waves.
        self.paraxial = True

        # By default the optical callback chooses the detector coordinates.
        self.x_screen = self.y_screen = None

        # A full-grid calculation saves the magnitude of the incident coherence for plotting.
        self._last_gamma = None

    def set_pupil_quadrature(self, radius, order=2):
        """Sample the transmission at several points inside each pupil pixel.

        Parameters
        ----------
        radius : float
            Radius of the circular pupil in micrometres. The circle must fit
            inside the grid.
        order : int
            Number of Gauss integration points along each axis of a pixel.

        Notes
        -----
        A pixel whose centre lies outside the circle may still have a corner
        inside it. The integration weights account for the part of the pixel
        that transmits light. This is useful when the boundary or the phase within a pixel
        matters to the diffracted field.

        Call this method before building the aperture. It updates x, y, xx, yy,
        Nx and Ny to describe the integration points. The physical pixels keep
        their original size. For a finite monitor, use angular sampling with
        model="ASM" or model="Fresnel". With the geometry fixed, you can
        differentiate the result with respect to the sampled pupil values.
        Supplement 1, Section S4.1, explains how refining the pupil sampling
        helps distinguish numerical error from a propagation approximation.
        """
        from diffractsim.util.pupil_quadrature import PupilQuadrature
        previous = getattr(self, "pupil_quadrature", None)
        nx, ny = (self.Nx, self.Ny) if previous is None else (previous.nx, previous.ny)
        self.pupil_quadrature = PupilQuadrature(self.extent_x, self.extent_y,
                                               nx, ny, radius, order)
        self.x, self.y = self.pupil_quadrature.x, self.pupil_quadrature.y
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

    # Illumination

    def set_illumination_from_plane(self, illum_func, distance, paraxial=True,
                                    angular_samples=1024, angular_limit=2.0,
                                    source_subpixels=1, source_bounds=None):
        """Describe the emitting plane by its radiance and its distance from the pupil.

        Parameters
        ----------
        illum_func : callable
            illum_func(x, y) gives the radiance in W m⁻² sr⁻¹. It receives
            arrays of source positions in micrometres.
        distance : float
            Positive distance from the source plane to the pupil in micrometres.
        paraxial : bool
            True uses the small-angle mapping. False uses the actual directions
            from the pupil centre to the source.
        angular_samples : int
            Samples per axis of the wide-angle convolution grid. Use at least 8.
        angular_limit : float
            Half-width of that grid in direction-cosine coordinates.
        source_subpixels : int
            Number of source samples per angular cell along each axis. Values
            above 1 help resolve a sharp source edge in the wide-angle calculation.
        source_bounds : tuple, optional
            (alpha_min, alpha_max, beta_min, beta_max) enclosing all nonzero
            radiance in angular coordinates. See set_angular_illumination.

        Notes
        -----
        The simulation needs the brightness arriving from each direction.
        Starting at the pupil centre, it follows that direction back to the
        emitting plane and reads the radiance there. The direction cosines
        (alpha, beta, c) are the components of a unit vector pointing from the
        pupil to the source. With c = sqrt(1 - alpha² - beta²), the source
        point is (distance*alpha/c, distance*beta/c). At small angles the
        paraxial model takes c as 1. Equations (14)–(15) give the geometry.

        A nearby point also sends a curved wavefront to the pupil. To reuse one
        diffraction pattern, the fast model uses the on-axis quadratic phase
        to approximate the curvature for every point at this distance. Section 3.5 of [1]
        explains the approximation. get_irradiance_rs can check it.
        """
        traced_distance = backend_name == "jax" and isinstance(distance, jax.core.Tracer)
        if not traced_distance and (not np.isfinite(distance) or distance <= 0):
            raise ValueError("Source distance must be finite and positive.")
        def angular_from_plane(alpha, beta):
            """Look up the emitting point seen from the pupil in this direction."""
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
        """Describe the source by its radiance in each direction as seen from the pupil.

        Parameters
        ----------
        illum_func : callable
            illum_func(alpha, beta) gives radiance in W m⁻² sr⁻¹ and accepts
            broadcastable arrays. The coordinates point from the pupil towards
            the emitter. For example, (0, 0) is a source on the optical axis.
        paraxial : bool
            True shifts the reference pattern on a flat detector using the
            small-angle approximation. False uses direction cosines. The
            optical callback chooses how the coherent field propagates.
        angular_samples : int
            Samples per axis of the wide-angle grid for a finite monitor.
            The default is 1024 and the minimum is 8.
        angular_limit : float
            Half-width of that grid. With regular full-grid angular sampling,
            use at least 2 to include all source and detector direction sums.
        source_subpixels : int
            Samples of source radiance per cell along each axis. A value of 1
            gives one sample. Increase it to resolve a narrow emitter or an
            edge without refining the entire coherent response.
        source_bounds : tuple, optional
            (alpha_min, alpha_max, beta_min, beta_max) outside which the
            radiance is zero. With fixed geometry these bounds can reduce
            the region needed for the convolution. See get_irradiance.

        Notes
        -----
        Use this description for a mixture of plane waves, for example after
        collimating an emitter array. Each direction illuminates the pupil
        with a plane wave. Switching from a source plane to this description
        thus removes the incident curvature. The source coordinates look towards
        the emitter, so an emitter at positive x sends light towards negative x.
        This explains the reversed source coordinates in Eq. (18).

        Directions in the forward hemisphere satisfy alpha² + beta² < 1. An
        incident and an outgoing direction can each lie inside this disk while
        their sum lies outside it. The coherent response must cover these sums.
        For a PeriodicGaussianPupil, cover the selected diffraction orders and
        choose a grid step that divides wavelength/period so each order falls
        on a sample. For get_farfield the callback supplies the grid directly.
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
        """Return the irradiance incident on the pupil before it transmits any light.

        Returns
        -------
        irradiance : scalar
            Incident irradiance at the pupil centre in W/m².

        Notes
        -----
        Set source_bounds to enclose the emitting directions. The integral adds
        the radiance in each cell times its area in direction-cosine coordinates.
        This area already includes the cosine that projects the incoming light
        onto the pupil plane. The angular grid spacing and source_subpixels
        control how finely the integral samples the source.

        The result is the zero-separation value of the coherence function in
        Eq. (16). For uniform illumination across a pupil, multiplying by its
        area in square metres gives the incident power in watts. This provides a
        reference for the fraction of power collected on a detector.
        """
        if self.angular_illumination is None:
            raise ValueError("Configure illumination first.")
        if self.source_bounds is None:
            raise ValueError("Set source_bounds to integrate the source support.")
        from .util.compact_source import source_kernel
        _, _, mass = source_kernel(self.angular_illumination, self.source_bounds,
            2*self.angular_limit/self.angular_samples, self.source_subpixels)
        return bd.sum(mass)

    # Propagation

    def get_irradiance_rs(self, finite_distance_model, z, *, source_order=3,
                          monitor_order=7, angular_samples=None, source_bounds=None,
                          Nx_mon=None, Ny_mon=None):
        """Check a finite-monitor prediction with the Rayleigh–Sommerfeld (RS) integral.

        Parameters
        ----------
        finite_distance_model : callable
            Use the same function as for get_irradiance. It supplies a single
            sampled pupil and ends with free-space propagation by z. The last
            step defines the monitor. The reference evaluates the RS kernel
            for either choice of coherent propagation model.
        z : float
            Fixed positive distance from the pupil to the monitor in micrometres.
        source_order, monitor_order : int
            Number of interpolation nodes per axis in source and monitor slope,
            respectively. A slope is a transverse displacement divided by the
            axial distance. Both orders must be at least 2. The defaults are
            3 and 7. Only a source plane needs the source interpolation.
        angular_samples : int, optional
            Angular resolution for this check. The default uses the configured
            spacing. Increasing it refines the angular integral while the number
            of returned monitor points stays the same.
        source_bounds : tuple, optional
            (alpha_min, alpha_max, beta_min, beta_max) enclosing the source.
            By default the reference uses the configured bounds or finds the
            nonzero source cells. For a finite source plane, all four corners
            of the rectangle must lie inside alpha² + beta² < 1.
        Nx_mon, Ny_mon : int, optional
            Override the sample counts of a zoom monitor when x_screen and
            y_screen are unset. Fewer points can make an initial check cheaper.

        Returns
        -------
        x, y, E : arrays
            Monitor coordinates in micrometres and irradiance in W/m². The
            coordinate and shape conventions are the same as in get_irradiance.

        Notes
        -----
        For a point on a source plane, Eq. (25) follows the spherical wave from
        the emitter across the pupil and then from the pupil to the detector.
        An angular source supplies a plane wave for the first leg. For each
        emitter, first add the complex fields from across the pupil. The squared
        magnitude of that sum gives its diffraction pattern. Then add the emitter
        patterns, weighting each by its radiance. This tests the detector
        remapping in Eq. (24) and, for a finite source plane, the shared incident
        curvature as well. The reference uses the true directions even when the
        fast calculation was set to paraxial.

        Evaluating this integral separately at every source and monitor point
        would be expensive. After accounting for the wavefront tilt, the
        remaining factors vary smoothly with direction. The code evaluates the
        full kernels at selected directions and interpolates the resulting
        intensity response between them. Increase source_order, monitor_order
        and angular_samples one at a time to check convergence. Source edges
        also depend on source_subpixels, and the pupil needs sufficient samples
        to resolve its transmission. Supplement 1, Sections S3.1 and S5.3, of [1]
        give the reference integral and an example of these checks.

        Keep the geometry and interpolation orders fixed. To differentiate
        source radiance with JAX, supply fixed source_bounds. Gradients with
        respect to the sampled pupil phase are also supported. For a periodic
        Gaussian pupil, sample the repeated transmission across the illuminated
        pupil, including its Gaussian envelope.
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
        # The reference traces the true direction back to the source plane,
        # including when the fast calculation used small angles.
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
        """Compute the irradiance on a flat detector at a finite distance.

        Parameters
        ----------
        finite_distance_model : callable
            A function finite_distance_model(F, z) that adds the optics and
            returns the propagated field. Assign the field returned by each
            step, for example F = F.add(aperture). For the wide-angle model,
            finish with F.propagate(z) or F.zoom_propagate(z, ...).
        z : float
            Positive distance from the final pupil to the detector in micrometres.

        Returns
        -------
        x, y, E : arrays
            Detector coordinates in micrometres and irradiance in W/m². For
            one-dimensional axes, E[j, i] is the irradiance at (x[i], y[j]).
            With paired coordinates, E has their broadcast shape.

        Notes
        -----
        At small angles, moving an emitter sideways shifts its diffraction
        pattern across the detector. Eq. (8) adds these shifted patterns with
        weights set by the source radiance.

        At wide angles, equal changes in direction correspond to larger
        distances near the edge of a flat detector than near its centre.
        Eq. (24) therefore expresses the reference pattern as power per unit
        area in direction-cosine coordinates. It shifts and adds the patterns
        there, then converts the sum back to power per unit detector area.
        Section 3.6 of [1] and Supplement 1, Section S2.7, explain the
        finite-distance approximation. Use get_irradiance_rs to check it.

        The source's paraxial setting selects the rule for adding intensities.
        The callback selects the coherent propagation and the monitor. The
        angular-spectrum method ("ASM") retains the full propagation phase,
        while "Fresnel" uses a paraxial approximation. For a wide-angle zoom,
        sampling="angular" evaluates the response at the required directions.
        With sampling="fft", the code propagates onto a rectangular grid and
        reads the intensity at the corresponding points.

        To choose different monitor points for the wide-angle calculation, set
        both x_screen and y_screen. Two one-dimensional arrays give all
        combinations of x and y. Two-dimensional arrays specify pairs of
        coordinates. With source_bounds, each row of pairs defines a region
        that can be evaluated separately. Keep these regions fixed when using
        JAX. Omit source_bounds to differentiate monitor coordinates on the
        full grid.

        If the source has known bounds and the monitor is fixed, a Fresnel
        angular zoom or an FFT zoom needs only the directions that contribute
        to the requested region. When all these directions lie inside the unit
        disk, an FFT zoom can extend the coherent grid at its original spacing.
        An angular-spectrum zoom of a regular, unpropagated pupil also uses
        smaller regions when its full spectrum would exceed 8192 samples per
        axis. This last calculation works in blocks and needs the region,
        including the source shifts, to fit inside the disk. These choices
        allow gradients with respect to the pupil values. PeriodicGaussianPupil
        uses its diffraction orders and requires a pair of monitor axes.
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
        # Give the reference wave the curvature of the on-axis emitter (Eq. 22).
        if self.plane_illumination is not None:
            F.E = F.E * bd.exp(1j*bd.pi*(F.xx**2+F.yy**2)/(self.λ*self.z1))
        # Delay the final propagation until the required monitor directions are known.
        if not self.paraxial:
            F = DeferredPropagation(F)
        F = finite_distance_model(F, z)

        if self.paraxial:
            # A source point seen to the right moves its pattern to the left.
            # Eq. (8) therefore uses the source at (-x/z, -y/z).
            x, y = F.x, F.y
            xx, yy = bd.meshgrid(x, y)
            J = self.angular_illumination(-xx/z, -yy/z)
            H = bd.abs(F.E)**2
            # Leave room beyond the detector edges for the shifted patterns.
            shape = (2*len(y), 2*len(x))
            C = bd.fft.irfft2(bd.fft.rfft2(J, s=shape)*bd.fft.rfft2(H, s=shape), s=shape)
            # The first sample of a convolution is at the sum of the two origins.
            # Read it back on the detector axes, interpolating between samples if needed.
            E = crop_convolution(C, x, y)
            # Save concrete values for plotting after a forward calculation.
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
        # A wide pupil needs fine angular sampling. If a full spectrum would be
        # too large, compute only the regions that can illuminate this detector.
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
        # With fixed geometry we can choose the required directions once,
        # even when JAX is differentiating the pupil values.
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
            # Combine the source with the broadened diffraction orders of the
            # Gaussian pupil, then project that angular pattern onto the detector.
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

        # A rectangular detector is one region. For separate groups of detector
        # points, each row of coordinate pairs describes one region.
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
        # Express the reference pattern on an angular grid so every source
        # direction can shift a copy of the same array.
        a, K = F.angular_propagate(z, self.angular_samples, self.angular_limit)
        da = a[1]-a[0]
        aa, bb = bd.meshgrid(a, a)

        # Reverse the source axes because they point towards the emitters.
        # Sampling within a cell helps resolve an edge that crosses it.
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

        # The FFT treats a grid as if it repeats. Pad both arrays with zeros
        # so light shifted past one edge does not reappear at the opposite edge.
        # Each source cell contributes its radiance times its angular area, da².
        shape = (2*len(a),)*2
        S = bd.fft.irfft2(bd.fft.rfft2(J, s=shape)*bd.fft.rfft2(K, s=shape), s=shape)*da**2
        xx, yy = (bd.meshgrid(x, y) if x.ndim == y.ndim == 1
                  else bd.broadcast_arrays(x, y))
        rho = bd.sqrt(z*z+xx*xx+yy*yy)
        alpha, beta = xx/rho, yy/rho
        S = map_coordinates(S, bd.array([(beta-2*a[0])/da, (alpha-2*a[0])/da]),
                            order=1, mode="constant", cval=0)
        # A fixed detector area spans less of the angular grid towards the edges.
        # The factor z²/rho⁴ converts power on that grid to power per detector area.
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
        """Find how the power leaving the pupil is distributed over outgoing directions.

        Parameters
        ----------
        far_field_model : callable
            A function far_field_model(F) that adds the optics and returns
            F.get_farfield(...). Its two uniformly spaced angular axes set the
            sampling for both the source and the coherent response.

        Returns
        -------
        alpha, beta, S : arrays
            Horizontal and vertical direction-cosine axes and an angular power
            distribution of shape (len(beta), len(alpha)). In diffractsim's
            power units, S is (dP/dΩ)/cos(theta) in picowatts per steradian.
            S is zero outside the forward disk alpha² + beta² < 1.

        Notes
        -----
        Each incident plane wave shifts the same coherent far-field pattern.
        For output (alpha, beta) and a source seen at (u, v), the required
        value of that pattern is at (alpha + u, beta + v). Choose callback
        axes that cover these sums as well as the emitting directions.
        An interval from -2 to 2 on each axis covers every physical pair.
        Keep the coherent response over the full interval until after adding
        the source contributions.

        For a prescribed mixture of plane waves, this is Eq. (18) within the
        scalar power convention of [1, 3]. For a source on a finite plane,
        Eq. (22) includes the common incident curvature. As Eq. (13) shows,
        integrating S over d(alpha) d(beta) gives power.
        For power per solid angle, use to_radiant_intensity(alpha, beta, S)
        and divide the result by W to express it in watts per steradian.
        """
        if self.angular_illumination is None:
            raise ValueError("Configure illumination before propagation.")
        F = MonochromaticField(self.λ, self.extent_x, self.extent_y,
                               self.Nx, self.Ny, intensity=1)
        if hasattr(self, "pupil_quadrature"):
            F.pupil_quadrature = self.pupil_quadrature
            F.x, F.y, F.xx, F.yy = self.x, self.y, self.xx, self.yy
        # Give the reference wave the curvature of the on-axis emitter (Eq. 22).
        if self.plane_illumination is not None:
            F.E = F.E*bd.exp(1j*bd.pi*(F.xx**2+F.yy**2)/(self.λ*self.z1))
        a, b, H = far_field_model(F)
        aa, bb = bd.meshgrid(a, b)
        # Eq. (18) needs the pupil response at outgoing plus source direction.
        # Reversing the source axes gives that sum in an ordinary convolution.
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

    # Plotting and unit conversion

    @staticmethod
    def to_radiant_intensity(alpha, beta, S):
        """Convert the far-field distribution to radiant intensity (power per solid angle).

        Parameters
        ----------
        alpha, beta : one-dimensional arrays
            Horizontal and vertical direction cosines returned by get_farfield.
        S : array
            Angular distribution returned by get_farfield, with shape
            (len(beta), len(alpha)).

        Returns
        -------
        intensity : array
            Radiant intensity dP/dΩ on the same grid. Divide by W to express
            the result in watts per steradian when S comes from get_farfield.

        Notes
        -----
        A patch of solid angle dΩ has projected area cos(theta) dΩ on the
        direction-cosine disk. This is why a distribution per unit area on
        that disk must be multiplied by cos(theta) to give power per solid
        angle. Here cos(theta) = sqrt(1 - alpha² - beta²). Points on or
        outside the rim receive zero. See Eq. (13) and Section 3.1 of [1].
        """
        from .util.backend_functions import backend as bd
        alpha, beta = bd.asarray(alpha), bd.asarray(beta)
        cosine = bd.sqrt(bd.clip(1-alpha[None, :]**2-beta[:, None]**2, 0, 1))
        return cosine*bd.asarray(S)

    @staticmethod
    def plot_map(x, y, values, *, ax=None, title="", labels=("x", "y"), colorbar_label="",
                    units=1, xlim=None, ylim=None, figsize=(5, 4),
                    cmap="inferno", vmin=None, vmax=None):
        """Display a sampled quantity with its physical coordinates and a colorbar.

        Parameters
        ----------
        x, y : one-dimensional arrays
            Ascending horizontal and vertical sample positions.
        values : array
            Values on the grid. Row j and column i correspond to (x[i], y[j]).
        ax : Matplotlib axes, optional
            Axes to draw on. If omitted, create and show a figure.
        title : str
            Plot title.
        labels : pair of strings
            Horizontal and vertical axis labels, including the displayed units.
        colorbar_label : str
            Name and units of the plotted quantity.
        units : float
            Positive divisor for the displayed coordinates. For example, mm
            displays positions supplied in micrometres on a millimetre scale.
        xlim, ylim : pairs, optional
            Visible intervals in the original coordinate units.
        figsize : pair
            Width and height in inches when creating a figure.
        cmap : colormap or name
            Matplotlib colormap.
        vmin, vmax : floats, optional
            Lower and upper limits of the color scale. By default the values
            determine each unspecified limit.

        Returns
        -------
        ax : Matplotlib axes
            Axes containing the map.

        Notes
        -----
        An image displays pixels at equal spacing. If the samples are unevenly
        spaced, as they are for pupil quadrature, the plot uses the nearest
        sample at each position on a uniform display grid. The simulation keeps
        its original samples. Plot after a calculation, outside JAX tracing,
        since Matplotlib needs the values on the CPU.
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
            # Each display pixel takes the value of the nearest physical sample.
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
        """Show where the source emits light or which directions it illuminates.

        Parameters
        ----------
        xlim, ylim : pairs, optional
            Visible intervals in micrometres on a source plane or in direction
            cosines for an angular view.
        title : str, optional
            By default the title identifies the source coordinates.
        angular : bool, optional
            True selects the angular view. False selects the original source
            plane. By default a source defined on a plane is shown on that plane.
        units : float, optional
            Divisor for the displayed coordinates. The defaults are mm for a
            plane and 1 for direction cosines.
        samples : int
            Number of plotting samples along each axis, at least 2.
        ax : Matplotlib axes, optional
            Axes to draw on. If omitted, create and show a figure.
        figsize : pair
            Width and height in inches when creating a figure.
        cmap : colormap or name
            Matplotlib colormap.

        Returns
        -------
        ax : Matplotlib axes
            Source plot with radiance in W m⁻² sr⁻¹.

        Notes
        -----
        The plane view initially spans the pupil width and height. For an emitter
        array wider than the pupil, set xlim and ylim to include the whole array.
        The angular view looks out from the pupil and initially spans -1 to 1
        on each axis. Both views sample the source function for the plot, so
        you can adjust the plotted detail independently of the propagation.
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
        """Show how the pupil transmits the amplitude or changes the phase of a field.

        Parameters
        ----------
        aperture : optical element, callable or array
            Supply an element with get_transmittance, a function of (x, y,
            wavelength), or complex transmission values that broadcast to (Ny, Nx).
        xlim, ylim : pairs, optional
            Visible intervals in micrometres.
        title : str, optional
            By default the title identifies the amplitude or phase view.
        units : float
            Positive divisor for the displayed coordinates. The default is mm.
        phase : bool
            False shows amplitude transmission |P|. True shows phase in radians,
            wrapped into the interval from 0 up to 2π.
        ax : Matplotlib axes, optional
            Axes to draw on. If omitted, create and show a figure.
        figsize : pair
            Width and height in inches when creating a figure.
        cmap : colormap or name, optional
            The defaults are gray for amplitude and twilight for phase.

        Returns
        -------
        ax : Matplotlib axes
            Axes containing the pupil plot.

        Notes
        -----
        Where the pupil blocks the light, the transmitted field has no defined
        phase, so that part of the phase plot is blank. If you set a circular
        pupil with set_pupil_quadrature, the plot also masks the area outside
        its radius.
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
        """Show the radiance at each position of the original emitting plane.

        Parameters
        ----------
        extent_x, extent_y : floats, optional
            Width and height in micrometres, centred on zero. By default the
            view spans the pupil. Use xlim and ylim for an off-centre source.
        figsize : pair
            Figure width and height in inches.
        **kwargs
            Options for plot_source_radiance, such as units, samples, title and
            ax. An explicit axis interval takes precedence over its extent.

        Returns
        -------
        ax : Matplotlib axes
            Source-plane plot with radiance in W m⁻² sr⁻¹.
        """
        if extent_x is not None:
            kwargs.setdefault("xlim", (-extent_x/2, extent_x/2))
        if extent_y is not None:
            kwargs.setdefault("ylim", (-extent_y/2, extent_y/2))
        return self.plot_source_radiance(angular=False, figsize=figsize, **kwargs)

    def plot_angular_illumination(self, source_extent_α=2.0, source_extent_β=2.0,
                                 figsize=(7, 6), **kwargs):
        """Show the source radiance as seen in each direction from the pupil.

        Parameters
        ----------
        source_extent_α, source_extent_β : floats
            Full horizontal and vertical widths in direction cosines, centred
            on zero. The default width of 2 covers -1 to 1 on each axis.
        figsize : pair
            Figure width and height in inches.
        **kwargs
            Options for plot_source_radiance, such as samples, cmap, title and
            ax. Use xlim or ylim to select a different interval on either axis.

        Returns
        -------
        ax : Matplotlib axes
            Angular source plot with radiance in W m⁻² sr⁻¹.
        """
        kwargs.setdefault("xlim", (-source_extent_α/2, source_extent_α/2))
        kwargs.setdefault("ylim", (-source_extent_β/2, source_extent_β/2))
        return self.plot_source_radiance(angular=True, figsize=figsize, **kwargs)


    def plot_mutual_coherence(self):
        """Show how strongly the incident fields at two pupil points are correlated.

        Returns
        -------
        ax : Matplotlib axes
            A map of |Γ| in W/m² against horizontal and vertical separation
            in micrometres.

        Notes
        -----
        Each point on this plot is a separation between two positions on the
        pupil. At the origin the two positions coincide, so |Γ| there gives the
        incident irradiance represented by the sampled source. The rest of the
        map shows how the correlation changes as the positions move apart.
        Equations (4) and (16) relate this function to the source radiance for
        small and wide angles, respectively.

        The map comes from the most recent calculation that stored it. After
        changing the source, run get_farfield or a full-grid get_irradiance
        outside JAX tracing to refresh it. A compact-monitor calculation or an
        RS check leaves the previous map in place.
        """
        if self._last_gamma is None:
            raise ValueError("Run a simulation before plotting mutual coherence.")
        return self.plot_map(self._coherence_x, self._coherence_y,
            self._last_gamma, ax=None, title="Mutual coherence magnitude",
            labels=("Separation x [µm]", "Separation y [µm]"), units=um,
            colorbar_label=r"$|\Gamma|$ [W m$^{-2}$]", vmin=0)

    def plot_irradiance(self, E, **kwargs):
        """Plot irradiance on the rectangular monitor used in your last calculation.

        Parameters
        ----------
        E : array
            Monitor irradiance in W/m².
        **kwargs
            Options for plot_intensity, including units, cmap, text, xlim,
            ylim and the positions of cross-sections. Use irradiance_units=1e-3
            to label the color scale in mW/m².

        Returns
        -------
        ax : Matplotlib axes
            Axes containing the irradiance map.

        Notes
        -----
        Compute E with get_irradiance or get_irradiance_rs outside JAX tracing
        so the simulator can remember the axes. If another calculation has since
        changed the monitor, pass the original x, y and E to plot_intensity.
        """
        if not hasattr(self, "_irradiance_coordinates"):
            raise ValueError("Call get_irradiance before plot_irradiance.")
        x, y = self._irradiance_coordinates
        return self.plot_intensity(x, y, E, **kwargs)

    def plot_intensity(self, x, y, I, square_root=False, figsize=(7, 6),
                       xlim=None, ylim=None, grid=False, text=None, units=mm,
                       slice_y_pos=None, slice_x_pos=None, dark_background=False,
                       cmap="inferno", vmin=0, vmax=None, irradiance_units=1):
        """Plot irradiance on the supplied monitor axes and optionally show a line profile.

        Parameters
        ----------
        x, y : one-dimensional arrays
            Horizontal and vertical monitor positions in micrometres.
        I : array
            Irradiance in W/m² with shape (len(y), len(x)).
        square_root : bool
            True displays the square root of the irradiance to reveal dim features.
        figsize : pair
            Figure width and height in inches.
        xlim, ylim : pairs, optional
            Visible intervals in micrometres.
        grid : bool
            True adds grid lines to the map.
        text : str, optional
            Plot title. By default the title gives the last propagation distance.
        units : float
            Positive divisor for the displayed positions. The default is mm.
        slice_y_pos, slice_x_pos : floats, optional
            A y position selects a horizontal profile and an x position selects a
            vertical one. Supply positions in micrometres. The plot uses the
            nearest sampled row or column.
        dark_background : bool
            True selects Matplotlib's dark background.
        cmap : colormap or name
            Matplotlib colormap.
        vmin, vmax : floats, optional
            Limits of the displayed color scale after unit conversion and any
            square root. By default the scale starts at zero and ends at the
            largest displayed value.
        irradiance_units : float
            Power unit for the display. Use 1 for W/m², 1e-3 for mW/m² or
            1e-6 for µW/m².

        Returns
        -------
        ax : Matplotlib axes
            Axes containing the map. Any requested profiles share a second panel.
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
        """Display the far-field distribution S on a map of outgoing directions.

        Parameters
        ----------
        alpha, beta : one-dimensional arrays
            Horizontal and vertical direction cosines returned by get_farfield.
        S : array
            Angular distribution returned by get_farfield, in picowatts per
            steradian and with shape (len(beta), len(alpha)).
        figsize : pair
            Figure width and height in inches.
        alpha_lim, beta_lim : pairs, optional
            Visible intervals in direction cosines.
        grid : bool
            True adds grid lines.
        text : str, optional
            Plot title. The default is “Angular far field”.
        dark_background : bool
            True selects Matplotlib's dark background.

        Returns
        -------
        ax : Matplotlib axes
            Axes containing the angular map.

        Notes
        -----
        The colorbar displays S in W/sr. S gives power when integrated over the
        direction-cosine plane. For radiant intensity per unit solid angle, use
        to_radiant_intensity to apply the cosine factor in Eq. (13).
        """
        plt.style.use("dark_background" if dark_background else "default")
        ax = self.plot_map(alpha, beta, S/W, ax=None, title=text or "Angular far field",
            labels=(r"$\alpha$", r"$\beta$"), colorbar_label=r"$S=(dP/d\Omega)/c$ [W sr$^{-1}$]",
            xlim=alpha_lim, ylim=beta_lim, figsize=figsize, vmin=0)
        if grid:
            ax.grid(alpha=0.2)
        return ax

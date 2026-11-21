from diffractsim import colour_functions as cf
import matplotlib.pyplot as plt
import progressbar
import time
import numpy as np
from .util.backend_functions import backend as bd
from .util.backend_functions import backend_name
from .util.constants import *
from .monochromatic_simulator import MonochromaticField
from .incoherent_simulator import IncoherentLightSimulation

"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


class PolychromaticSimulator:
    """
    Polychromatic driver class that records an optical setup once,
    then replays the same sequence on a fresh MonochromaticField
    for each wavelength sample of the spectrum.

    Usage:
        def optical_setup(F):
            F = F.add(ApertureFromImage(..., simulation=F))
            F = F.add(Lens(f=...))
            F = F.propagate(z=...)
            return F

        sim = PolychromaticSimulator(spectrum=2*cf.illuminant_d65,
                                     extent_x=18*mm, extent_y=18*mm,
                                     Nx=1024, Ny=1024,
                                     optical_setup=optical_setup)
        rgb = sim.get_colors()
        sim.plot_colors(rgb, xlim=[-7*mm, 7*mm], ylim=[-7*mm, 7*mm])
    """
    global bd
    global backend_name
    from .util.backend_functions import backend as bd
    from .util.backend_functions import backend_name

    # ---------- internal helper: recipe builder ----------
    class _RecipeBuilder:
        """
        Lightweight stand-in that mimics the MonochromaticField chaining API
        but only records steps. It also exposes grid attributes so optical
        elements that need `simulation=F` can size themselves during setup.
        """
        def __init__(self, extent_x, extent_y, Nx, Ny):
            global bd
            global backend_name
            from .util.backend_functions import backend as bd
            from .util.backend_functions import backend_name


            self.extent_x = extent_x
            self.extent_y = extent_y
            self.Nx = Nx
            self.Ny = Ny
            self.dx = extent_x / Nx
            self.dy = extent_y / Ny
            self.x = self.dx * (bd.arange(Nx) - Nx // 2)
            self.y = self.dy * (bd.arange(Ny) - Ny // 2)
            self.xx, self.yy = bd.meshgrid(self.x, self.y)

            self._steps = []  # list of tuples: (kind, args)

        # recorders: return self so user code can do F = F.add(...), F = F.propagate(...)
        def add(self, optical_element):
            self._steps.append(("optical_element", (optical_element,)))
            return self

        def propagate(self, z, scale_factor=1, model="ASM"):
            """Record ASM or two-step Fresnel propagation for each wavelength.

            Fresnel uses the paraxial approximation; arguments match MonochromaticField.
            Reference: D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
            SPIE (2011), doi:10.1117/3.858456.
            """
            if model not in ("ASM", "Fresnel"):
                raise ValueError('model must be "ASM" or "Fresnel".')
            self._steps.append(("propagate", (z, scale_factor, model)))
            return self


        def zoom_propagate(self, z, x_interval, y_interval, model="Fresnel",
                           sampling="fft", padding=1, Nx_mon=None, Ny_mon=None):
            for n in (Nx_mon, Ny_mon):
                if n is not None and (isinstance(n, (bool, np.bool_)) or
                        not isinstance(n, (int, np.integer)) or n < 2):
                    raise ValueError("Monitor counts must be integers >= 2.")
            self._steps.append(("zoom_propagate", (z, x_interval, y_interval, model,
                                                   sampling, padding, Nx_mon, Ny_mon)))
            self.Nx = self.Nx if Nx_mon is None else Nx_mon
            self.Ny = self.Ny if Ny_mon is None else Ny_mon
            self.x = bd.linspace(*x_interval, self.Nx)
            self.y = bd.linspace(*y_interval, self.Ny)
            self.dx, self.dy = self.x[1]-self.x[0], self.y[1]-self.y[0]
            self.extent_x, self.extent_y = self.Nx*self.dx, self.Ny*self.dy
            self.xx, self.yy = bd.meshgrid(self.x, self.y)
            return self

        def propagate_to_image_plane(self, pupil, M, zi, z0, scale_factor=1):
            self._steps.append(("propagate_to_image_plane", (pupil, M, zi, z0, scale_factor)))
            return self

        def propagate_to_lens_focal_plane(self, focal_length, x_interval, y_interval):
            self._steps.append(("propagate_to_lens_focal_plane", (focal_length, x_interval, y_interval)))
            return self

    # ---------- simulator ----------
    def __init__(self, spectrum, extent_x, extent_y, Nx, Ny,
                 optical_setup=None, spectrum_size=180, spectrum_divisions=30):

        """
        Initialize a spectral simulation and record its optical setup.

        Parameters
        ----------
        spectrum: incident spectral irradiance sampled at 400 wavelengths from
            380 to 779 nm, in W m^-2 nm^-1.
        extent_x: width of the input pupil grid, in micrometres.
        extent_y: height of the input pupil grid, in micrometres.
        Nx: horizontal number of input samples.
        Ny: vertical number of input samples.
        optical_setup: optional callback optical_setup(F) that adds optical
            elements and propagation steps, returning F. The recorded sequence
            is replayed for each wavelength.
        spectrum_size: number of samples used to resample the input spectrum.
        spectrum_divisions: number of wavelength partitions to propagate;
            must divide spectrum_size exactly.

        Notes
        -----
        Illumination starts as a coherent plane wave for each wavelength.
        Use set_angular_illumination for a source specified by direction, or
        set_illumination_from_plane for an emitting plane at a finite distance.
        """
        global bd
        global backend_name
        from .util.backend_functions import backend as bd
        from .util.backend_functions import backend_name

        # grid/meta
        self.extent_x = extent_x
        self.extent_y = extent_y
        self.Nx = Nx
        self.Ny = Ny

        self._input_grid = (extent_x, extent_y, Nx, Ny)

        # spectrum handling (same behaviour as previous class)
        if not (spectrum_size / spectrum_divisions).is_integer():
            raise ValueError("spectrum_size/spectrum_divisions must be an integer")

        if spectrum_size == 400:
            self.spectrum = bd.array(spectrum)
        else:
            self.spectrum = bd.array(
                bd.interp(bd.asarray(np.linspace(380, 779, spectrum_size)),
                          bd.asarray(np.linspace(380, 779, 400)),
                          bd.asarray(spectrum))
            )

        self.spectrum_size = spectrum_size
        self.spectrum_divisions = spectrum_divisions
        self.dλ_partition = (780 - 380) / self.spectrum_divisions
        # Wavelength samples are fixed configuration, including under JAX JIT.
        self.λ_list_samples = np.arange(380, 780, self.dλ_partition)  # in nm
        self.spec_partitions = bd.split(self.spectrum, self.spectrum_divisions)

        # colour system (same as before)
        self.cs = cf.ColourSystem(spectrum_size=spectrum_size,
                                  spec_divisions=spectrum_divisions,
                                  clip_method=1)

        # record the optical setup once
        builder = PolychromaticSimulator._RecipeBuilder(extent_x, extent_y, Nx, Ny)
        if optical_setup is not None:
            # user returns F, but we only need builder._steps
            optical_setup(builder)
        self._recipe_steps = builder._steps
        self._illumination = None  # None means coherent plane-wave illumination.

    # ---------- main API ----------
    def set_angular_illumination(self, illum_func, paraxial=False,
                                 angular_samples=1024, angular_limit=2.0,
                                 source_subpixels=1, source_bounds=None):
        """
        Illuminate the pupil with mutually incoherent incident directions.

        Parameters
        ----------
        illum_func: function illum_func(alpha, beta) giving the angular radiance
            profile shared by all wavelengths. Normalize its integral over
            direction cosines to one if spectrum should specify the incident
            spectral irradiance. None restores coherent plane-wave illumination.
        paraxial: True uses the paraxial convolution; False uses the nonparaxial
            convolution. optical_setup still chooses the coherent propagator.
        angular_samples: samples per axis of the nonparaxial convolution grid, at least 8.
        angular_limit: positive half-width of that grid in direction-cosine coordinates.
        source_subpixels: positive number of samples per cell axis for source averaging;
            1 samples the centre of each cell.
        source_bounds: optional (alpha_min, alpha_max, beta_min, beta_max)
            outside which the source is zero. These bounds let supported
            propagators evaluate only the region needed by the monitor.

        Returns
        -------
        simulation: this simulator, so calls can be chained.

        Notes
        -----
        This replaces the previous source, including any source-plane curvature.
        The optical setup must contain pupil-plane elements followed by a single
        propagate or zoom_propagate step.
        """
        return self._set_illumination(illum_func,
            paraxial=paraxial, angular_samples=angular_samples,
            angular_limit=angular_limit, source_subpixels=source_subpixels,
            source_bounds=source_bounds)

    def set_illumination_from_plane(self, illum_func, distance, paraxial=True,
                                    angular_samples=1024, angular_limit=2.0,
                                    source_subpixels=1, source_bounds=None):
        """
        Illuminate the pupil with an incoherent emitting plane.

        Parameters
        ----------
        illum_func: function illum_func(x, y) giving the source's spatial radiance
            profile, multiplied by spectrum at each wavelength. x and y are
            positions in the source plane, in micrometres.
        distance: positive, finite distance from the source to the pupil, in micrometres.
        paraxial: True uses small-angle source geometry and the paraxial convolution;
            False uses direction cosines and the nonparaxial convolution.
        angular_samples, angular_limit, source_subpixels, source_bounds: sampling
            options described in set_angular_illumination. source_bounds are
            angular bounds, not source-plane positions in micrometres.

        Returns
        -------
        simulation: this simulator, so calls can be chained.

        Notes
        -----
        IncoherentLightSimulation maps the source to incident directions and adds
        its quadratic wavefront curvature at each wavelength. Use the physical
        lens focal length in optical_setup, followed by one propagation to the
        monitor. This replaces any previously configured source.
        """
        if distance is None:
            raise ValueError("Source distance must be finite and positive.")
        return self._set_illumination(illum_func, distance=distance,
            paraxial=paraxial, angular_samples=angular_samples,
            angular_limit=angular_limit, source_subpixels=source_subpixels,
            source_bounds=source_bounds)

    def _set_illumination(self, illum_func, distance=None, **options):
        """Check the source and optical setup, then replace the source configuration."""
        if illum_func is None and distance is None:
            self._illumination = None
            return self
        if not callable(illum_func):
            raise TypeError("illum_func must be callable.")
        if distance is not None:
            traced = False
            if backend_name == "jax":
                from jax.core import Tracer
                traced = isinstance(distance, Tracer)
            if not traced and (not np.isfinite(distance) or distance <= 0):
                raise ValueError("Source distance must be finite and positive.")
        propagations = [kind for kind, _ in self._recipe_steps
                        if kind != "optical_element"]
        if (len(propagations) != 1 or self._recipe_steps[-1][0] not in
                ("propagate", "zoom_propagate")):
            raise ValueError("Incoherent illumination requires pupil-plane optics "
                             "followed by one propagate or "
                             "zoom_propagate step.")
        # A missing distance means angular illumination; otherwise it is a plane source.
        self._illumination = (illum_func, distance, options)
        return self

    def _run_optical_setup(self, F, z=None):
        """Replay the recipe; z is supplied by the incoherent callback API."""
        for kind, args in self._recipe_steps:
            if kind == "optical_element":
                (element,) = args
                F = F.add(element)
            elif kind == "propagate":
                z, scale_factor, model = args
                F = F.propagate(z=z, scale_factor=scale_factor, model=model)
            elif kind == "zoom_propagate":
                z, x_interval, y_interval, model, sampling, padding, Nx_mon, Ny_mon = args
                F = F.zoom_propagate(z, x_interval, y_interval, model=model,
                                     sampling=sampling, padding=padding, Nx_mon=Nx_mon, Ny_mon=Ny_mon)
            elif kind == "propagate_to_image_plane":
                pupil, M, zi, z0, scale_factor = args
                F = F.propagate_to_image_plane(pupil, M, zi, z0, scale_factor=scale_factor)
            elif kind == "propagate_to_lens_focal_plane":
                focal_length, x_interval, y_interval = args
                F = F.propagate_to_lens_focal_plane(focal_length, x_interval, y_interval)
            else:
                raise ValueError(f"Unknown step kind: {kind}")
        return F

    def get_colors(self):
        """
        Replay the optical setup at each wavelength and combine the output colors.

        Returns
        -------
        rgb: sRGB array with shape (monitor_Ny, monitor_Nx, 3), using the final
            monitor grid of the recorded setup.

        Notes
        -----
        Angular or finite-plane illumination is combined incoherently at
        each wavelength. A finite source also supplies the wavelength-dependent
        incident curvature. With no source configured, illumination is coherent.
        """
        t0 = time.time()
        sRGB_linear = None
        incoherent = None
        if self._illumination is not None:
            source, source_distance, options = self._illumination
            incoherent = IncoherentLightSimulation(self.λ_list_samples[0]*nm,
                                                   *self._input_grid)
            if source_distance is None:
                incoherent.set_angular_illumination(source, **options)
            else:
                incoherent.set_illumination_from_plane(source, source_distance, **options)
            distance = self._recipe_steps[-1][1][0]
        if backend_name == "jax":
            from jax.core import Tracer

        bar = progressbar.ProgressBar()

        for i in bar(range(self.spectrum_divisions)):
            λ = self.λ_list_samples[i] * nm  # micrometres, static under JIT
            if incoherent is None:
                F = MonochromaticField(λ, *self._input_grid, intensity=1 * W / m**2)
                F = self._run_optical_setup(F)
                Iλ = bd.real(F.E * bd.conjugate(F.E))
                x, y, z = F.x, F.y, F.z
                dx, dy = F.dx, F.dy
                extent_x, extent_y = F.extent_x, F.extent_y
            else:
                # One coherent response and source convolution per wavelength.
                incoherent.λ = λ
                x, y, Iλ = incoherent.get_irradiance(self._run_optical_setup, distance)
                z = distance
                dx, dy = x[1]-x[0], y[1]-y[0]
                extent_x, extent_y = len(x)*dx, len(y)*dy

            # convert this wavelength slice to XYZ and accumulate
            XYZ = self.cs.spec_partition_to_XYZ(bd.outer(Iλ, self.spec_partitions[i]), i)
            if sRGB_linear is None:
                sRGB_linear = bd.zeros((3, Iλ.size))
            sRGB_linear += self.cs.XYZ_to_sRGB_linear(XYZ)
            # Keep traced coordinates local; plotting state must not leak tracers.
            if backend_name != "jax" or not isinstance(Iλ, Tracer):
                self.Ny, self.Nx = Iλ.shape
                self.extent_x, self.extent_y = extent_x, extent_y
                self.x, self.y, self.z = x, y, z
                self.dx, self.dy = dx, dy

        if backend_name == 'cupy':
            bd.cuda.Stream.null.synchronize()

        rgb = self.cs.sRGB_linear_to_sRGB(sRGB_linear)
        rgb = (rgb.T).reshape((*Iλ.shape, 3))
        print("Computation Took", time.time() - t0)
        return rgb

    from .visualization import plot_colors

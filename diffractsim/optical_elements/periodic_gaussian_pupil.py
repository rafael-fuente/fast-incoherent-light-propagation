"""Periodic phase pupil with a Gaussian illumination footprint."""
import numpy as np
from .optical_element import OpticalElement
from ..util import backend_functions


class PeriodicGaussianPupil(OpticalElement):
    """
    Repeat a phase cell under the Gaussian amplitude exp(-(x²+y²)/radius²).

    Use as the complete pupil in an incoherent finite-distance callback
    with model="ASM" or "Fresnel", sampling="angular" and angular illumination.
    Separate order intensities are added, assuming negligible lobe overlap.
    ASM uses a first-order finite-distance covariance correction; angular
    Fresnel uses the analytic Gaussian width of the chirped pupil.
    """

    def __init__(self, phase, period, radius, order_limit=1.12):
        """
        Initialize the periodic phase and its Gaussian illumination footprint.

        Parameters
        ----------
        phase: square two-dimensional phase cell, in radians, without a repeated endpoint.
        period: physical period of the phase cell along each axis, in micrometres.
        radius: Gaussian 1/e amplitude radius (1/e² irradiance radius), in micrometres.
        order_limit: direction-cosine extent used to select integer orders on
            each axis; includes orders up to ceil(order_limit*period/wavelength).
        """
        bd = backend_functions.backend
        self.phase = bd.asarray(phase)
        if self.phase.ndim != 2 or self.phase.shape[0] != self.phase.shape[1]:
            raise ValueError("phase must be one square periodic cell.")
        self.period, self.radius, self.order_limit = period, radius, order_limit

    def get_transmittance(self, x, y, wavelength):
        """
        Sample the repeated phase pixels and Gaussian pupil amplitude.

        Parameters
        ----------
        x: horizontal pupil coordinates, in micrometres.
        y: vertical pupil coordinates, in micrometres, broadcastable with x.
        wavelength: wavelength in micrometres; required by the optical-element API.
            The supplied phase is already in radians and is not rescaled.

        Returns
        -------
        transmission: complex Gaussian amplitude times the repeated phase factor,
            with the broadcast coordinate shape.
        """
        bd = backend_functions.backend
        n = self.phase.shape[0]
        ix = bd.floor((x/self.period+0.5)*n).astype(int) % n
        iy = bd.floor((y/self.period+0.5)*n).astype(int) % n
        return bd.exp(-(x*x+y*y)/self.radius**2+1j*self.phase[iy, ix])

    def get_angular_irradiance(self, wavelength, z, illumination,
                               samples, limit, source_subpixels=1, model="ASM"):
        """
        Combine diffraction-order powers with the incoherent source radiance.

        Parameters
        ----------
        wavelength: wavelength of the light, in micrometres.
        z: positive pupil-to-monitor distance, in micrometres.
        illumination: function (alpha, beta) returning angular radiance in W m^-2 sr^-1.
        samples: number of samples per angular axis.
        limit: angular grid half-width; the step 2*limit/samples must divide
            wavelength/period, and the grid must contain the selected orders.
        source_subpixels: positive number of source samples per cell axis;
            values > 1 average radiance within each cell.
        model: "ASM" (default) or "Fresnel"; selects the finite-distance
            Gaussian response without changing the nonparaxial monitor mapping.

        Returns
        -------
        a: one-dimensional direction-cosine axis, shared by alpha and beta.
        S: angular output of shape (samples, samples), in picowatts/sr; the simulator
            projects it to monitor irradiance with z²/rho⁴.

        Notes
        -----
        The source and Gaussian envelope are combined in Fourier space, so
        narrow individual diffraction lobes need no separate fine grid.
        Orders centered outside the unit disk retain their unchirped Fourier
        continuation. This approximation requires well-separated orders and
        a grid with room for the source and Gaussian tails.
        """
        if model not in ("ASM", "Fresnel"):
            raise ValueError('model must be "ASM" or "Fresnel".')
        bd = backend_functions.backend
        du, da = wavelength/self.period, 2*limit/samples
        spacing = int(round(du/da))
        if spacing < 1 or not np.isclose(spacing*da, du, rtol=1e-10):
            raise ValueError("The angular grid step must divide wavelength/period.")
        m = np.arange(-int(np.ceil(self.order_limit/du)),
                       int(np.ceil(self.order_limit/du))+1)
        index = samples//2 + m*spacing
        if index[0] < 0 or index[-1] >= samples:
            raise ValueError("The angular grid must include the selected orders.")
        a = (bd.arange(samples)-samples//2)*da
        offset = bd.fft.ifftshift(a)
        fx = bd.fft.rfftfreq(samples, da)[None, :]
        fy = bd.fft.fftfreq(samples, da)[:, None]

        # Source radiance is configured by the simulator's illumination setter.
        if source_subpixels == 1:
            source = illumination(-offset[None, :], -offset[:, None])
        else:
            source = bd.zeros((samples, samples))
            offsets = ((np.arange(source_subpixels)+0.5)/source_subpixels-0.5)*da
            for dx in offsets:
                for dy in offsets:
                    source += illumination(-offset[None, :]-dx,
                                           -offset[:, None]-dy)/source_subpixels**2
        source = bd.where(offset[None, :]**2+offset[:, None]**2 < 1, source, 0)
        area = bd.pi*self.radius**2/2
        sigma = wavelength/(2*bd.pi*self.radius)
        kernel = area*bd.fft.rfft2(source)*bd.exp(-2*bd.pi**2*sigma**2*(fx*fx+fy*fy))

        # Fourier-coefficient powers, including the physical pixel sinc factor.
        n = self.phase.shape[0]
        power = bd.abs(bd.fft.fft2(bd.exp(1j*self.phase)))**2/n**4
        k = bd.asarray(m % n)
        pixel = bd.sinc(bd.asarray(m)/n)**2
        power = power[k[:, None], k[None, :]]*pixel[:, None]*pixel[None, :]

        u, v = bd.meshgrid(bd.asarray(m)*du, bd.asarray(m)*du)
        r2 = u*u+v*v
        if model == "Fresnel":
            # The quadratic pupil chirp adds variance (radius/2z)^2 to each
            # Gaussian lobe. Apply its exact Fourier factor to propagating orders.
            weights = (bd.ones_like(u), r2 < 1)
            multipliers = (1, bd.expm1(-2*bd.pi**2*(self.radius/(2*z))**2
                                      *(fx*fx+fy*fy)))
        else:
            # ASM's direction-dependent covariance, to first order.
            scale = (self.radius/(2*z))**2*bd.maximum(1-r2, 0)
            weights = (bd.ones_like(u), scale*(1-(2-r2)*u*u),
                       -scale*(2-r2)*u*v, scale*(1-(2-r2)*v*v))
            multipliers = (1, -2*bd.pi**2*fx*fx, -4*bd.pi**2*fx*fy,
                           -2*bd.pi**2*fy*fy)
        spectrum = bd.zeros((samples, samples//2+1), dtype=complex)
        for weight, multiplier in zip(weights, multipliers):
            grid = bd.zeros((samples, samples))
            if backend_functions.backend_name == "jax":
                grid = grid.at[index[:, None], index[None, :]].set(power*weight)
            else:
                grid[index[:, None], index[None, :]] = power*weight
            spectrum += multiplier*bd.fft.rfft2(grid)
        return a, bd.fft.irfft2(kernel*spectrum, s=(samples, samples))

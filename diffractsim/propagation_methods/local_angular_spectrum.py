"""Angular-spectrum integration over one off-axis rectangular monitor.

The pupil consists of constant rectangular pixels. Their Fourier transforms
are exact sinc factors, and both Fourier sums are separable matrix products.
Only frequencies reaching this monitor are retained, with a tapered margin.
This is useful for separated, coarse intensity images at large angles.
"""
from functools import lru_cache
import numpy as np
from ..util import backend_functions


@lru_cache(maxsize=32)
def _matrices(wavelength, z, nx, ny, dx, dy, x0, x1, mx, y0, y1, my):
    """Fixed geometry only: the current pupil is supplied on every evaluation."""
    xc, yc = (x0+x1)/2, (y0+y1)/2
    c = z/np.sqrt(z*z+xc*xc+yc*yc)
    axes = []
    for n, pitch, lo, hi, count, centre in (
            (nx, dx, x0, x1, mx, xc), (ny, dy, y0, y1, my, yc)):
        radius, half = n*pitch/2, (hi-lo)/2
        # Same spectral guard and period as the paper's converged tile rule.
        inner = (half+radius)*c/(z*wavelength)
        width = 1.25*inner
        reach = width*z*wavelength/c**3+radius
        df = 1/(2*(half+reach+2*radius)*0.7)
        offset = np.linspace(-width, width, int(np.ceil(2*width/df))+1)
        freq = centre*c/(z*wavelength)+offset
        taper = np.where(abs(offset) <= inner, 1.,
            0.5*(1+np.cos(np.pi*(abs(offset)-inner)/(width-inner))))
        pupil_axis = (np.arange(n)-n//2)*pitch
        forward = (pitch*np.sinc(freq*pitch))[:, None]*np.exp(
            -2j*np.pi*np.outer(freq, pupil_axis))
        inverse = np.exp(2j*np.pi*np.outer(np.linspace(lo, hi, count)-centre, freq))
        axes.append((freq, taper, forward, inverse))
    fx, wx, Fx, Gx = axes[0]
    fy, wy, Fy, Gy = axes[1]
    longitudinal = wavelength**-2-fy[:, None]**2-fx[None, :]**2
    if np.any(longitudinal <= 0):
        raise ValueError('The local spectral patch reaches grazing angles; use sampling="angular".')
    transfer = np.exp(2j*np.pi*(z*np.sqrt(longitudinal)
        +xc*fx[None, :]+yc*fy[:, None]))
    transfer *= wy[:, None]*wx[None, :]*(fx[1]-fx[0])*(fy[1]-fy[0])
    return Fx, Fy, Gx, Gy, transfer


def local_angular_spectrum(field, z, x_interval, y_interval, nx, ny):
    """Complex field from a centred pupil on a fixed rectangular monitor.

    Pupil amplitudes/phases are differentiable. Geometry is prepared with NumPy;
    use the existing angular sampler when differentiating distances or grids.
    Refine the monitor pitch and check the spectral guard for a new geometry.
    """
    if hasattr(field, "pupil_quadrature"):
        raise ValueError("Local ASM uses constant physical pixels, not pupil quadrature.")
    bd = backend_functions.backend
    matrices = _matrices(float(field.λ), float(z), field.Nx, field.Ny,
        float(field.dx), float(field.dy), *map(float, x_interval), nx,
        *map(float, y_interval), ny)
    Fx, Fy, Gx, Gy, transfer = map(bd.asarray, matrices)
    spectrum = Fy @ field.E @ Fx.T
    return Gy @ (spectrum*transfer) @ Gx.T


def local_angular_response(field, z, alpha, beta, dx, dy, x_bounds, y_bounds):
    """Sample K = z² |U|²/c⁴ near one compact angular region."""
    from .angular_spectrum_response import sample_analytic_spectrum
    bd = backend_functions.backend
    c = np.sqrt(1-alpha*alpha-beta*beta)
    x, y = z*alpha/c, z*beta/c
    # Interpolate intensity, whose envelope is resolved without the carrier.
    axes = [np.arange(np.floor(lo/h)-3, np.ceil(hi/h)+4)*h
            for (lo, hi), h in ((x_bounds, dx), (y_bounds, dy))]
    xa, ya = axes
    U = local_angular_spectrum(field, z, (xa[0], xa[-1]), (ya[0], ya[-1]),
                               len(xa), len(ya))
    # Angular rectangles contain corners that no source-shifted detector uses.
    # Exclude them before sampling: they need neither a field nor a gradient.
    keep = np.flatnonzero((x >= xa[3]) & (x <= xa[-4])
                         & (y >= ya[3]) & (y <= ya[-4]))
    H = sample_analytic_spectrum(bd.abs(U)**2,
        bd.asarray((x.ravel()[keep]-xa[0])/dx), bd.asarray((y.ravel()[keep]-ya[0])/dy),
        0., 1., order=6)
    K = bd.zeros(alpha.size)
    values = H*bd.asarray(z*z/c.ravel()[keep]**4)
    if backend_functions.backend_name == "jax":
        K = K.at[keep].set(values, indices_are_sorted=True, unique_indices=True)
    else:
        K[keep] = values
    return K.reshape(alpha.shape)

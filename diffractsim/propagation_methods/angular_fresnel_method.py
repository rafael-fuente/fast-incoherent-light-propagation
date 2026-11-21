"""Angular Fresnel response: pupil chirp, Fourier transform and screen mapping.

The outgoing pupil chirp uses the paraxial approximation. The
direction-cosine mapping, spherical carrier and cos(theta)**4 irradiance
factor are retained. These geometric factors do not turn the chirp into
the full angular-spectrum propagator.

References
----------
D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
SPIE (2011), doi:10.1117/3.858456.
Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
diffraction using the Bluestein method," Light: Science & Applications 9,
119 (2020), doi:10.1038/s41377-020-00362-z.
R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
propagation of partially coherent light (2026), accompanying manuscript.
"""
import numpy as np
from ..util import backend_functions
from ..util.bluestein_FFT import bluestein_fft2


# Samples per Fourier fringe for local cubic interpolation. This controls the
# internal spectrum only; it does not change the requested Cartesian monitor.
_OVERSAMPLING = 8


def _traced(*values):
    if backend_functions.backend_name != "jax":
        return False
    import jax
    return any(isinstance(v, jax.core.Tracer) for v in values)


def _check_distance(z):
    if not _traced(z) and (not np.isfinite(float(z)) or float(z) <= 0):
        raise ValueError("Angular Fresnel propagation requires a positive finite distance.")


def _chirped_pupil(field, z):
    bd = backend_functions.backend
    return field.E*bd.exp(1j*bd.pi*(field.xx**2+field.yy**2)/(field.λ*z))


def _fourier_amplitude(field, values, a, b, center=(0, 0)):
    """Pupil transform / wavelength on uniform direction-cosine axes.
    
    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    bd = backend_functions.backend
    if hasattr(field, "pupil_quadrature"):
        A = field.pupil_quadrature.fourier_transform(values, a/field.λ, b/field.λ)/field.λ
        return A*bd.exp(2j*bd.pi*(center[0]*a[None, :]+center[1]*b[:, None])/field.λ)
    A = bluestein_fft2(values, a[0]/field.λ, a[-1]/field.λ, 1/field.dx,
                      b[0]/field.λ, b[-1]/field.λ, 1/field.dy, Nx=len(a), Ny=len(b))
    A *= field.dx*field.dy/field.λ
    A *= bd.exp(-2j*bd.pi*(field.x[0]-center[0])*a/field.λ)[None, :]
    A *= bd.exp(-2j*bd.pi*(field.y[0]-center[1])*b/field.λ)[:, None]
    return A


def _cubic(spectrum, u, v, periodic=False):
    """Tensor cubic Lagrange interpolation of complex amplitudes (16 samples).
    
    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    bd = backend_functions.backend
    def indices_weights(t, size):
        i = bd.floor(t).astype(int)
        if not periodic:
            i = bd.clip(i, 1, size-3)
        t = t-i
        weights = (-t*(t-1)*(t-2)/6, (t+1)*(t-1)*(t-2)/2,
                   -(t+1)*t*(t-2)/2, (t+1)*t*(t-1)/6)
        indices = tuple((i+j-1) % size if periodic else i+j-1 for j in range(4))
        return indices, weights
    ix, wx = indices_weights(u, spectrum.shape[1])
    iy, wy = indices_weights(v, spectrum.shape[0])
    result = bd.zeros_like(u, dtype=spectrum.dtype)
    for j in range(4):
        row = sum(wx[i]*spectrum[iy[j], ix[i]] for i in range(4))
        result = result+wy[j]*row
    return result


def _sample_amplitude(field, values, alpha, beta, bounds=None):
    """Sample the smooth transform; restore any input-grid translation exactly.
    
    A small Bluestein window is used when its size is known. For traced input
    grids (e.g. repeated zooms under JIT), an oversampled periodic FFT keeps
    shapes fixed and also supports derivatives of the grid geometry.
    
    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    bd = backend_functions.backend
    alpha, beta = bd.broadcast_arrays(bd.asarray(alpha), bd.asarray(beta))
    center = (field.x[field.Nx//2], field.y[field.Ny//2])
    quadrature = hasattr(field, "pupil_quadrature")
    if bounds is None and not _traced(alpha, beta):
        bounds = ((float(alpha.min()), float(alpha.max())),
                  (float(beta.min()), float(beta.max())))
    if bounds is None and quadrature:
        # Eq. (24) adds two propagating directions; leave an interpolation margin.
        bounds = ((-2., 2.), (-2., 2.))
    axes = None
    if bounds is not None and not _traced(field.extent_x, field.extent_y, field.λ):
        steps = (float(field.λ)/(_OVERSAMPLING*float(field.extent_x)),
                 float(field.λ)/(_OVERSAMPLING*float(field.extent_y)))
        sizes = tuple(int(np.ceil((hi-lo)/h))+5 for (lo, hi), h in zip(bounds, steps))
        if quadrature or (sizes[0] <= _OVERSAMPLING*field.Nx and sizes[1] <= _OVERSAMPLING*field.Ny):
            axes = tuple(lo-2*h+bd.arange(n)*h for (lo, _), h, n in zip(bounds, steps, sizes))
    if axes is not None:
        a, b = axes
        A = _fourier_amplitude(field, values, a, b, center)
        result = _cubic(A, (alpha-a[0])/(a[1]-a[0]), (beta-b[0])/(b[1]-b[0]))
    else:
        if quadrature:
            raise ValueError("Keep physical-pixel geometry fixed under JAX jit.")
        nx, ny = _OVERSAMPLING*field.Nx, _OVERSAMPLING*field.Ny
        # The centered discrete pupil transform is periodic in frequency.
        # Wrapping its interpolation coordinates does not assume periodic light.
        A = bd.fft.fft2(values, s=(ny, nx))*field.dx*field.dy/field.λ
        A *= bd.exp(2j*bd.pi*(field.Nx//2)*bd.arange(nx)/nx)[None, :]
        A *= bd.exp(2j*bd.pi*(field.Ny//2)*bd.arange(ny)/ny)[:, None]
        u = bd.mod(alpha*field.dx/field.λ, 1)*nx
        v = bd.mod(beta*field.dy/field.λ, 1)*ny
        result = _cubic(A, u, v, periodic=True)
    return result*bd.exp(-2j*bd.pi*(center[0]*alpha+center[1]*beta)/field.λ)


def angular_fresnel_method(field, z, x_interval, y_interval, Nx_mon=None, Ny_mon=None):
    """Evaluate angular Fresnel propagation on a rectangular monitor.
    
    Parameters
    ----------
    field: input MonochromaticField, including its pupil samples and wavelength.
    z: positive propagation distance, in micrometres.
    x_interval: first and last horizontal monitor coordinates, in micrometres.
    y_interval: first and last vertical monitor coordinates, in micrometres.
    Nx_mon: horizontal monitor count, at least 2; None uses field.Nx.
    Ny_mon: vertical monitor count, at least 2; None uses field.Ny.
    
    Returns
    -------
    x, y: endpoint-inclusive horizontal and vertical monitor axes, in micrometres.
    E: complex propagated field with shape (len(y), len(x)).
    
    Notes
    -----
    Let E_0(ξ, η) be the complex input field, including the pupil
    transmission and any earlier optics. It is zero outside the input pupil.
    With wavelength λ and positive distance z, the propagation model is

        E_AF(x, y; z) =
            z / (i*λ*(z^2 + x^2 + y^2))
            * exp(2*π*i*sqrt(z^2 + x^2 + y^2)/λ)
            * integral(ξ=-inf..inf) integral(η=-inf..inf) [
                E_0(ξ, η)
                * exp(i*π*(ξ^2 + η^2)/(λ*z))
                * exp(-2*π*i*(x*ξ + y*η)
                      /(λ*sqrt(z^2 + x^2 + y^2)))
              ] dη dξ
    
    Here (ξ, η) are coordinates in the input plane and (x, y) are
    coordinates on the monitor. All lengths use the same units. The first
    exponential is the spherical phase factor; the second is the quadratic
    Fresnel phase in the pupil; the third evaluates its Fourier transform
    at direction cosines x/sqrt(z^2+x^2+y^2) and y/sqrt(z^2+x^2+y^2).

    Numerically, FFT/Bluestein transforms (or the supplied pupil quadrature)
    evaluate the pupil transform. Cubic interpolation samples it at the
    required directions. The factors outside the integral are then applied
    directly. The monitor must resolve phase before further propagation.

    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    bd = backend_functions.backend
    _check_distance(z)
    nx = field.Nx if Nx_mon is None else Nx_mon
    ny = field.Ny if Ny_mon is None else Ny_mon
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2
           for n in (nx, ny, field.Nx, field.Ny)):
        raise ValueError("Input and monitor counts must be integers >= 2.")
    for interval in (x_interval, y_interval):
        if len(interval) != 2:
            raise ValueError("Use two endpoints for each output interval.")
        if not _traced(*interval) and (not np.isfinite(interval).all() or interval[1] <= interval[0]):
            raise ValueError("Output intervals must be finite and increasing.")
    bounds = ((-1., 1.), (-1., 1.))
    if not _traced(z, *x_interval, *y_interval):
        # Extrema occur at the edges or at the closest point to either axis.
        xx, yy = np.meshgrid([*x_interval, np.clip(0, *x_interval)],
                             [*y_interval, np.clip(0, *y_interval)])
        rho = np.sqrt(float(z)**2+xx*xx+yy*yy)
        bounds = ((float((xx/rho).min()), float((xx/rho).max())),
                  (float((yy/rho).min()), float((yy/rho).max())))
    x, y = bd.linspace(*x_interval, nx), bd.linspace(*y_interval, ny)
    xx, yy = bd.meshgrid(x, y)
    rho = bd.sqrt(z*z+xx*xx+yy*yy)
    A = _sample_amplitude(field, _chirped_pupil(field, z), xx/rho, yy/rho, bounds)
    E = z/(1j*rho**2)*bd.exp(2j*bd.pi*rho/field.λ)*A
    return x, y, E


def angular_fresnel_response(field, z, samples=1024, limit=2.0):
    """Compute the angular Fresnel intensity kernel on a uniform angular grid.
    
    Parameters
    ----------
    field: input MonochromaticField.
    z: positive pupil-to-monitor distance, in micrometres.
    samples: number of samples per angular axis, integer >= 8.
    limit: angular grid half-width, at least 2.
    
    Returns
    -------
    a: one-dimensional direction-cosine axis, shared by alpha and beta.
    K: weighted coherent intensity of shape (samples, samples).
    
    Notes
    -----
    Inside the unit disk, K = |FT(E*chirp)/wavelength|². Outside it, retain
    the unchirped pupil Fourier intensity for the shifted-source convolution.
    The uniform angular grid is evaluated directly without interpolation.
    
    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    _check_distance(z)
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 8:
        raise ValueError("Use integer angular_samples >= 8.")
    if not np.isfinite(limit) or limit < 2:
        raise ValueError("Use angular_limit >= 2.")
    bd = backend_functions.backend
    a = (bd.arange(samples)-samples//2)*(2*limit/samples)
    A = _fourier_amplitude(field, _chirped_pupil(field, z), a, a)
    B = _fourier_amplitude(field, field.E, a, a)
    inside = a[None, :]**2+a[:, None]**2 < 1
    return a, bd.where(inside, bd.abs(A)**2, bd.abs(B)**2)


def angular_fresnel_response_at(field, z, alpha, beta):
    """Evaluate the angular Fresnel intensity kernel at paired directions.
    
    Parameters
    ----------
    field: input MonochromaticField.
    z: positive pupil-to-monitor distance, in micrometres.
    alpha: horizontal direction cosines, possibly shifted by source directions.
    beta: vertical direction cosines, broadcastable with alpha.
    
    Returns
    -------
    K: weighted coherent intensity with the broadcast coordinate shape,
        using the normalization and continuation of angular_fresnel_response.
    
    References
    ----------
    R. de la Fuente Herrezuelo, Efficient convolution-based nonparaxial
    propagation of partially coherent light (2026), accompanying manuscript.
    """
    _check_distance(z)
    bd = backend_functions.backend
    # Compact monitors supply fixed Cartesian angular grids. Transform these
    # directly instead of interpolating a larger, oversampled pupil spectrum.
    if isinstance(alpha, np.ndarray) and isinstance(beta, np.ndarray):
        u, v = np.broadcast_arrays(alpha, beta)
        a, b = np.unique(u), np.unique(v)
        rectangular = min(len(a), len(b)) >= 2 and u.size == len(a)*len(b)
        if rectangular:
            rectangular = (np.array_equal(u.reshape(len(b), len(a)), np.broadcast_to(a, (len(b), len(a))))
                           and np.array_equal(v.reshape(len(b), len(a)), np.broadcast_to(b[:, None], (len(b), len(a))))
                           and np.allclose(np.diff(a), a[1]-a[0], rtol=1e-9, atol=1e-15)
                           and np.allclose(np.diff(b), b[1]-b[0], rtol=1e-9, atol=1e-15))
        if rectangular:
            A = _fourier_amplitude(field, _chirped_pupil(field, z), bd.asarray(a), bd.asarray(b))
            inside = a[None, :]**2+b[:, None]**2 < 1
            K = bd.abs(A)**2
            if not np.all(inside):
                B = _fourier_amplitude(field, field.E, bd.asarray(a), bd.asarray(b))
                K = bd.where(bd.asarray(inside), K, bd.abs(B)**2)
            return K.reshape(u.shape)
    alpha, beta = bd.broadcast_arrays(bd.asarray(alpha), bd.asarray(beta))
    A = _sample_amplitude(field, _chirped_pupil(field, z), alpha, beta)
    inside = alpha*alpha+beta*beta < 1
    if not _traced(inside) and bool(bd.all(inside)):
        return bd.abs(A)**2
    B = _sample_amplitude(field, field.E, alpha, beta)
    return bd.where(inside, bd.abs(A)**2, bd.abs(B)**2)

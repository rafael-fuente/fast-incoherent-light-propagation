"""Convolve an extended source with the response near each monitor region.

Source bounds let us skip directions that cannot contribute to the monitor.
We integrate the source once, then reuse that small kernel for each region.
Interpolation allows source edges to be sampled more finely than the response.
"""
from functools import lru_cache
import numpy as np
from scipy.fft import next_fast_len
from . import backend_functions
from .constants import m


def lagrange_weights(t, order=6):
    """
    Construct polynomial interpolation weights on a uniform one-dimensional grid.

    Parameters
    ----------
    t: one-dimensional NumPy array of positions measured in grid steps.
    order: number of neighboring samples in the interpolation stencil, default 6.

    Returns
    -------
    indices: integer grid indices with shape (len(t), order).
    weights: interpolation weights with the same shape as indices.
    """
    first = np.floor(t).astype(int)-order//2+1
    indices = first[:, None]+np.arange(order)
    weights = np.ones_like(indices, dtype=float)
    for j in range(order):
        for k in range(order):
            if j != k:
                weights[:, j] *= (t-first-k)/(j-k)
    return indices, weights


@lru_cache(maxsize=8)
def source_rule(bounds, step, subpixels):
    """
    Build and cache integration geometry for a compact angular source.

    Parameters
    ----------
    bounds: tuple (alpha_min, alpha_max, beta_min, beta_max) enclosing the source.
    step: spacing of the coarse direction-cosine grid.
    subpixels: positive number of fine source samples per coarse cell axis.

    Returns
    -------
    a, b: coarse horizontal and vertical angular axes, including stencil margins.
    u, v: fine horizontal and vertical source quadrature axes.
    wx, wy: matrices depositing the fine samples on the coarse axes;
        each includes its one-dimensional integration step.

    Notes
    -----
    Only geometry is cached; no source radiance or pupil values are cached.
    """
    axes, fine_axes, matrices = [], [], []
    for low, high in (bounds[:2], bounds[2:]):
        # Fine cells cover the declared support; radiance is zero outside it.
        lo, hi = int(np.floor(low/step)), int(np.ceil(high/step))
        coarse = np.arange(lo-3, hi+4)*step
        fine = (lo+(np.arange((hi-lo)*subpixels)+0.5)/subpixels)*step
        indices, weights = lagrange_weights((fine-coarse[0])/step)
        matrix = np.zeros((len(coarse), len(fine)))
        matrix[indices, np.arange(len(fine))[:, None]] = weights*step/subpixels
        axes.append(coarse)
        fine_axes.append(fine)
        matrices.append(matrix)
    return (*axes, *fine_axes, *matrices)


def source_kernel(illumination, bounds, step, subpixels):
    """
    Integrate source radiance into a compact angular convolution kernel.

    Parameters
    ----------
    illumination: function (alpha, beta) returning radiance in W m^-2 sr^-1.
    bounds: (alpha_min, alpha_max, beta_min, beta_max); source is zero outside.
    step: coarse direction-cosine grid spacing.
    subpixels: positive number of integration samples per cell axis.

    Returns
    -------
    a, b: horizontal and vertical axes of the reversed source kernel.
    mass: integrated source weights of shape (len(b), len(a)), including
        d(alpha)*d(beta); summing them gives incident irradiance in W/m².

    Notes
    -----
    Polynomial deposition preserves subcell source positions. Its weights
    may contain small negative entries even for nonnegative radiance.
    """
    bd = backend_functions.backend
    a, b, u, v, wx, wy = source_rule(tuple(bounds), step, subpixels)
    uu, vv = bd.meshgrid(bd.asarray(u), bd.asarray(v))
    L = bd.where(uu*uu+vv*vv < 1, illumination(uu, vv), 0)
    mass = bd.asarray(wy) @ L @ bd.asarray(wx).T
    # Eq. (24) uses K(q+p), so reverse the source to write a convolution.
    return -a[::-1], -b[::-1], mass[::-1, ::-1]


@lru_cache(maxsize=32)
def _pixel_matrices(wavelength, z, nx, ny, dx, dy, x0, x1, mx, y0, y1, my):
    """Separable ASM integrals for constant pixels and one fixed image tile."""
    xc, yc = (x0+x1)/2, (y0+y1)/2
    c = z/np.sqrt(z*z+xc*xc+yc*yc)
    axes = []
    for n, pitch, lo, hi, count, centre in (
            (nx, dx, x0, x1, mx, xc), (ny, dy, y0, y1, my, yc)):
        radius, half = n*pitch/2, (hi-lo)/2
        # The paper's converged spectral margin and Fourier integration period.
        inner = (half+radius)*c/(z*wavelength)
        width = 1.25*inner
        reach = width*z*wavelength/c**3+radius
        df = 1/(2*(half+reach+2*radius)*0.7)
        offset = np.linspace(-width, width, int(np.ceil(2*width/df))+1)
        frequency = centre*c/(z*wavelength)+offset
        taper = np.where(abs(offset) <= inner, 1.,
            .5*(1+np.cos(np.pi*(abs(offset)-inner)/(width-inner))))
        # Each rectangular pixel contributes its exact sinc Fourier transform.
        forward = (pitch*np.sinc(frequency*pitch))[:, None]*np.exp(
            -2j*np.pi*np.outer(frequency, (np.arange(n)-n//2)*pitch))
        inverse = np.exp(2j*np.pi*np.outer(np.linspace(lo, hi, count)-centre, frequency))
        axes.append((frequency, taper, forward, inverse))
    fx, wx, Fx, Gx = axes[0]
    fy, wy, Fy, Gy = axes[1]
    longitudinal = wavelength**-2-fy[:, None]**2-fx[None, :]**2
    if np.any(longitudinal <= 0):
        return None  # Keep the existing contour rule if the patch reaches grazing angles.
    transfer = np.exp(2j*np.pi*(z*np.sqrt(longitudinal)+xc*fx[None, :]+yc*fy[:, None]))
    transfer *= wy[:, None]*wx[None, :]*(fx[1]-fx[0])*(fy[1]-fy[0])
    return Fx, Fy, Gx, Gy, transfer


def _pixel_response(pupil, z, a, b, dx, dy, x_bounds, y_bounds):
    """Evaluate the current pupil, then map its tile intensity to K=z²|U|²/c⁴."""
    from ..propagation_methods.direct_quadrature_angular_spectrum_method import sample_analytic_spectrum
    bd = backend_functions.backend
    xa, ya = [np.arange(np.floor(lo/h)-3, np.ceil(hi/h)+4)*h
              for (lo, hi), h in ((x_bounds, dx), (y_bounds, dy))]
    matrices = _pixel_matrices(
        float(pupil.λ), float(z), pupil.Nx, pupil.Ny, float(pupil.dx), float(pupil.dy),
        xa[0], xa[-1], len(xa), ya[0], ya[-1], len(ya))
    if matrices is None:
        return None
    Fx, Fy, Gx, Gy, transfer = map(bd.asarray, matrices)
    U = Gy @ ((Fy @ pupil.E @ Fx.T)*transfer) @ Gx.T
    c = np.sqrt(1-a*a-b*b)
    x, y = z*a/c, z*b/c
    # The angular rectangle includes unused corners outside the physical tile.
    keep = np.flatnonzero((x >= xa[3]) & (x <= xa[-4]) & (y >= ya[3]) & (y <= ya[-4]))
    H = sample_analytic_spectrum(bd.abs(U)**2,
        bd.asarray((x.ravel()[keep]-xa[0])/dx), bd.asarray((y.ravel()[keep]-ya[0])/dy),
        0., 1., order=6)*bd.asarray(z*z/c.ravel()[keep]**4)
    K = bd.zeros(a.size)
    if backend_functions.backend_name == "jax":
        K = K.at[keep].set(H, indices_are_sorted=True, unique_indices=True)
    else:
        K[keep] = H
    return K


def _fft_response(response, z, alpha, beta):
    """Extend the coherent FFT monitor to cover the source convolution.

    ``alpha`` and ``beta`` are fixed NumPy arrays of the required direction
    cosines, including shifts from the source. Add monitor samples at the
    original spacing and padding, then return K = z² |U|² / c⁴ at those
    directions, where c = sqrt(1-alpha²-beta²).

    The extension preserves the selected ASM bandwidth. Geometry stays fixed,
    while gradients can still pass through the pupil values.
    """
    from ..propagation_methods.coherent_response import DeferredPropagation
    c = np.sqrt(1-alpha*alpha-beta*beta)
    screen = (z*alpha/c, z*beta/c)
    axes = response.screen_axes(host=True)
    intervals, counts = [], []
    for axis, points in zip(axes, screen):
        pitch = (axis[-1]-axis[0])/(len(axis)-1)
        lo = min(0, int(np.floor((points.min()-axis[0])/pitch))-1)
        hi = max(len(axis)-1, int(np.ceil((points.max()-axis[0])/pitch))+1)
        intervals.append([float(axis[0]+lo*pitch), float(axis[0]+hi*pitch)])
        counts.append(hi-lo+1)
    expanded = DeferredPropagation(response.pupil).zoom_propagate(z,
        intervals[0], intervals[1], Nx_mon=counts[0], Ny_mon=counts[1],
        model=response.model, sampling="fft", padding=response.options["padding"])
    return expanded.angular_propagate(z, alpha=alpha, beta=beta)


def _required_response(ka, kb, ja, jb, alpha, beta, step):
    """Mark the response samples that can affect the requested monitor points.

    Each point uses six interpolation samples per axis after convolution.
    Enlarge that stencil by the full source kernel to include every response
    sample that contributes. A cumulative sum combines these rectangles
    without looping over their individual pixels.
    """
    ix = np.floor((alpha-ka[0]-ja[0])/step).astype(int).ravel()
    iy = np.floor((beta-kb[0]-jb[0])/step).astype(int).ravel()
    x0, x1 = np.clip(ix-2-(len(ja)-1), 0, len(ka)), np.clip(ix+4, 0, len(ka))
    y0, y1 = np.clip(iy-2-(len(jb)-1), 0, len(kb)), np.clip(iy+4, 0, len(kb))
    edges = np.zeros((len(kb)+1, len(ka)+1), dtype=np.int32)
    for rows, cols, sign in ((y0, x0, 1), (y1, x0, -1), (y0, x1, -1), (y1, x1, 1)):
        np.add.at(edges, (rows, cols), sign)
    return edges.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)[:-1, :-1] > 0


def compact_irradiance(response, z, x, y, illumination, bounds, step, subpixels,
                       tiled_asm=False):
    """
    Compute nonparaxial irradiance using only the angular regions that matter.

    Build one source kernel, convolve it with the coherent response around each
    monitor region, then interpolate the result onto the requested points.
    A rectangular monitor is passed as one row of paired coordinates.

    Parameters
    ----------
    response: DeferredPropagation recording the final coherent propagation.
    z: positive distance from the final pupil to the monitor, in micrometres.
    x, y: two-dimensional arrays of paired monitor coordinates, in micrometres.
        Each row describes one region; the arrays must broadcast together.
    illumination: function (alpha, beta) returning radiance in W m^-2 sr^-1.
    bounds: angular source support (alpha_min, alpha_max, beta_min, beta_max).
    step: direction-cosine spacing shared by the response and source kernel.
    subpixels: number of source samples per cell along each axis.
        Larger values resolve source edges without refining the response grid.
    tiled_asm: evaluate the ASM response in blocks for one rectangular monitor.
        The caller enables this when the full spectrum would be too large.

    Returns
    -------
    E: irradiance in W/m², with the broadcast shape of x and y.

    Notes
    -----
    Source bounds and monitor geometry must stay fixed. Pupil and source
    values remain differentiable. With tiled_asm=True, the angular rectangle
    including all source shifts must lie inside alpha² + beta² < 1.
    """
    from ..propagation_methods.direct_quadrature_angular_spectrum_method import sample_analytic_spectrum
    bd = backend_functions.backend
    to_numpy = bd.asnumpy if backend_functions.backend_name == "cupy" else np.asarray
    x, y = np.broadcast_arrays(to_numpy(x), to_numpy(y))
    rho = np.sqrt(z*z+x*x+y*y)
    alpha, beta = x/rho, y/rho
    ja, jb, J = source_kernel(illumination, bounds, step, subpixels)
    regions = []
    for a, b in zip(alpha, beta):
        # Leave an interpolation stencil beyond every source-shifted point.
        ka = np.arange(np.floor((a.min()-ja[-1])/step)-4,
                       np.ceil((a.max()-ja[0])/step)+5)*step
        kb = np.arange(np.floor((b.min()-jb[-1])/step)-4,
                       np.ceil((b.max()-jb[0])/step)+5)*step
        regions.append((ka, kb))
    if not tiled_asm:
        directions = [np.meshgrid(a, b) for a, b in regions]
        u = np.concatenate([a.ravel() for a, b in directions])
        v = np.concatenate([b.ravel() for a, b in directions])
    # Uniform rectangular detector tiles can use separable pixel integrals.
    # Sparse detectors and physical-pixel quadrature retain the existing rule.
    axes = [np.unique(x[0]), np.unique(y[0])]
    rectangular = min(map(len, axes)) > 1 and np.prod([len(q) for q in axes]) == x.shape[1]
    if rectangular:
        dx, dy = [(q[-1]-q[0])/(len(q)-1) for q in axes]
        rectangular = all(np.allclose(np.diff(q), h, rtol=1e-9, atol=1e-15*m)
                          for q, h in zip(axes, (dx, dy)))
        rectangular &= all(np.allclose(q-q[:, :1], q[0]-q[0, 0], rtol=1e-9, atol=1e-15*m)
                           for q in (x, y))
    angular_asm = response.sampling == "angular" and response.model == "ASM"
    pupil = response.pupil
    pixel_grid = np.isscalar(pupil.z) and pupil.z == 0 and not hasattr(pupil, "pupil_quadrature")
    K = None
    if tiled_asm:
        from ..propagation_methods.zoom_angular_spectrum_method import compact_angular_spectrum_response
        required = _required_response(*regions[0], ja, jb, alpha[0], beta[0], step)
        K = compact_angular_spectrum_response(pupil, z, *regions[0], required=required).ravel()
    if K is None and angular_asm and rectangular and pixel_grid and np.all(u*u+v*v < 1):
        kernels = []
        sa, sb = np.meshgrid([-ja[-1]-3*step, -ja[0]+3*step],
                             [-jb[-1]-3*step, -jb[0]+3*step])
        for row, (a, b) in enumerate(directions):
            qa, qb = alpha[row, :, None]+sa.ravel(), beta[row, :, None]+sb.ravel()
            c = np.sqrt(1-qa*qa-qb*qb)
            xs, ys = z*qa/c, z*qb/c
            kernel = _pixel_response(pupil, z, a, b, dx, dy, (xs.min(), xs.max()), (ys.min(), ys.max()))
            if kernel is None:
                break
            kernels.append(kernel)
        else:
            K = bd.concatenate(kernels)
    fft_zoom = response.sampling == "fft" and response.method == "zoom_propagate"
    if K is None and fft_zoom and np.all(u*u+v*v < 1):
        K = _fft_response(response, z, u, v)
    elif K is None and angular_asm:
        # These are regular convolution grids: use the same contour rule as
        # the full-grid ASM kernel, while omitting its unused directions.
        from ..propagation_methods.direct_quadrature_angular_spectrum_method import _angular_spectrum_amplitude_at
        K = bd.abs(_angular_spectrum_amplitude_at(response.pupil, z, u, v,
                                                 dense=True, fixed_grid=True))**2
    elif K is None:
        K = response.angular_propagate(z, alpha=u, beta=v)
    values, start = [], 0
    for row, (ka, kb) in enumerate(regions):
        size = len(ka)*len(kb)
        local = K[start:start+size].reshape(len(kb), len(ka))
        shape = tuple(next_fast_len(n, real=True) for n in
                      (len(kb)+len(jb)-1, len(ka)+len(ja)-1))
        S = bd.fft.irfft2(bd.fft.rfft2(local, s=shape)*bd.fft.rfft2(J, s=shape), s=shape)
        # The two origins can differ; the interpolation helper uses one origin.
        a0, b0 = ka[0]+ja[0], kb[0]+jb[0]
        values.append(sample_analytic_spectrum(S, bd.asarray(alpha[row]-a0),
            bd.asarray(beta[row]-b0), 0., step, order=6))
        start += size
    return bd.maximum(bd.stack(values), 0)*bd.asarray(z*z/rho**4)

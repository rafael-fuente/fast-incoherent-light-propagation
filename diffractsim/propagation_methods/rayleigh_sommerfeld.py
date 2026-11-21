"""Evaluate the source-resolved RS integral with interpolated kernel factors.

Removing the incident and outgoing plane-wave tilts leaves smooth functions
of source and monitor position. We interpolate their intensity response, as
in the paper's image reference. Fourier transforms do the pupil integrals;
zero-padded convolutions add the independent source intensities.
"""
from contextlib import nullcontext
import numpy as np
from scipy.fft import next_fast_len
from ..util import backend_functions
from ..util.bluestein_FFT import bluestein_fft
from ..util.compact_source import source_kernel, lagrange_weights


def _nodes(low, high, order):
    """Place interpolation nodes across a fixed interval, including its ends."""
    if low == high:
        return np.array([low])
    return (low+high)/2+(high-low)/2*np.cos(np.pi*np.arange(order)/(order-1))


def _weight(nodes, index, value):
    """Return one Lagrange weight; a constant interval has weight one."""
    bd = backend_functions.backend
    result = bd.ones_like(value)
    for j in range(len(nodes)):
        denominator = bd.where(index == j, 1., nodes[index]-nodes[j])
        result *= bd.where(index == j, 1., (value-nodes[j])/denominator)
    return result


def _residual(field, distance, tx, ty):
    """Return the full RS factor after removing its plane-wave tilt.

    tx and ty are transverse slopes. The removed constant phase cancels in
    intensity. Rationalizing the remaining path difference avoids subtracting
    two large distances, which is important for metre-scale monitors.
    """
    bd = backend_functions.backend
    c = 1/bd.sqrt(1+tx*tx+ty*ty)
    centre_distance = distance/c
    dot = c*(tx*field.xx+ty*field.yy)
    r2 = field.xx**2+field.yy**2
    R = centre_distance*bd.sqrt(1-2*dot/centre_distance+r2/centre_distance**2)
    phase = (r2-dot*dot)/(R+centre_distance-dot)
    k = 2*bd.pi/field.λ
    return (centre_distance/R)**2*(1+1j/(k*R))*bd.exp(1j*k*phase)


def _support(illumination, step, subpixels, bounds):
    """Find a finite source window without caching any radiance values.

    Explicit bounds keep the geometry fixed when source parameters are traced.
    Otherwise inspect all physical source cells, including their subpixels.
    No nonzero radiance is discarded by a relative-intensity threshold.
    """
    if bounds is not None:
        bounds = tuple(bounds)
        if (len(bounds) != 4 or not np.isfinite(bounds).all() or
                bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
            raise ValueError("source_bounds must contain four finite, increasing angular limits.")
        return bounds
    bd = backend_functions.backend
    if backend_functions.backend_name == 'jax':
        import jax
        context = jax.ensure_compile_time_eval()
    else:
        context = nullcontext()
    with context:
        axis = np.arange(-int(np.ceil(1/step)), int(np.ceil(1/step))+1)*step
        a, b = bd.meshgrid(bd.asarray(axis), bd.asarray(axis))
        active = bd.zeros(a.shape, dtype=bool)
        offsets = ((np.arange(subpixels)+.5)/subpixels-.5)*step
        for du in offsets:
            for dv in offsets:
                u, v = a+du, b+dv
                active |= (u*u+v*v < 1) & (illumination(u, v) != 0)
        if backend_functions.backend_name == 'jax' and isinstance(active, jax.core.Tracer):
            raise ValueError("Pass source_bounds when differentiating source parameters.")
        host = bd.asnumpy(active) if backend_functions.backend_name == 'cupy' else np.asarray(active)
    rows, cols = np.nonzero(host)
    if not len(rows):
        return None
    return (axis[cols.min()]-step/2, axis[cols.max()]+step/2,
            axis[rows.min()]-step/2, axis[rows.max()]+step/2)


def rayleigh_sommerfeld_irradiance(field, z, x, y, illumination, z1,
        angular_step, source_subpixels, source_bounds, source_order, monitor_order):
    """Integrate the two-leg RS intensity on fixed monitor coordinates.

    The pupil field contains only the optical transmission. For finite z1,
    each interpolation node retains the exact incident spherical amplitude,
    phase and obliquity factor. An angular source instead uses plane waves.
    Every node also uses the exact outgoing RS kernel, including 1/(k R).

    Interpolation is numerical, not a common-curvature or shifted-response
    approximation. Increase source_order and monitor_order independently to
    check convergence. The angular step and source subpixels control the
    source integral. Pupil values and explicit-bound source values remain
    differentiable; geometry and interpolation orders must remain fixed.
    """
    bd = backend_functions.backend
    is_jax = backend_functions.backend_name == 'jax'
    xx, yy = np.meshgrid(x, y) if x.ndim == y.ndim == 1 else np.broadcast_arrays(x, y)
    shape = xx.shape
    tx, ty = xx.ravel()/z, yy.ravel()/z
    c = 1/np.sqrt(1+tx*tx+ty*ty)
    alpha, beta = tx*c, ty*c
    bounds = _support(illumination, angular_step, source_subpixels, source_bounds)
    if bounds is None:
        return bd.zeros(shape)

    # Interpolate in physical source slopes, always inside the source disk.
    if np.isfinite(z1):
        amax = max(abs(bounds[0]), abs(bounds[1]))
        bmax = max(abs(bounds[2]), abs(bounds[3]))
        if amax*amax+bmax*bmax >= 1:
            raise ValueError("For a finite-plane RS reference, source_bounds must fit inside the unit disk.")
        cmin = np.sqrt(1-amax*amax-bmax*bmax)
        snx = _nodes(-amax/cmin, amax/cmin, source_order)
        sny = _nodes(-bmax/cmin, bmax/cmin, source_order)
    else:
        snx = sny = np.array([0.])
    mnx, mny = _nodes(tx.min(), tx.max(), monitor_order), _nodes(ty.min(), ty.max(), monitor_order)

    # The source kernel includes its angular integration area and is reversed
    # so that an ordinary linear convolution evaluates H(q + p).
    ja, jb, _ = source_kernel(illumination, bounds, angular_step, source_subpixels)
    def response_axis(q, source_axis):
        lo = int(np.floor((q.min()-source_axis[-1])/angular_step))-4
        hi = int(np.ceil((q.max()-source_axis[0])/angular_step))+4
        return np.arange(lo, hi+1)*angular_step
    a, b = response_axis(alpha, ja), response_axis(beta, jb)
    fft_shape = (next_fast_len(len(b)+len(jb)-1), next_fast_len(len(a)+len(ja)-1))
    ix, wx = lagrange_weights((alpha-a[0]-ja[0])/angular_step)
    iy, wy = lagrange_weights((beta-b[0]-jb[0])/angular_step)
    ix, iy, wx, wy = map(bd.asarray, (ix, iy, wx, wy))
    mx = bd.stack([_weight(bd.asarray(mnx), i, bd.asarray(tx)) for i in range(len(mnx))])
    my = bd.stack([_weight(bd.asarray(mny), i, bd.asarray(ty)) for i in range(len(mny))])
    a, b, snx, sny, mnx, mny = map(bd.asarray, (a, b, snx, sny, mnx, mny))

    def spectrum_intensity(values):
        if hasattr(field, 'pupil_quadrature'):
            A = field.pupil_quadrature.fourier_transform(values, a/field.λ, b/field.λ)/field.λ
        else:
            A = bluestein_fft(values, 1, a[0]/field.λ, a[-1]/field.λ, 1/field.dx, len(a))
            A = bluestein_fft(A, 0, b[0]/field.λ, b[-1]/field.λ, 1/field.dy, len(b))
            A *= field.dx*field.dy/field.λ
        return bd.abs(A)**2

    def sample(values):
        result = bd.zeros(alpha.size)
        for j in range(iy.shape[1]):
            result += wy[:, j]*bd.sum(values[iy[:, j, None], ix]*wx, axis=1)
        return result

    monitor_nodes = bd.asarray([(i, j) for j in range(len(mny)) for i in range(len(mnx))])
    def source_contribution(result, indices):
        i, j = indices
        def weighted_source(u, v):
            if not np.isfinite(z1):
                return illumination(u, v)
            cs = bd.sqrt(bd.maximum(1-u*u-v*v, 1e-30))
            return illumination(u, v)*_weight(snx, i, u/cs)*_weight(sny, j, v/cs)
        _, _, mass = source_kernel(weighted_source, bounds, angular_step, source_subpixels)
        source_fft = bd.fft.rfft2(mass, s=fft_shape)
        incident = _residual(field, z1, snx[i], sny[j]) if np.isfinite(z1) else 1.
        pupil = field.E*incident
        def monitor_contribution(result, indices):
            u, v = indices
            H = spectrum_intensity(pupil*_residual(field, z, mnx[u], mny[v]))
            S = bd.fft.irfft2(source_fft*bd.fft.rfft2(H, s=fft_shape), s=fft_shape)
            return result+mx[u]*my[v]*sample(S), None
        if is_jax:
            result, _ = jax.lax.scan(jax.checkpoint(monitor_contribution), result, monitor_nodes)
        else:
            for pair in monitor_nodes:
                result, _ = monitor_contribution(result, pair)
        return result, None

    source_nodes = bd.asarray([(i, j) for j in range(len(sny)) for i in range(len(snx))])
    result = bd.zeros(alpha.size)
    if is_jax:
        import jax
        result, _ = jax.lax.scan(jax.checkpoint(source_contribution), result, source_nodes)
    else:
        for pair in source_nodes:
            result, _ = source_contribution(result, pair)
    # Keep the absolute scale and any interpolation error visible; do not clip
    # or normalize the reference to the prediction it is meant to check.
    return (result*bd.asarray(c**4/z**2)).reshape(shape)

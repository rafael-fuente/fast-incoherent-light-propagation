"""Full angular-spectrum response on a wide-angle, direction-cosine grid.

The Weyl angular-spectrum integral is evaluated on a steepest-descent contour.
This removes its rapidly oscillating carrier without a Fresnel approximation.
Only the pupil Fourier spectrum is sampled; no real-space RS kernel is used.
"""
import numpy as np
from functools import lru_cache
from ..util import backend_functions
from ..util.bluestein_FFT import bluestein_fft


@lru_cache(maxsize=32)
def _contour_needs_refinement(wavelength, z, step, epsilon):
    """Check whether complex interpolation would strongly amplify round-off.

    The estimate uses the contour's imaginary offsets and the interpolation
    weights. It depends only on geometry and precision, so changing the pupil
    phase reuses the decision. It detects poor numerical conditioning; it is
    not an estimate of the final diffraction error.
    """
    nodes, weights = np.polynomial.laguerre.laggauss(6)
    tau = nodes/(2*np.pi/wavelength*z)
    R = np.sqrt(tau*tau-2j*tau)
    phi = np.linspace(0, np.pi/2, 33)
    indices = np.arange(12)
    # Bound each numerator over one real interpolation cell.
    distance = np.maximum(abs(5-indices), abs(6-indices))
    denominators = np.array([np.prod([abs(j-l) for l in indices if l != j])
                             for j in indices])
    norms = []
    for angle in (np.cos(phi), np.sin(phi)):
        imaginary = abs(R.imag[:, None])*angle/step
        factors = np.sqrt(distance**2+imaginary[..., None]**2)
        terms = np.prod(factors, axis=-1)[..., None]/factors/denominators
        norms.append(terms.sum(axis=-1))
    amplification = np.sum(weights*abs(1+1j*tau)*np.max(norms[0]*norms[1], axis=1))
    return bool(epsilon*amplification > 1e-4)


def _contour_rule(wavelength, z, extent, spectrum_step, stride, dtype):
    """Choose a well-conditioned interpolation grid for a fixed distance.

    Keep the original fast rule whenever its weights are well behaved. For
    a poorly conditioned rule, spread the interpolation nodes over a wider
    real-frequency interval and refine the contour quadrature. A higher
    polynomial order alone would amplify round-off further.

    Return the spectrum stride, interpolation order, radial node count and
    azimuth count. This changes only internal integration; the angular
    convolution grid and the requested monitor retain their sampling.
    """
    epsilon = np.finfo(np.empty((), dtype=dtype).real.dtype).eps
    if not _contour_needs_refinement(wavelength, z, spectrum_step*stride, epsilon):
        return stride, 12, 6, 12
    # Widen the stencil while staying below the pupil spectrum's Nyquist step.
    stride = max(stride, int(np.ceil(.6*wavelength/(extent*spectrum_step))))
    return stride, 24, 8, 16


def _sample_analytic_spectrum(spectrum, u, v, start, step, order=12):
    """Local polynomial continuation of an entire pupil Fourier spectrum."""
    bd = backend_functions.backend
    u, v = (u-start)/step, (v-start)/step
    ix, iy = bd.floor(u.real).astype(int)-(order//2-1), bd.floor(v.real).astype(int)-(order//2-1)
    nodes = bd.arange(order)[:, None]
    wx = bd.ones((order, u.size), dtype=u.dtype)
    wy = bd.ones((order, v.size), dtype=v.dtype)
    for j in range(order):
        denominator = bd.where(nodes == j, 1, nodes-j)
        wx *= bd.where(nodes == j, 1, (u-ix-j)[None, :]/denominator)
        wy *= bd.where(nodes == j, 1, (v-iy-j)[None, :]/denominator)
    value = bd.zeros_like(u, dtype=bd.result_type(spectrum, u))
    if backend_functions.backend_name == 'jax':
        return interpolate_spectrum(spectrum, ix, iy, wx, wy)
    for j in range(order):
        row = bd.sum(wx*spectrum[iy[None, :]+j, ix[None, :]+bd.arange(order)[:, None]], axis=0)
        value += wy[j]*row
    return value


if backend_functions.JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    @jax.custom_vjp
    def interpolate_spectrum(spectrum, ix, iy, wx, wy):
        bd = jnp
        columns = ix[None, :]+bd.arange(wx.shape[0])[:, None]
        def add_row(value, j):
            row = bd.sum(wx*spectrum[iy[None, :]+j, columns], axis=0)
            return value+wy[j]*row, None
        value, _ = jax.lax.scan(add_row, bd.zeros_like(ix, dtype=spectrum.dtype),
                                bd.arange(wx.shape[0]))
        return value

    def interpolation_forward(spectrum, ix, iy, wx, wy):
        return interpolate_spectrum(spectrum, ix, iy, wx, wy), (spectrum, ix, iy, wx, wy)

    def interpolation_backward(saved, gradient):
        # Accumulate directly into one spectrum adjoint; avoid a full zero
        # spectrum for every interpolation row in the default gather transpose.
        bd = jnp
        spectrum, ix, iy, wx, wy = saved
        columns = ix[None, :]+bd.arange(wx.shape[0])[:, None]
        def add_row(carry, j):
            gs, gx, gy = carry
            rows = iy[None, :]+j
            values = spectrum[rows, columns]
            gs = gs.at[rows, columns].add(gradient[None, :]*wy[j]*wx)
            gx = gx+gradient[None, :]*wy[j]*values
            gy = gy.at[j].set(gradient*bd.sum(wx*values, axis=0))
            return (gs, gx, gy), None
        initial = (bd.zeros_like(spectrum), bd.zeros_like(wx, dtype=spectrum.dtype),
                   bd.zeros_like(wy, dtype=spectrum.dtype))
        (gs, gx, gy), _ = jax.lax.scan(add_row, initial, bd.arange(wx.shape[0]))
        gx = gx if bd.iscomplexobj(wx) else gx.real
        gy = gy if bd.iscomplexobj(wy) else gy.real
        return gs, None, None, gx, gy

    interpolate_spectrum.defvjp(interpolation_forward, interpolation_backward)
    _jax_sample_analytic_spectrum = jax.jit(_sample_analytic_spectrum, static_argnames=("order",))


def sample_analytic_spectrum(spectrum, u, v, start, step, order=12):
    """
    Interpolate a finite pupil's Fourier spectrum at real or complex directions.

    Parameters
    ----------
    spectrum: two-dimensional complex spectrum on a uniform angular grid.
    u: one-dimensional array of horizontal direction cosines, possibly complex.
    v: matching array of vertical direction cosines, possibly complex.
    start: first coordinate of both spectrum axes.
    step: angular spacing of both spectrum axes.
    order: number of interpolation points per axis, default 12.

    Returns
    -------
    values: interpolated complex spectrum, with the same shape as u.

    Notes
    -----
    The interpolation stencil must lie within the supplied grid. Uses the
    active backend and retains derivatives through spectrum and coordinates.
    """
    sample = (_jax_sample_analytic_spectrum if backend_functions.backend_name == 'jax'
              else _sample_analytic_spectrum)
    return sample(spectrum, u, v, start, step, order)



def angular_spectrum_response(field, z, samples=1024, limit=2.0):
    """
    Evaluate the full angular-spectrum intensity kernel on a uniform grid.

    Parameters
    ----------
    field: input MonochromaticField representing a finite pupil.
    z: positive pupil-to-monitor distance, in micrometres.
    samples: number of samples per angular axis, at least 8.
    limit: angular grid half-width, at least 2.

    Returns
    -------
    a: one-dimensional direction-cosine axis, shared by alpha and beta.
    K: weighted coherent intensity with shape (samples, samples).

    Notes
    -----
    Inside the unit disk, K = z² |E(z*alpha/c, z*beta/c)|² / c⁴, where
    c = sqrt(1-alpha²-beta²). Outside it, K continues as the pupil Fourier
    intensity divided by wavelength² for the shifted-source convolution.

    The Weyl integral is evaluated on a steepest-descent contour. Setting
    its longitudinal direction cosine to 1+i*t/(k*rho) removes the rapid
    carrier, leaving exp(-t) for Gauss-Laguerre integration. The finite
    pupil has an entire Fourier transform, allowing complex interpolation.
    Fixed-geometry calculations check the conditioning of the interpolation.
    Difficult cases use a wider interpolation stencil and a finer contour
    rule; the usual fast rule is retained elsewhere. Pupil derivatives are
    preserved. Traced distances retain the original rule. As with other
    numerical propagators, demanding geometries need convergence checks.
    """
    bd, backend_name = backend_functions.backend, backend_functions.backend_name
    traced_distance = backend_name == 'jax' and isinstance(z, jax.core.Tracer)
    if (not traced_distance and (not np.isfinite(z) or z <= 0)) or samples < 8 or limit < 2:
        raise ValueError('Use z > 0, samples >= 8 and limit >= 2.')
    da = 2*limit/samples
    a = (bd.arange(samples)-samples//2)*da
    factor = max(2, int(np.ceil(8*limit*max(field.extent_x, field.extent_y)/(field.λ*samples))))
    n = factor*samples
    fine_a = a[0]+bd.arange(n)*da/factor
    # Complex continuation needs a stencil spanning the pupil bandwidth.
    # An excessively fine grid gives huge cancelling extrapolation weights.
    stride = max(1, int(n*field.λ/(8*limit*max(field.extent_x, field.extent_y))))
    k = 2*bd.pi/field.λ
    if hasattr(field, "pupil_quadrature"):
        spectrum = field.pupil_quadrature.fourier_transform(field.E, fine_a/field.λ, fine_a/field.λ)/field.λ
    else:
        spectrum = bluestein_fft(field.E, 1, fine_a[0]/field.λ, fine_a[-1]/field.λ, 1/field.dx, n)
        spectrum = bluestein_fft(spectrum, 0, fine_a[0]/field.λ, fine_a[-1]/field.λ, 1/field.dy, n)
        spectrum *= field.dx*field.dy/field.λ
        spectrum *= bd.exp(-1j*k*field.x[0]*fine_a)[None, :]
        spectrum *= bd.exp(-1j*k*field.y[0]*fine_a)[:, None]
    K = bd.abs(spectrum[::factor, ::factor])**2

    order, radial_nodes, azimuth_count = 12, 6, 12
    if not traced_distance and not hasattr(field, "pupil_quadrature"):
        stride, order, radial_nodes, azimuth_count = _contour_rule(
            float(field.λ), float(z), max(field.extent_x, field.extent_y),
            da/factor, stride, spectrum.dtype)

    grid = (np.arange(samples)-samples//2)*da
    ax, by = np.meshgrid(grid, grid)
    rows, cols = np.nonzero(ax*ax+by*by < 1)
    alpha, beta = bd.asarray(ax[rows, cols]), bd.asarray(by[rows, cols])
    r = bd.sqrt(alpha*alpha+beta*beta)
    c = bd.sqrt(1-r*r)
    cosine = bd.where(r > 0, alpha/bd.maximum(r, 1e-30), 1.)
    sine = bd.where(r > 0, beta/bd.maximum(r, 1e-30), 0.)
    nodes, weights = np.polynomial.laguerre.laggauss(radial_nodes)
    azimuths = 2*np.pi*np.arange(azimuth_count)/azimuth_count
    def contribution(t, weight, phi):
        tau = t*c/(k*z)
        C = 1+1j*tau
        R = bd.sqrt(tau*tau-2j*tau)
        u = alpha*C+R*(c*cosine*bd.cos(phi)-sine*bd.sin(phi))
        v = beta*C+R*(c*sine*bd.cos(phi)+cosine*bd.sin(phi))
        A = sample_analytic_spectrum(spectrum[::stride, ::stride], u, v,
                                     fine_a[0], stride*da/factor, order=order)
        return weight*(C-r/c*R*bd.cos(phi))*A/len(azimuths)

    response = bd.zeros_like(alpha, dtype=complex)
    if backend_name == 'jax':
        # One compiled loop; recompute interpolation during differentiation
        # instead of retaining a tape for every contour node.
        @jax.checkpoint
        def accumulate(response, parameters):
            return response+contribution(*parameters), None

        parameters = bd.asarray([(t, w, phi) for t, w in zip(nodes, weights)
                                 for phi in azimuths])
        response, _ = jax.lax.scan(accumulate, response, parameters)
        # nonzero() gives distinct, sorted coordinates.
        K = K.at[rows, cols].set(bd.abs(response)**2,
                                 indices_are_sorted=True, unique_indices=True)
    else:
        for t, weight in zip(nodes, weights):
            for phi in azimuths:
                response += contribution(t, weight, phi)
        K[rows, cols] = bd.abs(response)**2
    return a, K


def angular_spectrum_response_at(field, z, alpha, beta):
    """
    Evaluate the full angular-spectrum intensity kernel at paired directions.

    Parameters
    ----------
    field: input MonochromaticField representing a finite pupil.
    z: positive pupil-to-monitor distance, in micrometres.
    alpha: array of horizontal direction cosines.
    beta: array of vertical direction cosines with the same shape as alpha.

    Returns
    -------
    K: weighted coherent intensity with the coordinate shape and the
        normalization described in angular_spectrum_response.

    Notes
    -----
    Sparse coordinates avoid an unused full angular image. The Fourier
    usual grid uses four samples per pupil bandwidth, 24-point interpolation
    and a 12 x 32 contour rule. A poorly conditioned grid uses more widely
    spaced interpolation nodes. Physical pupil quadrature is supported.
    """
    bd = backend_functions.backend
    return bd.abs(_angular_spectrum_amplitude_at(field, z, alpha, beta))**2


def _angular_spectrum_amplitude_at(field, z, alpha, beta, full_grid=False, dense=False,
                                   fixed_grid=False):
    """Complex response before the physical carrier and monitor weight.

    Keeping this amplitude makes zoom propagation phase preserving. The
    existing intensity API uses exactly the same spectrum and contour rule.
    full_grid keeps the Fourier grid independent of a zoom window or distance.
    dense uses the 12-point, 6 x 12 rule of angular_spectrum_response.
    fixed_grid applies its fixed numerical integration weights in blocks;
    the current pupil spectrum and its gradient are still evaluated anew.
    An unstable fixed-geometry stencil uses the refined rule instead of
    those cached weights. The usual cached path is retained otherwise.
    """
    bd, backend_name = backend_functions.backend, backend_functions.backend_name
    traced_distance = backend_name == 'jax' and isinstance(z, jax.core.Tracer)
    shape = alpha.shape
    alpha, beta = alpha.ravel(), beta.ravel()
    limit = 2.0
    if not full_grid and isinstance(alpha, np.ndarray) and np.all(alpha*alpha+beta*beta < 1):
        limit = float(np.ceil((max(np.max(abs(alpha)), np.max(abs(beta)))+.15)*20)/20)
    n = int(np.ceil(8*limit*max(field.extent_x, field.extent_y)/field.λ))
    if full_grid:
        n = max(n, 64)  # Leave room for the 24-point stencil for subwavelength pupils.
    step = 2*limit/n
    a = -limit+bd.arange(n)*step
    k = 2*bd.pi/field.λ
    if hasattr(field, "pupil_quadrature"):
        spectrum = field.pupil_quadrature.fourier_transform(field.E, a/field.λ, a/field.λ)/field.λ
    else:
        spectrum = bluestein_fft(field.E, 1, a[0]/field.λ, a[-1]/field.λ, 1/field.dx, n)
        spectrum = bluestein_fft(spectrum, 0, a[0]/field.λ, a[-1]/field.λ, 1/field.dy, n)
        spectrum *= field.dx*field.dy/field.λ
        spectrum *= bd.exp(-1j*k*field.x[0]*a)[None, :]
        spectrum *= bd.exp(-1j*k*field.y[0]*a)[:, None]

    rule = None
    if not traced_distance and not hasattr(field, "pupil_quadrature"):
        candidate = _contour_rule(float(field.λ), float(z),
            max(field.extent_x, field.extent_y), step, 1, spectrum.dtype)
        if candidate[1:] != (12, 6, 12):
            stride, *rule = candidate
            spectrum = spectrum[::stride, ::stride]
            step *= stride
            rule = tuple(rule)
            if not dense:
                rule = (max(rule[0], 24), max(rule[1], 12), max(rule[2], 32))

    if rule is None and fixed_grid and not traced_distance and isinstance(alpha, np.ndarray) and np.all(alpha*alpha+beta*beta < 1):
        from ..util.fixed_response import fixed_response
        return fixed_response(spectrum, field.λ, float(z), n, limit, alpha, beta, dense).reshape(shape)

    if rule is None and not traced_distance and isinstance(alpha, np.ndarray) and alpha.size <= 5000 and np.all(alpha*alpha+beta*beta < 1):
        indices, coefficients = prepare_spectrum_stencil(field.λ, float(z), n, limit,
            np.asarray(alpha, dtype=np.float64).tobytes(), np.asarray(beta, dtype=np.float64).tobytes(), dense)
        response = bd.sum(bd.asarray(coefficients)*spectrum.ravel()[bd.asarray(indices)], axis=1)
        return response.reshape(shape)

    block = min(alpha.size, 2048)
    count = (alpha.size+block-1)//block
    alpha_blocks = bd.pad(bd.asarray(alpha), (0, count*block-alpha.size)).reshape(count, block)
    beta_blocks = bd.pad(bd.asarray(beta), (0, count*block-beta.size)).reshape(count, block)
    def evaluate_block(_, directions):
        values = _evaluate_spectrum_amplitude_at(spectrum, k, z, a[0], step,
                                                *directions, dense=dense, rule=rule)
        return None, values
    if backend_name == 'jax':
        _, values = jax.lax.scan(jax.checkpoint(evaluate_block), None, (alpha_blocks, beta_blocks))
    else:
        values = bd.stack([evaluate_block(None, pair)[1] for pair in zip(alpha_blocks, beta_blocks)])
    return values.ravel()[:alpha.size].reshape(shape)


def evaluate_spectrum_at(spectrum, k, z, start, step, alpha, beta):
    """
    Compute paired-direction intensity from an existing pupil spectrum.

    Parameters
    ----------
    spectrum: complex pupil Fourier integral divided by wavelength, on a square grid.
    k: wavenumber 2*pi/wavelength, in inverse micrometres.
    z: positive pupil-to-monitor distance, in micrometres.
    start: first direction-cosine coordinate of both spectrum axes.
    step: angular grid spacing.
    alpha: one-dimensional horizontal direction cosines.
    beta: matching vertical direction cosines.

    Returns
    -------
    K: weighted coherent intensity at each pair, as in angular_spectrum_response_at.
    """
    bd = backend_functions.backend
    return bd.abs(_evaluate_spectrum_amplitude_at(spectrum, k, z, start, step, alpha, beta))**2


def _evaluate_spectrum_amplitude_at(spectrum, k, z, start, step, alpha, beta, dense=False, rule=None):
    bd, backend_name = backend_functions.backend, backend_functions.backend_name
    order, radial_nodes, azimuth_count = rule or ((12, 6, 12) if dense else (24, 12, 32))
    propagating = alpha*alpha+beta*beta < 1
    continuation = sample_analytic_spectrum(spectrum, alpha, beta, start, step, order)
    alpha, beta = bd.where(propagating, alpha, 0), bd.where(propagating, beta, 0)
    c = bd.sqrt(1-alpha*alpha-beta*beta)
    ex = 1-alpha*alpha/(1+c)
    ey = 1-beta*beta/(1+c)
    cross = -alpha*beta/(1+c)
    nodes, weights = np.polynomial.laguerre.laggauss(radial_nodes)
    azimuths = 2*np.pi*np.arange(azimuth_count)/azimuth_count

    parameters = bd.asarray([(t, w, phi) for t, w in zip(nodes, weights) for phi in azimuths])
    def add_direction(response, parameters):
        t, weight, phi = parameters
        tau = t*c/(k*z)
        C = 1+1j*tau
        R = bd.sqrt(tau*tau-2j*tau)
        u = alpha*C+R*(ex*bd.cos(phi)+cross*bd.sin(phi))
        v = beta*C+R*(cross*bd.cos(phi)+ey*bd.sin(phi))
        A = sample_analytic_spectrum(spectrum, u, v, start, step, order)
        weight = weight*(C-R*(alpha*bd.cos(phi)+beta*bd.sin(phi))/c)/len(azimuths)
        return response+weight*A, None
    response = bd.zeros_like(alpha, dtype=complex)
    if backend_name == 'jax':
        response, _ = jax.lax.scan(jax.checkpoint(add_direction), response, parameters)
    else:
        for parameters_row in parameters:
            response, _ = add_direction(response, parameters_row)
    return bd.where(propagating, response, continuation)


@lru_cache(maxsize=2)
def prepare_spectrum_stencil(wavelength, z, n, limit, alpha_bytes, beta_bytes, dense=False):
    """
    Cache the angular integration weights for fixed geometry.

    Parameters
    ----------
    wavelength: wavelength in micrometres.
    z: positive pupil-to-monitor distance, in micrometres.
    n: number of samples per spectrum axis.
    limit: spectrum half-width in direction-cosine coordinates.
    alpha_bytes: horizontal directions serialized from a float64 NumPy array.
    beta_bytes: matching serialized vertical directions; pairs lie inside
        the propagating unit disk.
    dense: True uses 12-point interpolation and a 6 x 12 contour rule;
        False uses 24-point interpolation and a 12 x 32 rule.

    Returns
    -------
    indices: flattened spectrum indices with one stencil per direction.
    coefficients: complex integration weights with the same shape as indices.

    Notes
    -----
    Only fixed geometry is reused; no pupil spectrum or response is cached.
    """
    alpha = np.frombuffer(alpha_bytes, dtype=np.float64)
    beta = np.frombuffer(beta_bytes, dtype=np.float64)
    step, start = 2*limit/n, -limit
    c = np.sqrt(1-alpha*alpha-beta*beta)
    ex, ey, cross = 1-alpha*alpha/(1+c), 1-beta*beta/(1+c), -alpha*beta/(1+c)
    order = 12 if dense else 24
    nodes, weights = np.polynomial.laguerre.laggauss(6 if dense else 12)
    azimuth_count = 12 if dense else 32
    parameters = np.array([(t, w, phi) for t, w in zip(nodes, weights)
                           for phi in 2*np.pi*np.arange(azimuth_count)/azimuth_count])
    t, w, phi = (parameters[:, j, None] for j in range(3))
    tau = t*c/(2*np.pi/wavelength*z)
    C, R = 1+1j*tau, np.sqrt(tau*tau-2j*tau)
    u = (alpha*C+R*(ex*np.cos(phi)+cross*np.sin(phi))-start)/step
    v = (beta*C+R*(cross*np.cos(phi)+ey*np.sin(phi))-start)/step
    weight = w*(C-R*(alpha*np.cos(phi)+beta*np.sin(phi))/c)/azimuth_count
    ix, iy = np.floor(u.real).astype(int)-(order//2-1), np.floor(v.real).astype(int)-(order//2-1)
    left, bottom = ix.min(axis=0), iy.min(axis=0)
    width, height = int((ix-left).max())+order, int((iy-bottom).max())+order
    coefficients = np.zeros((alpha.size, height, width), complex)
    point = np.arange(alpha.size)[:, None]
    interpolation_nodes = np.arange(order)[:, None]
    for uu, vv, xx, yy, ww in zip(u, v, ix, iy, weight):
        wx, wy = np.ones((order, alpha.size), complex), np.ones((order, alpha.size), complex)
        for j in range(order):
            denominator = np.where(interpolation_nodes == j, 1, interpolation_nodes-j)
            wx *= np.where(interpolation_nodes == j, 1, (uu-xx-j)[None, :]/denominator)
            wy *= np.where(interpolation_nodes == j, 1, (vv-yy-j)[None, :]/denominator)
        columns = (xx-left)[:, None]+np.arange(order)
        for j in range(order):
            rows = (yy-bottom+j)[:, None]
            coefficients[point, rows, columns] += (ww*wy[j]*wx).T
    yy, xx = np.indices((height, width))
    indices = (bottom[:, None]+yy.ravel())*n+left[:, None]+xx.ravel()
    return indices.astype(np.int32), coefficients.reshape(alpha.size, -1)

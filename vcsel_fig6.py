"""Small forward models for the paper's VCSEL comparison.

These helpers leave diffractsim unchanged. Static grids and interpolation
weights are prepared once; phase-dependent calculations use JAX throughout.
"""
import numpy as np
from scipy import fft
import jax
import jax.numpy as jnp
from scipy.ndimage import map_coordinates, spline_filter


def order_coefficients(phase, orders):
    """Fourier coefficients of the repeated, piecewise-constant phase cell."""
    n = phase.shape[0]
    index = jnp.asarray(orders % n)
    factor = jnp.sinc(jnp.asarray(orders)/n)*jnp.exp(
        1j*jnp.pi*jnp.asarray(orders)*(1-1/n))
    spectrum = jnp.fft.fft2(jnp.exp(1j*phase))/n**2
    return spectrum[index[:, None], index[None, :]]*factor[:, None]*factor[None, :]


def _interpolator(axis, x, y, weight=1, order=1):
    """Prepare a bilinear or six-point sampler at fixed monitor coordinates."""
    n, step = len(axis), axis[1]-axis[0]
    fx, fy = (np.asarray(x)-axis[0])/step, (np.asarray(y)-axis[0])/step
    if order == 5:
        def stencil(position):
            start = np.floor(position.ravel()).astype(np.int32)-2
            t = position.ravel()-start
            indices = start[:, None]+np.arange(6, dtype=np.int32)
            if indices.min() < 0 or indices.max() >= n:
                raise ValueError("The interpolation stencil lies outside the angular grid.")
            weights = np.ones((t.size, 6))
            for i in range(6):
                for j in range(6):
                    if i != j:
                        weights[:, i] *= (t-j)/(i-j)
            return jnp.asarray(indices), jnp.asarray(weights)
        ix, wx = stencil(fx)
        iy, wy = stencil(fy)
        scale = jnp.asarray(np.broadcast_to(weight, fx.shape).ravel())
        def sample(values):
            flat = values.ravel()
            result = jnp.zeros(fx.size)
            for j in range(6):
                gathered = flat.take(iy[:, j, None]*n+ix)
                result += wy[:, j]*jnp.sum(gathered*wx, axis=1)
            return (result*scale).reshape(fx.shape)
        return sample
    if order != 1:
        raise ValueError("Use interpolation order 1 or 5.")
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    if min(ix.min(), iy.min()) < 0 or max(ix.max(), iy.max())+1 >= n:
        raise ValueError("The angular grid does not cover the monitor.")
    dx, dy = fx-ix, fy-iy
    indices, weights = [], []
    for i, j, w in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                    (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        indices.append(jnp.asarray(((iy+j)*n+ix+i).ravel()))
        weights.append(jnp.asarray((w*weight).ravel()))

    def sample(values):
        flat = values.ravel()
        return sum(flat.take(i)*w for i, w in zip(indices, weights)).reshape(fx.shape)
    return sample


def prepare_forward(model, *, x, y, z, wavelength, period, radius,
                    source_positions, source_sigma, emitter_power,
                    samples_per_order=6, order_limit=1.12):
    """Prepare one of the four design models on a common physical monitor.

    eq8 translates a paraxial response. mapped8 maps the central response
    into direction cosines, then translates it on the screen without c^4.
    eq22 uses the far-field angular convolution. eq24 also includes the
    first-order finite-distance Gaussian covariance from Supplement S6.3.
    The returned function takes phase in radians and returns W/m^2.
    The two translating models use a uniform monitor with square pixels.
    """
    if model not in ("eq8", "mapped8", "eq22", "eq24"):
        raise ValueError("Unknown VCSEL model.")
    du = wavelength/period
    spacing = 3 if model == "eq8" else samples_per_order
    limit = 1.95 if model == "eq8" else order_limit
    h = du/spacing
    orders = np.arange(-int(np.ceil(limit/du)), int(np.ceil(limit/du))+1)
    sigma = np.hypot(source_sigma, wavelength/(2*np.pi*radius))
    positions = np.asarray(source_positions)
    guard = int(np.ceil((abs(positions).max()+6*sigma)/h))+2
    n = fft.next_fast_len(2*(orders[-1]*spacing+guard)+8, real=True)
    axis = (np.arange(n)-n//2)*h
    index = n//2+orders*spacing
    off = fft.ifftshift(axis)
    gaussian = np.exp(-off**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    if model in ("eq8", "mapped8"):
        kernel = np.outer(gaussian, gaussian)
    else:
        mixture = np.exp(-(off[:, None]+positions)**2/(2*sigma**2)).sum(1)
        kernel = emitter_power*np.outer(mixture, mixture)/(2*np.pi*sigma**2)
    kernel_hat = jnp.asarray(fft.rfft2(kernel))
    xx, yy = np.meshgrid(np.asarray(x)/z, np.asarray(y)/z)
    rho = np.sqrt(1+xx*xx+yy*yy)

    def place(values):
        return jnp.zeros((n, n)).at[index[:, None], index[None, :]].set(values)

    if model in ("eq22", "eq24"):
        sample = _interpolator(axis, xx/rho, yy/rho, 1/(z*z*rho**4))
        u, v = np.meshgrid(orders*du, orders*du)
        r2 = u*u+v*v
        scale = (radius/(2*z))**2*np.maximum(1-r2, 0)
        cov = [jnp.asarray(scale*(1-(2-r2)*u*u)),
               jnp.asarray(-scale*(2-r2)*u*v),
               jnp.asarray(scale*(1-(2-r2)*v*v))]
        fx, fy = fft.rfftfreq(n, h)[None, :], fft.fftfreq(n, h)[:, None]
        multipliers = [jnp.asarray(-2*np.pi**2*fx*fx),
                       jnp.asarray(-4*np.pi**2*fx*fy),
                       jnp.asarray(-2*np.pi**2*fy*fy)]

        def forward(phase):
            power = abs(order_coefficients(phase, orders))**2
            spectrum = jnp.fft.rfft2(place(power))
            if model == "eq24":
                for weight, factor in zip(cov, multipliers):
                    spectrum += factor*jnp.fft.rfft2(place(power*weight))
            return sample(jnp.fft.irfft2(kernel_hat*spectrum, s=(n, n)))
    else:
        qx, qy = np.meshgrid(positions, positions)
        q = np.stack([qx.ravel(), qy.ravel()], axis=1)
        landing = -q if model == "eq8" else -q/np.sqrt(1-(q*q).sum(1))[:, None]
        step = float(x[1]-x[0])/z
        if not (np.allclose(np.diff(x)/z, step) and np.allclose(np.diff(y)/z, step)):
            raise ValueError("The translating models require a uniform monitor with square pixels.")
        shifts = np.rint(landing/step).astype(int)
        margin = int(abs(shifts).max())+3
        xe = np.arange(len(x)+2*margin)*step+x[0]/z-margin*step
        ye = np.arange(len(y)+2*margin)*step+y[0]/z-margin*step
        xe, ye = np.meshgrid(xe, ye)
        rr = np.sqrt(1+xe*xe+ye*ye) if model == "mapped8" else 1
        sample = _interpolator(axis, xe/rr, ye/rr, 1/z**2)
        shifts = jnp.asarray(shifts)

        def forward(phase):
            power = abs(order_coefficients(phase, orders))**2
            central = sample(jnp.fft.irfft2(
                kernel_hat*jnp.fft.rfft2(place(power)), s=(n, n)))
            def add_emitter(i, image):
                start = (margin-shifts[i, 1], margin-shifts[i, 0])
                return image+emitter_power*jax.lax.dynamic_slice(
                    central, start, (len(y), len(x)))
            return jax.lax.fori_loop(0, len(q), add_emitter, jnp.zeros(xx.shape))
    return jax.jit(forward)


def _lagrange(nodes, value):
    """Interpolation weights at fixed source or monitor coordinates."""
    result = []
    for i, node in enumerate(nodes):
        weight = np.ones_like(value)
        for j, other in enumerate(nodes):
            if i != j:
                weight *= (value-other)/(node-other)
        result.append(weight)
    return np.asarray(result)


def prepare_rs_reference(*, x, y, z, wavelength, period, radius,
                         source_positions, source_sigma, emitter_power,
                         samples_per_order=6, order_limit=1.12,
                         monitor_order=7, envelope_samples=512,
                         pair_radius=1, pair_tolerance=1e-10,
                         interpolation_order=5,
                         progress=False):
    """Prepare an outgoing RS reference for the periodic Gaussian pupil.

    The exact RS kernel is integrated over the smooth Gaussian envelope.
    Fourier coefficients supply the fine periodic phase, so no giant tiled
    pupil is needed. The Gaussian source integral is a separate convolution.
    Nearby order-pair interference is retained unless its absolute bound is
    below pair_tolerance times the central source-kernel peak. Increase
    pair_radius to check the omitted, more distant pairs.

    Six-point angular sampling avoids sharing the design model's bilinear
    smoothing. Monitor-node interpolation, envelope quadrature, and the angular grid are
    numerical controls. They do not replace the RS kernel by a Fresnel kernel.
    All lengths here are metres and powers are watts. The returned evaluator
    remains differentiable with respect to phase; geometry is fixed.
    """
    if samples_per_order % 2:
        raise ValueError("Use an even samples_per_order for order-pair midpoints.")
    if monitor_order < 2 or envelope_samples < 128:
        raise ValueError("Use at least two monitor nodes and 128 envelope samples.")
    du, spacing = wavelength/period, samples_per_order
    h = du/spacing
    orders = np.arange(-int(np.ceil(order_limit/du)), int(np.ceil(order_limit/du))+1)
    positions = np.asarray(source_positions)
    sigma = np.hypot(source_sigma, wavelength/(2*np.pi*radius))
    guard = int(np.ceil((abs(positions).max()+8*sigma)/h))+2
    n = fft.next_fast_len(2*(orders[-1]*spacing+guard)+8, real=True)
    axis = (np.arange(n)-n//2)*h
    xx, yy = np.meshgrid(np.asarray(x)/z, np.asarray(y)/z)
    rho = np.sqrt(1+xx*xx+yy*yy)
    sample = _interpolator(axis, xx/rho, yy/rho, 1/(z*z*rho**4), interpolation_order)
    nodes_x = (xx.min()+xx.max())/2+(xx.max()-xx.min())/2*np.cos(
        np.pi*np.arange(monitor_order)/(monitor_order-1))
    nodes_y = (yy.min()+yy.max())/2+(yy.max()-yy.min())/2*np.cos(
        np.pi*np.arange(monitor_order)/(monitor_order-1))
    if xx.min() == xx.max():
        nodes_x = np.array([xx.min()])
    if yy.min() == yy.max():
        nodes_y = np.array([yy.min()])
    wx = _lagrange(nodes_x, np.asarray(x)/z)
    wy = _lagrange(nodes_y, np.asarray(y)/z)

    # The FFT spacing is one sixty-fourth of an order. Half-order shifts are
    # exact integer shifts; increasing envelope_samples refines the pupil.
    count = envelope_samples
    dq, dr = du/64, period*64/count
    r = (np.arange(count)-count//2)*dr
    rx, ry = np.meshgrid(r, r)
    envelope = np.exp(-(rx*rx+ry*ry)/radius**2)
    q = fft.fftfreq(count)*count*dq
    source = np.exp(-(q[None, :]**2+q[:, None]**2)/(2*source_sigma**2))
    source_hat = fft.fft2(source/(2*np.pi*source_sigma**2))*dq*dq
    area, k = np.pi*radius**2/2, 2*np.pi/wavelength
    compact = np.arange(-guard, guard+1)
    ka, kb = np.meshgrid(compact*h, compact*h)
    source_coordinates = [np.array([(kb+sy)/dq+count//2,
                                    (ka+sx)/dq+count//2])
                          for sy in positions for sx in positions]
    pairs = [(0, 0)]+[(dx, dy) for dy in range(pair_radius+1)
                     for dx in range(-pair_radius, pair_radius+1)
                     if dy > 0 or dx > 0]
    kernels, skipped_bound = [], 0.0
    threshold = pair_tolerance*emitter_power/(2*np.pi*sigma**2)
    for j, ty in enumerate(nodes_y):
        for i, tx in enumerate(nodes_x):
            c = 1/np.sqrt(1+tx*tx+ty*ty)
            distance = z/c
            dot = c*(tx*rx+ty*ry)
            r2 = rx*rx+ry*ry
            R = distance*np.sqrt(1-2*dot/distance+r2/distance**2)
            path = (r2-dot*dot)/(R+distance-dot)
            residual = (distance/R)**2*(1+1j/(k*R))*np.exp(1j*k*path)
            amplitude = fft.fftshift(fft.fft2(fft.ifftshift(envelope*residual)))
            amplitude *= dr*dr/(wavelength*np.sqrt(area))
            weight_bound = abs(wx[i]).max()*abs(wy[j]).max()
            for dx, dy in pairs:
                plus = np.roll(amplitude, (-32*dy, -32*dx), axis=(0, 1))
                minus = np.roll(amplitude, (32*dy, 32*dx), axis=(0, 1))
                response = plus*minus.conj()
                broadened = fft.fftshift(fft.ifft2(
                    fft.fft2(fft.ifftshift(response))*source_hat))
                filtered = spline_filter(broadened.real)+1j*spline_filter(broadened.imag)
                kernel = sum(emitter_power*map_coordinates(filtered, coord,
                    order=3, mode="constant", cval=0, prefilter=False)
                    for coord in source_coordinates)
                if dx == dy == 0:
                    kernel = kernel.real
                elif abs(kernel).max() < threshold:
                    # Cauchy-Schwarz bounds the sum of |b_m b_(m+d)*| by one.
                    skipped_bound += 2*abs(kernel).max()*weight_bound/z**2
                    continue
                kernels.append((dx, dy, jnp.asarray(kernel),
                                jnp.asarray(wx[i]), jnp.asarray(wy[j])))
        if progress:
            print(f"RS kernels: {j+1}/{len(nodes_y)} monitor rows", flush=True)

    indices = n//2+orders*spacing
    compact_indices = compact % n

    def evaluate(phase):
        coefficients = order_coefficients(phase, orders)
        spectra = {}
        for dx, dy in {(item[0], item[1]) for item in kernels}:
            rows = np.arange(len(orders))
            cols = np.arange(len(orders))
            rows = rows[(rows+dy >= 0) & (rows+dy < len(orders))]
            cols = cols[(cols+dx >= 0) & (cols+dx < len(orders))]
            values = coefficients[rows[:, None], cols[None, :]]*jnp.conj(
                coefficients[(rows+dy)[:, None], (cols+dx)[None, :]])
            iy, ix = indices[rows]+dy*spacing//2, indices[cols]+dx*spacing//2
            grid = jnp.zeros((n, n), dtype=complex).at[iy[:, None], ix[None, :]].set(values)
            spectra[dx, dy] = jnp.fft.rfft2(grid.real) if dx == dy == 0 else jnp.fft.fft2(grid)
        image = jnp.zeros(xx.shape)
        for dx, dy, kernel, weight_x, weight_y in kernels:
            grid = jnp.zeros((n, n), dtype=kernel.dtype)
            grid = grid.at[compact_indices[:, None], compact_indices[None, :]].set(kernel)
            if dx == dy == 0:
                angular = jnp.fft.irfft2(spectra[dx, dy]*jnp.fft.rfft2(grid), s=(n, n))
            else:
                angular = 2*jnp.fft.ifft2(spectra[dx, dy]*jnp.fft.fft2(grid)).real
            image += weight_y[:, None]*weight_x[None, :]*sample(angular)
        return image

    # A centered six-point stencil has sum(abs(weights)) < 2 on each axis.
    # Include that possible amplification in the discarded-pair bound.
    bound_factor = 4 if interpolation_order == 5 else 1
    details = dict(monitor_order=monitor_order, envelope_samples=count,
        angular_samples=n, samples_per_order=spacing, pair_radius=pair_radius,
        interpolation_order=interpolation_order,
        interpolation_bound_factor=bound_factor,
        retained_node_pair_kernels=len(kernels),
        skipped_pair_absolute_bound_W_m2=float(bound_factor*skipped_bound))
    return evaluate, details

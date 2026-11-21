"""The paper's source-dependent flat-top target, built once before optimization.

This example helper contains the target construction only. The source, pupil,
forward model, loss and Adam update remain visible in the notebook.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def make_flattop_target(x, y, z, field_degrees, *, angular_samples, angular_step,
                        source_kernel, input_power, iterations=60):
    """Return relative target irradiance, its flat interior and collection field.

    x, y and z are in metres; field_degrees gives the full horizontal/vertical
    angles. source_kernel(alpha, beta) is the angular power density of one
    diffraction order, including source width and Gaussian-pupil broadening;
    its integral is input_power. The uniform angular grid matches propagation.

    The multiplicative update flattens the interior while retaining the edge
    roll-off imposed by the source. Array calculations use JAX; this fixed target
    is constructed once, outside the phase-optimization loop.
    """
    a = (jnp.arange(angular_samples)-angular_samples//2)*angular_step
    aa, bb = jnp.meshgrid(a, a)
    c2 = jnp.maximum(1-aa*aa-bb*bb, 1e-12)
    half_x, half_y = np.tan(np.deg2rad(np.asarray(field_degrees)/2))
    inside = ((aa*aa+bb*bb < 1) & (abs(aa/jnp.sqrt(c2)) <= half_x)
              & (abs(bb/jnp.sqrt(c2)) <= half_y))
    offset = jnp.fft.ifftshift(a)
    kernel_fft = jnp.fft.rfft2(source_kernel(-offset[None, :], -offset[:, None]))

    def convolve(image, spectrum=kernel_fft):
        return jnp.fft.irfft2(jnp.fft.rfft2(image)*spectrum, s=image.shape)

    # The entire source footprint must lie inside the desired angular field.
    coverage = convolve(inside*angular_step**2)/input_power
    safe = inside & (coverage >= 0.999)
    projection = jnp.where(aa*aa+bb*bb < 1, c2*c2, 0)
    response = jnp.where(inside, 1/(c2*c2), 0)*angular_step**2
    denominator = jnp.maximum(convolve(projection, jnp.conj(kernel_fft)), 1e-300)
    level = (projection*convolve(response))[angular_samples//2, angular_samples//2]

    @jax.jit
    def flatten(response):
        projected = projection*convolve(response)
        ratio = jnp.where(safe, level/jnp.where(safe, projected, 1), 1)
        correction = convolve(projection*ratio, jnp.conj(kernel_fft))
        return jnp.where(inside, response*correction/denominator, 0)

    for _ in range(iterations):
        response = flatten(response)

    # Eq. (24): project angular power density onto the flat monitor, then scale.
    xx, yy = jnp.meshgrid(x, y)
    rho = jnp.sqrt(z*z+xx*xx+yy*yy)
    coordinates = jnp.array([yy/rho/angular_step+angular_samples//2,
                             xx/rho/angular_step+angular_samples//2])
    target = z*z/rho**4*map_coordinates(convolve(response), coordinates,
                                       order=1, mode="constant", cval=0)
    target /= target[target.shape[0]//2, target.shape[1]//2]
    iy, ix = jnp.rint(coordinates).astype(int)
    field = (abs(xx) <= z*half_x) & (abs(yy) <= z*half_y)
    return target, safe[iy, ix], field

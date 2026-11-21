"""Memory-bounded application of a fixed angular-spectrum integration rule.

Only integration/interpolation weights are fixed. Each changed pupil still
supplies a new Fourier spectrum, and its gradient uses the same linear rule.
"""
import numpy as np
from . import backend_functions


def fixed_response(spectrum, wavelength, z, n, limit, alpha, beta, dense):
    """
    Evaluate the angular-spectrum amplitude in blocks using fixed geometry.

    Parameters
    ----------
    spectrum: complex (n, n) pupil Fourier integral divided by wavelength.
    wavelength: wavelength in micrometres, fixed for the integration geometry.
    z: positive propagation distance in micrometres, fixed for this calculation.
    n: sample count per spectrum axis.
    limit: spectrum half-width in direction-cosine coordinates.
    alpha: one-dimensional NumPy array of horizontal direction cosines.
    beta: matching vertical direction cosines, inside the propagating disk.
    dense: True uses the 12-point interpolation and 6 x 12 contour rule;
        False uses 24-point interpolation and the 12 x 32 rule.

    Returns
    -------
    amplitude: complex response with one value per paired direction, before
        the physical carrier and monitor projection factor.

    Notes
    -----
    Only integration weights are fixed. The current spectrum is evaluated
    on each call and retains its gradient.
    """
    from ..propagation_methods.direct_quadrature_angular_spectrum_method import prepare_spectrum_stencil
    bd = backend_functions.backend
    block = 512
    pieces = [prepare_spectrum_stencil(wavelength, z, n, limit,
        np.asarray(alpha[i:i+block], dtype=np.float64).tobytes(),
        np.asarray(beta[i:i+block], dtype=np.float64).tobytes(), dense)
        for i in range(0, alpha.size, block)]
    width = max(indices.shape[1] for indices, _ in pieces)
    indices = np.zeros((len(pieces), block, width), np.int32)
    coefficients = np.zeros(indices.shape, complex)
    for i, (ix, c) in enumerate(pieces):
        indices[i, :len(ix), :ix.shape[1]] = ix
        coefficients[i, :len(c), :c.shape[1]] = c
    if backend_functions.backend_name == "jax":
        return apply_stencil(spectrum, bd.asarray(indices), bd.asarray(coefficients)).ravel()[:alpha.size]
    return bd.concatenate([bd.sum(bd.asarray(c)*spectrum.ravel()[bd.asarray(ix)], axis=1)
                           for ix, c in pieces])[:alpha.size]


if backend_functions.JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def apply_stencil(spectrum, indices, coefficients):
        def sample(_, pair):
            ix, c = pair
            return None, jnp.sum(c*spectrum.ravel()[ix], axis=1)
        return jax.lax.scan(sample, None, (indices, coefficients))[1]

    def forward(spectrum, indices, coefficients):
        return apply_stencil(spectrum, indices, coefficients), (indices, coefficients, spectrum)

    def backward(saved, gradient):
        indices, coefficients, spectrum = saved
        def accumulate(result, data):
            ix, c, g = data
            return result.at[ix].add(c*g[:, None]), None
        result, _ = jax.lax.scan(accumulate, jnp.zeros(spectrum.size, spectrum.dtype),
                                (indices, coefficients, gradient))
        return result.reshape(spectrum.shape), None, None

    apply_stencil.defvjp(forward, backward)

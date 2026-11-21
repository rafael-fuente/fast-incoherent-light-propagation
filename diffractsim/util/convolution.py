"""Read a zero-padded convolution on the original uniform coordinate grid."""

from .backend_functions import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def _crop_jax(C, x, y):
        # Convolution adds origins: its sample zero is at (2*x[0], 2*y[0]).
        nx, ny = len(x), len(y)
        ux, uy = (x-2*x[0])/(x[1]-x[0]), (y-2*y[0])/(y[1]-y[0])
        ix = jnp.clip(jnp.rint(ux[0]), -nx-1, C.shape[1]+1).astype(jnp.int32)
        iy = jnp.clip(jnp.rint(uy[0]), -ny-1, C.shape[0]+1).astype(jnp.int32)
        shape = (ny+2, nx+2)  # Include one neighboring sample on each side.
        inside = (ix >= 1) & (iy >= 1) & (ix+nx < C.shape[1]) & (iy+ny < C.shape[0])
        C = jax.lax.cond(inside,
            lambda C: jax.lax.dynamic_slice(C, (iy-1, ix-1), shape),
            lambda C: jax.lax.dynamic_slice(
                jnp.pad(C, ((ny+2, ny+2), (nx+2, nx+2))),
                (iy+ny+1, ix+nx+1), shape), C)

        # Work only on this small rectangle; slices have inexpensive adjoints.
        for axis, u, start in ((1, ux, ix), (0, uy, iy)):
            fraction = u-(start+jnp.arange(len(u)))
            if axis == 0:
                fraction = fraction[:, None]
            left = jnp.where(fraction < 0, -fraction, 0)
            right = jnp.where(fraction >= 0, fraction, 0)
            before, centre, after = ((C[:, :-2], C[:, 1:-1], C[:, 2:]) if axis == 1
                                     else (C[:-2, :], C[1:-1, :], C[2:, :]))
            # At integer coordinates only the central crop contributes.
            # Otherwise one neighbor supplies ordinary linear interpolation.
            C = left*before+(1-left-right)*centre+right*after
        return C


def crop_convolution(C, x, y):
    """
    Sample a zero-padded linear convolution on the original coordinate axes.

    Parameters
    ----------
    C: two-dimensional convolution of arrays sampled on x and y; its first
        sample corresponds to (2*x[0], 2*y[0]).
    x: ascending, uniformly sampled horizontal input axis.
    y: ascending, uniformly sampled vertical input axis.

    Returns
    -------
    values: convolution on the original grid, with shape (len(y), len(x));
        fractional offsets use linear interpolation with zero outside the grid.

    Notes
    -----
    FFT padding and physical integration factors belong to the caller.
    JAX supports differentiation through the data and uniform grid.
    """
    from .backend_functions import backend as bd, backend_name
    if backend_name == "jax":
        return _crop_jax(C, x, y)

    ux, uy = (x-2*x[0])/(x[1]-x[0]), (y-2*y[0])/(y[1]-y[0])
    ix, iy = int(bd.rint(ux[0])), int(bd.rint(uy[0]))
    nx, ny = len(x), len(y)
    if (0 <= ix <= C.shape[1]-nx and 0 <= iy <= C.shape[0]-ny
            and bool(bd.all(ux == ix+bd.arange(nx)))
            and bool(bd.all(uy == iy+bd.arange(ny)))):
        return C[iy:iy+ny, ix:ix+nx]

    if backend_name == "cupy":
        from cupyx.scipy.ndimage import map_coordinates
    else:
        from scipy.ndimage import map_coordinates
    xx, yy = bd.meshgrid(x, y)
    return map_coordinates(C, bd.array([(yy-2*y[0])/(y[1]-y[0]),
                                        (xx-2*x[0])/(x[1]-x[0])]),
                           order=1, mode="constant", cval=0)

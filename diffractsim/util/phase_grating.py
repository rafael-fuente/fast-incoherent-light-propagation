"""Periodic cubic phase interpolation using the active array backend."""


def periodic_phase_grating(phase, x, y, period):
    """
    Repeat and smoothly interpolate a phase cell on Cartesian pupil coordinates.

    Parameters
    ----------
    phase: two-dimensional periodic phase cell in radians; no repeated endpoint.
    x: horizontal coordinates in micrometres, as a one-dimensional axis or meshgrid.
    y: vertical coordinates in micrometres, as a one-dimensional axis or meshgrid.
    period: cell period along both axes, in micrometres; phase[0, 0] is at (0, 0)
        modulo the period.

    Returns
    -------
    transmission: complex unit-amplitude array of shape (len(y), len(x))
        for one-dimensional axes, or the shape of the Cartesian meshgrid.

    Notes
    -----
    Interpolates exp(i*phase) with a periodic cubic B-spline, then projects
    to unit modulus. The separable passes avoid large interpolation tapes.
    The active backend retains JAX gradients through phase and period;
    grid shapes stay fixed. An exactly zero interpolated phasor returns
    unit transmission with zero derivative.
    """
    from .backend_functions import backend as bd

    phase = bd.asarray(phase)
    ny, nx = phase.shape
    fx, fy = bd.fft.fftfreq(nx), bd.fft.fftfreq(ny)
    response = (2+bd.cos(2*bd.pi*fy))[:, None]*(2+bd.cos(2*bd.pi*fx))[None, :]/9
    coefficients = bd.fft.ifft2(bd.fft.fft2(bd.exp(1j*phase))/response)

    x, y = bd.asarray(x), bd.asarray(y)
    if x.ndim == 2:
        x = x[0, :]
    if y.ndim == 2:
        y = y[:, 0]
    col, row = (x/period % 1)*nx, (y/period % 1)*ny
    ix, iy = bd.floor(col).astype(int), bd.floor(row).astype(int)
    tx, ty = col-ix, row-iy

    def weights(t):
        return ((1-t)**3/6, (3*t**3-6*t*t+4)/6,
                (-3*t**3+3*t*t+3*t+1)/6, t**3/6)

    along_x = bd.zeros((ny, col.size), dtype=complex)
    for i, wx in enumerate(weights(tx)):
        along_x = along_x+wx[None, :]*coefficients[:, (ix+i-1) % nx]
    value = bd.zeros((row.size, col.size), dtype=complex)
    for j, wy in enumerate(weights(ty)):
        value = value+wy[:, None]*along_x[(iy+j-1) % ny, :]
    magnitude = bd.abs(value)
    safe_magnitude = bd.where(magnitude > 0, magnitude, 1)
    return bd.where(magnitude > 0, value/safe_magnitude, 1+0j)

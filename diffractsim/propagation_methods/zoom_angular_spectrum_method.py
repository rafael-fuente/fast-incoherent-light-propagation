"""Angular-spectrum propagation on chosen rectangular monitors.

The zoom propagator uses FFT/Bluestein evaluation, or delegates its angular
sampling to direct quadrature. The compact response below uses tiled
FFT/Bluestein monitors and stays here because its final propagation does
use Fourier transforms. All paths retain their original sampling rules.

References
----------
D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
SPIE (2011), doi:10.1117/3.858456.
Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
diffraction using the Bluestein method," Light: Science & Applications 9,
119 (2020), doi:10.1038/s41377-020-00362-z.
"""
import numpy as np
from collections import defaultdict
from scipy.fft import next_fast_len
from ..util import backend_functions
from ..util.bluestein_FFT import bluestein_fft
from .direct_quadrature_angular_spectrum_method import _angular_spectrum_amplitude_at

# Limit the temporary arrays used by one compact ASM block.
_MAX_SPECTRUM_SAMPLES = 6144
_MAX_MONITOR_SAMPLES = 3072


def zoom_angular_spectrum_method(field, z, x_interval, y_interval,
                                 sampling="fft", padding=1, Nx_mon=None, Ny_mon=None):
    """Propagate with the full angular-spectrum kernel to a rectangular monitor.
    
    Parameters
    ----------
    field: input MonochromaticField, including its pupil grid and wavelength.
    z: finite propagation distance, in micrometres; positive for angular sampling.
    x_interval: first and last horizontal monitor coordinates, in micrometres.
    y_interval: first and last vertical monitor coordinates, in micrometres.
    sampling: "fft" for FFT/Bluestein transforms; "angular" for direct
        finite-pupil spectrum integration along a complex contour.
    padding: positive integer refining FFT frequency spacing; must be 1
        with angular sampling.
    Nx_mon: horizontal monitor count, at least 2; None uses field.Nx.
    Ny_mon: vertical monitor count, at least 2; None uses field.Ny.
    
    Returns
    -------
    x, y: endpoint-inclusive monitor axes, in micrometres.
    E: complex propagated field of shape (len(y), len(x)).
    
    Notes
    -----
    FFT bandwidth is limited by the coarser input/output pitch. Padding
    refines frequency spacing and enlarges the periodic window, but cannot
    recover omitted frequencies. Angular sampling supports nonuniform pupil
    quadrature and wide monitors; contour orders require convergence checks
    in very near fields. Keep sampling, padding and array sizes fixed under JIT.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    """
    bd = backend_functions.backend
    if sampling not in ("fft", "angular"):
        raise ValueError('sampling must be "fft" or "angular".')
    if not isinstance(padding, (int, np.integer)) or padding < 1:
        raise ValueError("padding must be a positive integer.")
    if sampling == "angular" and padding != 1:
        raise ValueError('padding applies only to sampling="fft".')
    if min(field.Nx, field.Ny) < 2:
        raise ValueError("ASM zoom requires at least two samples per axis.")
    if len(x_interval) != 2 or len(y_interval) != 2:
        raise ValueError("Use two endpoints for each output interval.")
    for interval in (x_interval, y_interval):
        if all(np.isscalar(v) for v in interval):
            if not np.isfinite(interval).all() or interval[1] <= interval[0]:
                raise ValueError("Output intervals must be finite and increasing.")
    if backend_functions.backend_name == "jax":
        import jax
        traced = isinstance(z, jax.core.Tracer)
    else:
        traced = False
    if not traced and not np.isfinite(float(z)):
        raise ValueError("Propagation distance must be finite.")
    if sampling == "angular" and not traced and float(z) <= 0:
        raise ValueError("Angular sampling requires a positive distance.")

    Nx_mon = field.Nx if Nx_mon is None else Nx_mon
    Ny_mon = field.Ny if Ny_mon is None else Ny_mon
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2
           for n in (Nx_mon, Ny_mon)):
        raise ValueError("Monitor counts must be integers >= 2.")

    x = bd.linspace(x_interval[0], x_interval[1], Nx_mon)
    y = bd.linspace(y_interval[0], y_interval[1], Ny_mon)
    if sampling == "angular":
        xx, yy = bd.meshgrid(x, y)
        rho = bd.sqrt(z*z+xx*xx+yy*yy)
        amplitude = _angular_spectrum_amplitude_at(field, z, xx/rho, yy/rho,
            full_grid=True, dense=not hasattr(field, "pupil_quadrature"))
        E = z/(1j*rho**2)*bd.exp(2j*bd.pi*rho/field.λ)*amplitude
        return x, y, E
    if hasattr(field, "pupil_quadrature"):
        raise ValueError('Nonuniform pupil quadrature requires sampling="angular".')

    dx = (x_interval[1]-x_interval[0])/(Nx_mon-1)
    dy = (y_interval[1]-y_interval[0])/(Ny_mon-1)
    # Cover both grids: smaller monitors must not truncate the input FFT.
    nx, ny = padding*max(field.Nx, Nx_mon), padding*max(field.Ny, Ny_mon)
    sx, sy = bd.maximum(field.dx, dx), bd.maximum(field.dy, dy)
    fx = (bd.arange(nx)-nx//2)/(nx*sx)
    fy = (bd.arange(ny)-ny//2)/(ny*sy)
    fixed_grid = all(np.isscalar(v) for v in (dx, dy, field.dx, field.dy))

    # Use the ordinary FFT whenever these frequencies match the input grid.
    if fixed_grid and dx <= field.dx and dy <= field.dy:
        spectrum = bd.fft.fftshift(bd.fft.fft2(field.E, s=(ny, nx)))
    else:
        spectrum = bluestein_fft(field.E, 1, fx[0], fx[-1], 1/field.dx, nx)
        spectrum = bluestein_fft(spectrum, 0, fy[0], fy[-1], 1/field.dy, ny)
    spectrum *= field.dx*field.dy
    spectrum *= bd.exp(-2j*bd.pi*field.x[0]*fx)[None, :]
    spectrum *= bd.exp(-2j*bd.pi*field.y[0]*fy)[:, None]
    kz = 2*bd.pi*bd.sqrt((1/field.λ)**2-fx[None, :]**2-fy[:, None]**2+0j)
    spectrum *= bd.exp(1j*z*kz)

    # The complex square root above retains evanescent decay for positive z.
    if fixed_grid and dx >= field.dx and dy >= field.dy:
        spectrum *= bd.exp(2j*bd.pi*x[0]*fx)[None, :]
        spectrum *= bd.exp(2j*bd.pi*y[0]*fy)[:, None]
        E = bd.fft.ifft2(bd.fft.ifftshift(spectrum))/(dx*dy)
        E = E[:Ny_mon, :Nx_mon]
    else:
        dfx, dfy = 1/(nx*sx), 1/(ny*sy)
        E = bluestein_fft(spectrum, 1, -x[0], -x[-1], 1/dfx, Nx_mon)
        E = bluestein_fft(E, 0, -y[0], -y[-1], 1/dfy, Ny_mon)*dfx*dfy
        E *= bd.exp(2j*bd.pi*fx[0]*x)[None, :]
        E *= bd.exp(2j*bd.pi*fy[0]*y)[:, None]
    return x, y, E


def _transform(values, axis, start, stop, sampling, count):
    """Evaluate a Fourier transform over the requested frequency interval.
    
    Return ``count`` samples from ``start`` to ``stop`` along ``axis``.
    ``sampling`` is the input sampling frequency. A fast convolution length
    avoids padding every transform to the next power of two.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    """
    return bluestein_fft(values, axis, start, stop, sampling, count,
        fft_size=next_fast_len(values.shape[axis]+count-1))


def _tile_geometry(field, z, a, b, block, margin, refinement):
    """Divide the angular grid into blocks and choose their integration grids.
    
    Each block holds up to ``block`` samples per axis. Its monitor window and
    frequency interval include margins for diffraction and interpolation.
    Return those bounds, the monitor points and direction cosines, shared
    transform sizes, and the number of blocks along each axis.
    
    This uses only fixed geometry. The pupil values are evaluated separately,
    so changing the phase does not require changing these grids.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    """
    na, nb = (len(a)+block-1)//block, (len(b)+block-1)//block
    ap = np.pad(a, (0, na*block-len(a)), mode="edge")
    bp = np.pad(b, (0, nb*block-len(b)), mode="edge")
    points, geometry, sizes = [], [], []
    for j in range(nb):
        for i in range(na):
            aa, bb = np.meshgrid(ap[i*block:(i+1)*block], bp[j*block:(j+1)*block])
            c = np.sqrt(1-aa*aa-bb*bb)
            x, y = z*aa/c, z*bb/c
            xc, yc = (x.min()+x.max())/2, (y.min()+y.max())/2
            c0 = z/np.sqrt(z*z+xc*xc+yc*yc)
            axes, counts = [], []
            for p, centre, extent in ((x, xc, field.extent_x), (y, yc, field.extent_y)):
                half, radius = np.ptp(p)/2, extent/2
                inner = (half+radius)*c0/(z*field.λ)
                # Six samples per intensity fringe leave room for interpolation.
                h = 1/(6*margin*inner)
                lo, hi = p.min()-4*h, p.max()+4*h
                half = (hi-lo)/2
                inner = (half+radius)*c0/(z*field.λ)
                width = margin*inner
                reach = width*z*field.λ/c0**3+radius
                df = 1/(2*(half+reach+2*radius)*.7)
                axes.append([lo, hi, centre*c0/(z*field.λ), inner, width])
                counts.append([int(np.ceil(2*width/df*refinement))+1,
                               int(np.ceil((hi-lo)*6*width))+1])
            geometry.append(axes)
            sizes.append(counts)
            points.append(np.stack([x.ravel(), y.ravel(), c.ravel()]))
    sizes = np.max(sizes, axis=0)
    sizes = tuple(tuple(next_fast_len(max(16, int(n))) for n in axis) for axis in sizes)
    return np.asarray(geometry), np.asarray(points), sizes, (na, nb)


def _adaptive_asm_tiles(field, z, a, b, required, block, margin, refinement):
    """Split only the monitor regions whose transforms would be too large.
    
    Skip regions that cannot contribute to the requested irradiance. Trim
    empty edges before choosing a region's integration grid, and subdivide
    that region if it still exceeds the memory limits. Nearby grid sizes
    share a batch so JAX needs only a few compiled loops.
    
    Each batch contains window geometry, evaluation points, output indices
    and transform sizes. Padded output indices lie one past the response
    array; the evaluator discards them when assembling the result.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    """
    groups = defaultdict(list)

    def visit(i, j, size):
        """Keep a useful region if its transforms fit; otherwise split that region.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
        diffraction using the Bluestein method," Light: Science & Applications 9,
        119 (2020), doi:10.1038/s41377-020-00362-z.
        """
        rows, cols = np.nonzero(required[j:j+size, i:i+size])
        if not len(rows):
            return
        i0, i1 = i+cols.min(), i+cols.max()+1
        j0, j1 = j+rows.min(), j+rows.max()+1
        geometry, points, sizes, _ = _tile_geometry(
            field, z, a[i0:i1], b[j0:j1], size, margin, refinement)
        (nx, mx), (ny, my) = sizes
        if max(nx, ny) > _MAX_SPECTRUM_SAMPLES or max(mx, my) > _MAX_MONITOR_SAMPLES:
            if size <= 8:
                raise ValueError("This ASM region exceeds the compact memory budget; "
                                 "use a smaller pupil or a more distant monitor.")
            for dy in (0, size//2):
                for dx in (0, size//2):
                    visit(i+dx, j+dy, size//2)
            return
        indices = np.full((size, size), len(a)*len(b), dtype=np.int32)
        indices[:j1-j0, :i1-i0] = np.arange(j0, j1)[:, None]*len(a)+np.arange(i0, i1)
        key = (size, int(np.ceil(max(nx, ny)/1024)))
        groups[key].append((geometry[0], points[0], indices.ravel(), sizes))

    for j in range(0, len(b), block):
        for i in range(0, len(a), block):
            visit(i, j, block)
    for tiles in groups.values():
        geometry, points, indices, sizes = zip(*tiles)
        yield (np.asarray(geometry), np.asarray(points), np.asarray(indices),
               tuple(map(tuple, np.max(sizes, axis=0))))


def compact_angular_spectrum_response(field, z, a, b, *, margin=1.5, refinement=1,
                                     required=None):
    """Compute the coherent response needed for a bounded source convolution.
    
    Evaluate small overlapping monitor windows to avoid storing the full
    angular spectrum at once. Each window uses the full ASM propagation
    factor, including evanescent decay, and a tapered frequency interval.
    
    Parameters
    ----------
    field: unpropagated MonochromaticField on a uniform pupil grid.
    z: positive distance from the pupil to the monitor, in micrometres.
    a, b: fixed NumPy axes of horizontal and vertical direction cosines.
        Every pair must satisfy a² + b² < 1.
    margin: frequency-window margin, greater than 1; default 1.5.
        Increase it to check whether the window includes enough frequencies.
    refinement: frequency-sampling multiplier; default 1.
        Increase it to check the accuracy of the integration.
    required: optional Boolean mask of shape (len(b), len(a)). It marks every
        response sample needed by the source convolution and interpolation.
        Adaptive evaluation may skip samples outside this mask. None requests
        the entire angular rectangle.
    
    Returns
    -------
    K: weighted intensity of shape (len(b), len(a)). For c = sqrt(1-a²-b²),
        K = z² |U(z*a/c, z*b/c)|² / c⁴. This is the response that is convolved
        with the source; it is not yet the incoherent monitor irradiance.
    
    Notes
    -----
    IncoherentLightSimulation selects this helper for large angular ASM zooms
    with fixed geometry and source bounds. Pupil values remain differentiable;
    JAX recomputes each block during backpropagation to limit memory use.
    The integration controls above do not change source or monitor sampling.
    Small calculations keep their original uniform blocks. If those blocks
    are too large, only the demanding regions are subdivided.
    
    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    """
    from .direct_quadrature_angular_spectrum_method import sample_analytic_spectrum
    bd = backend_functions.backend
    if max(abs(a))**2+max(abs(b))**2 >= 1:
        raise ValueError("Compact ASM quadrature requires propagating directions.")
    block = min(256, max(len(a), len(b)))
    geometry, points, sizes, (na, nb) = _tile_geometry(
        field, float(z), a, b, block, margin, refinement)
    (nx, mx), (ny, my) = sizes
    adaptive = max(nx, ny) > _MAX_SPECTRUM_SAMPLES or max(mx, my) > _MAX_MONITOR_SAMPLES
    if adaptive:
        required = np.ones((len(b), len(a)), dtype=bool) if required is None else np.asarray(required)
        if required.shape != (len(b), len(a)) or required.dtype != bool:
            raise ValueError("required must be a Boolean mask matching the angular grid.")
        # Powers of two ensure subdivision also covers odd-sized angular grids.
        block = 1 << (block-1).bit_length()
        batches = _adaptive_asm_tiles(field, float(z), a, b, required, block, margin, refinement)
        del geometry, points
    else:
        batches = [(geometry, points, None, sizes)]

    def evaluate_tile(E, data):
        """Propagate the current pupil to one window and return its weighted intensity.
        
        References
        ----------
        D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
        SPIE (2011), doi:10.1117/3.858456.
        Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
        diffraction using the Bluestein method," Light: Science & Applications 9,
        119 (2020), doi:10.1038/s41377-020-00362-z.
        """
        bounds, point = data
        lx, hx, cx, inner_x, width_x = bounds[0]
        ly, hy, cy, inner_y, width_y = bounds[1]
        fx, fy = bd.linspace(cx-width_x, cx+width_x, nx), bd.linspace(cy-width_y, cy+width_y, ny)
        dfx, dfy = (2*width_x)/(nx-1), (2*width_y)/(ny-1)
        A = _transform(E, 1, fx[0], fx[-1], 1/field.dx, nx)
        A = _transform(A, 0, fy[0], fy[-1], 1/field.dy, ny)
        A *= field.dx*field.dy
        A *= bd.exp(-2j*bd.pi*field.x[0]*fx)[None, :]
        A *= bd.exp(-2j*bd.pi*field.y[0]*fy)[:, None]
        wx = bd.where(abs(fx-cx) <= inner_x, 1.,
            .5*(1+bd.cos(bd.pi*(abs(fx-cx)-inner_x)/(width_x-inner_x))))
        wy = bd.where(abs(fy-cy) <= inner_y, 1.,
            .5*(1+bd.cos(bd.pi*(abs(fy-cy)-inner_y)/(width_y-inner_y))))
        longitudinal = bd.sqrt(field.λ**-2-fx[None, :]**2-fy[:, None]**2+0j)
        A *= bd.exp(2j*bd.pi*z*longitudinal)*wy[:, None]*wx[None, :]
        U = _transform(A, 1, -lx, -hx, 1/dfx, mx)
        U = _transform(U, 0, -ly, -hy, 1/dfy, my)*dfx*dfy
        # The remaining inverse-transform factors have unit modulus.
        sx, sy = (hx-lx)/(mx-1), (hy-ly)/(my-1)
        H = sample_analytic_spectrum(bd.abs(U)**2,
            (point[0]-lx)/sx, (point[1]-ly)/sy, 0., 1., order=6)
        return H*z*z/point[2]**4

    if adaptive:
        pieces, offsets = [bd.zeros(1)], np.zeros(len(a)*len(b), dtype=np.int32)
        start = 1
    if backend_functions.backend_name == "jax":
        import jax
    for geometry, points, indices, sizes in batches:
        (nx, mx), (ny, my) = sizes
        if backend_functions.backend_name == "jax":
            @jax.checkpoint
            def evaluate(_, data):
                """Recompute one block during differentiation to limit memory use.
                
                References
                ----------
                D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
                SPIE (2011), doi:10.1117/3.858456.
                Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
                diffraction using the Bluestein method," Light: Science & Applications 9,
                119 (2020), doi:10.1038/s41377-020-00362-z.
                """
                return None, evaluate_tile(field.E, data)
            _, values = jax.lax.scan(evaluate, None, (bd.asarray(geometry), bd.asarray(points)))
        else:
            values = bd.stack([evaluate_tile(field.E, (bd.asarray(g), bd.asarray(p)))
                               for g, p in zip(geometry, points)])
        if not adaptive:
            values = values.reshape(nb, na, block, block).transpose(0, 2, 1, 3)
            return values.reshape(nb*block, na*block)[:len(b), :len(a)]
        # Work out the fixed assembly order in NumPy, outside the JAX graph.
        indices = indices.ravel()
        keep = indices < offsets.size
        offsets[indices[keep]] = start+np.flatnonzero(keep)
        pieces.append(values.ravel())
        start += values.size
    values = bd.concatenate(pieces)
    if backend_functions.backend_name == "jax":
        # Every index is already valid. A direct gather avoids compiling
        # millions of constant bounds checks for the assembled monitor.
        dims = jax.lax.GatherDimensionNumbers(offset_dims=(), collapsed_slice_dims=(0,),
                                              start_index_map=(0,))
        result = jax.lax.gather(values, bd.asarray(offsets[:, None]), dims, (1,),
                                mode="promise_in_bounds")
    else:
        result = values[bd.asarray(offsets)]
    return result.reshape(len(b), len(a))

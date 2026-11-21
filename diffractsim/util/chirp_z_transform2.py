from .backend_functions import backend as bd
from .backend_functions import backend_name

def chirpz(x, A, W, M, fft_size=None):
    """Evaluate the z-transform along the last axis at M points on a contour.

    The points are z_k = A * W**(-k), where k runs from 0 to M-1. The
    transform at each point is the sum of x_n * z_k**(-n). An FFT convolution
    computes all these sums together.

    Parameters
    ----------
    x: input samples along the last axis; other axes are treated as batches.
    A: complex starting point of the contour.
    W: complex factor controlling its spacing and shape.
    M: number of output samples.
    fft_size: optional convolution length, at least x.shape[-1] + M - 1.
        None pads to a power of two. Choosing another valid length changes
        only the internal padding.

    Reference
    ---------
    Rabiner, L. R., R. W. Schafer and C. M. Rader, “The Chirp z-Transform
    Algorithm,” IEEE Transactions on Audio and Electroacoustics,
    AU-17(2):86–92, 1969.

    Originally written by Stefan van der Walt:
    http://www.mail-archive.com/numpy-discussion@scipy.org/msg01812.html
    """
    global bd
    global backend_name

    from .backend_functions import backend as bd
    from .backend_functions import backend_name

    if backend_name == 'jax': 
        import jax
        if jax.config.jax_enable_x64:
            complex_ = bd.complex128
        else:
            complex_ = bd.complex64
    else:
        complex_ = complex

    x = bd.asarray(x, dtype=complex_)
    P = x.shape

    N = P[-1]
    L = 1 << (M + N - 2).bit_length() if fft_size is None else fft_size
    if L < M+N-1:
        raise ValueError("fft_size must cover the linear chirp convolution.")

    n = bd.arange(N, dtype=float)
    y = bd.power(A, -n) * bd.power(W, n ** 2 / 2.)
    y = y[None, :] * x
    Y = bd.fft.fft(y, L)

    n = bd.arange(L, dtype=float)
    v = bd.zeros(L, dtype=complex_)
    if backend_name == 'jax':
        v = v.at[:M].set(bd.power(W, -n[:M] ** 2 / 2.))
        v = v.at[L-N+1:].set(bd.power(W, -(L - n[L-N+1:]) ** 2 / 2.))
    else:
        v[:M] = bd.power(W, -n[:M] ** 2 / 2.)
        v[L-N+1:] = bd.power(W, -(L - n[L-N+1:]) ** 2 / 2.)

    V = bd.fft.fft(v)

    g = bd.fft.ifft(V[None, :] * Y)[..., :M]
    k = bd.arange(M)
    g = g * bd.power(W, k ** 2 / 2.)[None, :]

    # Return result
    return g

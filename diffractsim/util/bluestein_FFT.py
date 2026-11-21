from .backend_functions import backend as bd
from .chirp_z_transform import chirpz


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


def bluestein_fft(x, axis, f0, f1, fs, M, fft_size=None):
    """Evaluate a Fourier transform at evenly spaced frequencies of your choice.

    The output has M samples from f0 to f1 along the selected axis. The
    Bluestein algorithm evaluates these samples using an FFT convolution,
    so the output length can differ from the input length.

    Parameters
    ----------
    x: input array.
    axis: axis along which to transform the samples.
    f0, f1: first and last output frequencies, in the same units as fs.
    fs: input sampling frequency.
    M: number of output frequencies, at least 2.
    fft_size: optional internal FFT length, at least x.shape[axis] + M - 1.
        None uses the original power-of-two padding. A supplied length
        changes the padding, not the requested transform or output grid.

    Reference
    ---------
    Leo I. Bluestein, “A linear filtering approach to the computation of the
    discrete Fourier transform,” Northeast Electronics Research and
    Engineering Meeting Record 10, 218–219 (1968).
    """
    global bd
    from .backend_functions import backend as bd

    # Swap axes
    x = bd.swapaxes(a=x, axis1=axis, axis2=-1)

    # Normalize frequency range
    phi0 = 2.0 * bd.pi * f0 / fs
    phi1 = 2.0 * bd.pi * f1 / fs
    d_phi = (phi1 - phi0) / (M - 1)

    # Determine shape of signal
    A = bd.exp(1j * phi0)
    W = bd.exp(-1j * d_phi)
    X = chirpz(x=x, A=A, W=W, M=M, fft_size=fft_size)

    return bd.swapaxes(a=X, axis1=axis, axis2=-1)


def bluestein_fft2(U, fx0, fx1, fxs, fy0, fy1, fys, Nx=None, Ny=None):
    """
    Evaluate the 2DFT on optional independent output counts Nx and Ny.

    Parameters
    ----------

    U: array to evaluate 2DFT

    fx0: lower bound of x frequency bandwidth
    fx1: upper bound of x frequency bandwidth
    fxs: sampling x frequency

    fy0: lower bound of y frequency bandwidth
    fy1: upper bound of y frequency bandwidth
    fys: sampling y frequency


    """
    Ny = U.shape[0] if Ny is None else Ny
    Nx = U.shape[1] if Nx is None else Nx
    return bluestein_fft( bluestein_fft(U, f0=fy0, f1=fy1, fs=fys, M=Ny, axis=0), f0=fx0, f1=fx1, fs=fxs, M=Nx, axis=1)





def bluestein_ifft(X, axis, f0, f1, fs, M):
    """
    bluestein iFFT function to evaluate the iDFT
    coefficients for the rows of an array in the frequency range [f0, f1]
    using M output points.
    
    Parameters
    ----------

    x: array to evaluate iDFT (along last dimension of array)
    f0: lower bound of frequency bandwidth
    f1: upper bound of frequency bandwidth
    fs: sampling frequency
    M: number of points used when evaluating the iDFT (N <= signal length)
    axis: axis along which the ifft's are computed (defaults to last axis)

    """
    global bd
    from .backend_functions import backend as bd

    # Swap axes
    X = bd.swapaxes(a=X, axis1=axis, axis2=-1)

    N = X.shape[-1]

    phi0 = f0 / fs * 2.0 * bd.pi / N
    phi1 = f1 / fs * 2.0 * bd.pi / N
    d_phi = (phi1 - phi0) / (M - 1)

    A = bd.exp(-1j * phi0)
    W = bd.exp(1j * d_phi)
    x = chirpz(x=X, A=A, W=W, M=M) / N

    return bd.swapaxes(a=x, axis1=axis, axis2=-1)

def bluestein_ifft2(U, fx0, fx1, fxs,   fy0, fy1, fys):
    """
    bluestein iFFT function to evaluate the i2DFT

    Parameters
    ----------

    U: array to evaluate 2DFT

    fx0: lower bound of x frequency bandwidth
    fx1: upper bound of x frequency bandwidth
    fxs: sampling x frequency

    fy0: lower bound of y frequency bandwidth
    fy1: upper bound of y frequency bandwidth
    fys: sampling y frequency


    """
    Ny, Nx = U.shape
    return bluestein_ifft( bluestein_ifft(U, f0=fy0, f1=fy1, fs=fys, M=Ny, axis=0), f0=fx0, f1=fx1, fs=fxs, M=Nx, axis=1)



def bluestein_fftfreq(f0, f1, M):
    """
    Return frequency values of the bluestein FFT
    coefficients returned by bluestein_fft().
    
    Parameters
    ----------

    f0: lower bound of frequency bandwidth
    f1: upper bound of frequency bandwidth
    fs: sampling rate
    
    """
    global bd
    from .backend_functions import backend as bd

    df = (f1 - f0) / (M - 1)
    return bd.arange(M) * df + f0


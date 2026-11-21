import numpy as np
from ..util.backend_functions import backend as bd
from ..util.bluestein_FFT import bluestein_fft2, bluestein_fftfreq

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def zoom_fresnel_method(simulation, E, z, λ, x_interval, y_interval, Nx_mon=None, Ny_mon=None):
    """Evaluate paraxial Fresnel propagation on a chosen rectangular monitor.

    A pupil chirp turns the Fresnel integral into a Fourier transform.
    Bluestein's algorithm evaluates that transform over the requested
    interval and sample count. This uses the paraxial approximation;
    changing the transform algorithm does not remove that approximation.

    Parameters
    ----------
    simulation: input field grid.
    E: complex field samples on that grid.
    z: nonzero propagation distance in micrometres.
    λ: wavelength in micrometres.
    x_interval, y_interval: first and last monitor coordinates in micrometres.
    Nx_mon, Ny_mon: output sample counts, at least 2; None keeps the input count.

    Returns
    -------
    x, y, E: endpoint-inclusive axes and propagated complex field.

    References
    ----------
    Y. Hu et al., "Efficient full-path optical calculation of scalar and vector
    diffraction using the Bluestein method," Light: Science & Applications 9,
    119 (2020), doi:10.1038/s41377-020-00362-z.
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    """

    global bd
    from ..util.backend_functions import backend as bd

    
    Nx_mon = simulation.Nx if Nx_mon is None else Nx_mon
    Ny_mon = simulation.Ny if Ny_mon is None else Ny_mon
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2
           for n in (Nx_mon, Ny_mon)):
        raise ValueError("Monitor counts must be integers >= 2.")

    E = bluestein_fft2(E * bd.exp(1j * 2*bd.pi/λ /(2*z) *(simulation.xx**2 + simulation.yy**2)), 
                        x_interval[0] / (z*λ), x_interval[1] / (z*λ), 1/simulation.dx, 
                        y_interval[0] / (z*λ), y_interval[1] / (z*λ), 1/simulation.dy, Nx=Nx_mon, Ny=Ny_mon)

    fx_zfft = bluestein_fftfreq(x_interval[0]/ (z*λ),x_interval[1]/ (z*λ), Nx_mon)
    fy_zfft = bluestein_fftfreq(y_interval[0]/ (z*λ),y_interval[1]/ (z*λ), Ny_mon)

    fxx, fyy = bd.meshgrid(fx_zfft, fy_zfft)
    x_shift = simulation.x[0]
    y_shift = simulation.y[0]
    factor = (simulation.dx*simulation.dy * bd.exp(-1j*2*bd.pi*x_shift*fxx -1j*2*bd.pi*y_shift*fyy))

    x = fx_zfft*(z*λ)
    y = fy_zfft*(z*λ)

    xx, yy = bd.meshgrid(x, y)
    return x,y, E*factor * bd.exp(1j*bd.pi/(λ*z)  * (xx**2 + yy**2)  +   1j*2*bd.pi/λ * z ) / (1j*z*λ)



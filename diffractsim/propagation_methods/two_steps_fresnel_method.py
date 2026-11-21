import numpy as np
from ..util.backend_functions import backend as bd

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def two_steps_fresnel_method(simulation, E, z, λ, scale_factor):
    """Propagate with two-step Fresnel diffraction to a scaled grid.

    The two chirps and Fourier transforms change the grid width and height
    by scale_factor while preserving the sample counts. This uses the
    paraxial approximation. E is the input field; z (nonzero) and wavelength
    λ are in micrometres. Return the new x and y axes and complex field.
    The input simulation is not modified.

    Use MonochromaticField.propagate(..., model="Fresnel") to select this
    method. Use zoom_propagate when monitor bounds or counts need to vary
    independently, or model="ASM" when the paraxial approximation is unsuitable.

    References
    ----------
    D. G. Voelz, Computational Fourier Optics: A MATLAB Tutorial,
    SPIE (2011), doi:10.1117/3.858456.
    """
    global bd
    from ..util.backend_functions import backend as bd


    L1 = simulation.extent_x
    L2 = simulation.extent_x*scale_factor


    fft_E = bd.fft.fftshift(bd.fft.fft2(E * bd.exp(1j * np.pi/(z * λ) * (L1-L2)/L1 * (simulation.xx**2 + simulation.yy**2) )  ))
    fx = bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.dx))
    fy = bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.dy))
    fx, fy = bd.meshgrid(fx, fy)

    E = bd.fft.ifft2(bd.fft.ifftshift( bd.exp(- 1j * np.pi * λ * z * L1/L2 * (fx**2 + fy**2))  *  fft_E) )

    E = L1/L2 * bd.exp(1j * 2*np.pi/λ * z   - 1j * np.pi/(z * λ)* (L1-L2)/L2 * ((simulation.xx*scale_factor)**2 + (simulation.yy*scale_factor)**2)) * E
    return simulation.x*scale_factor,  simulation.y*scale_factor, E

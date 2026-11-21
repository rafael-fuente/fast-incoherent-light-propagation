import numpy as np
import diffractsim

diffractsim.set_backend("CUDA")  # Change to "CPU" for the NumPy backend.

from diffractsim import PolychromaticSimulator, Lens, load_image_as_function, cf, cm, um

"""
MPL 2.0 License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

z0 = 40*cm  # Distance from the source plane to the lens.
zi = 40*cm  # Distance from the lens to the image plane.
pupil_radius = 20*cm
focal_length = z0*zi/(z0+zi)
object_width = 14*um
M = -zi/z0
image_halfwidth = abs(M)*object_width/2

# Each point of the horse emits independently. Keep the original image's
# amplitude convention by squaring its grayscale values to obtain intensity.
object_amplitude = load_image_as_function("./apertures/horse.png",
    x_interval=(-object_width/2, object_width/2),
    y_interval=(-object_width/2, object_width/2))
# Convert the original object-irradiance scale to radiance through the pupil.
pupil_solid_angle = np.pi*(pupil_radius/z0)**2

def source_radiance(x, y):
    """Radiance of independently emitting points in the source plane."""
    return object_amplitude(x, y)**2/pupil_solid_angle


def optical_setup(F):
    F = F.add(Lens(f=focal_length, radius=pupil_radius))
    # Leave space around the image for the diffraction tails in the convolution.
    return F.zoom_propagate(zi,
        x_interval=[-2*image_halfwidth, 2*image_halfwidth],
        y_interval=[-2*image_halfwidth, 2*image_halfwidth],
        model="Fresnel", sampling="fft")


sim = PolychromaticSimulator(
    spectrum=2*M**2*cf.illuminant_d65,
    extent_x=2.2*pupil_radius, extent_y=2.2*pupil_radius,
    Nx=1024, Ny=1024,
    spectrum_size=180, spectrum_divisions=30,
    optical_setup=optical_setup,
)
# Include the source distance and wavefront curvature at every wavelength.
sim.set_illumination_from_plane(source_radiance, distance=z0, paraxial=True)

rgb = sim.get_colors()
sim.plot_colors(rgb, figsize=(5, 5), units=um,
    xlim=[-image_halfwidth, image_halfwidth], ylim=[-image_halfwidth, image_halfwidth])

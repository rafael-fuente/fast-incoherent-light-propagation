import diffractsim
diffractsim.set_backend("CUDA")

from diffractsim import PolychromaticSimulator, ApertureFromImage, CircularAperture, cf, nm, mm, cm



zi = 50*cm # distance from the image plane to the exit pupil
z0 = 50*cm # distance from the exit pupil to the current simulation plane
M = -zi/z0 # magnification factor

def optical_setup(F):
    F = F.add(ApertureFromImage( "./apertures/horse.png",  image_size=(1.0 * mm, 1.0 * mm), simulation = F))


    F = F.propagate_to_image_plane(pupil = CircularAperture(radius = 6*mm) ,M = M,  zi = zi, z0 = z0)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum=2 * cf.illuminant_d65, extent_x= 1.5 * mm, extent_y= 1.5 * mm, Nx=2048, Ny=2048,
    spectrum_size = 180, spectrum_divisions = 30
)

rgb = F.get_colors()

F.plot_colors(rgb, figsize=(5, 5), xlim=[-0.4*mm,0.4*mm], ylim=[-0.4*mm,0.4*mm])

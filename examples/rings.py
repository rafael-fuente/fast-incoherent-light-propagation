import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticSimulator,ApertureFromImage, cf, mm, cm

def optical_setup(F):
    F = F.add(ApertureFromImage("./apertures/rings.jpg", image_size = (12.0 * mm,12.0 * mm), simulation = F))


    F = F.propagate(z=150*cm)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=25.0 * mm,
    extent_y=25.0 * mm,
    Nx=1500,
    Ny=1500,
    spectrum_size = 200, spectrum_divisions = 40  # increase these values to improve color resolution
)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm, 8*mm], ylim=[-8*mm, 8*mm])

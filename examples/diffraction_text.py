import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticSimulator,ApertureFromImage, cf, mm, cm

def optical_setup(F):
    F = F.add(ApertureFromImage("./apertures/diffraction_text.jpg", image_size=(15 * mm, 15 * mm), simulation = F))


    F = F.propagate(z=150*cm)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=20 * mm,
    extent_y=20 * mm,
    Nx=1600,
    Ny=1600,
)

rgb = F.get_colors()

F.plot_colors(rgb, xlim=[-10*mm, 10*mm], ylim=[-10*mm, 10*mm])

import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticSimulator, ApertureFromImage, cf, mm, cm

def optical_setup(F):
    F = F.add(ApertureFromImage("./apertures/rectangular_grating.jpg",image_size = (1.2 * mm,1.2 * mm), simulation = F))

    F = F.propagate(100*cm)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum=10 * cf.illuminant_d65,
    extent_x=30.0 * mm,
    extent_y=30.0 * mm,
    Nx=2048,
    Ny=2048,
)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm, 8*mm], ylim=[-8*mm, 8*mm])

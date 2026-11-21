import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticSimulator,ApertureFromImage, cf, mm, cm

def optical_setup(F):
    F = F.add(ApertureFromImage("./apertures/hexagon.jpg", image_size=(5.6 * mm, 5.6 * mm), simulation = F))

    F = F.propagate(z=80*cm)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum=2 * cf.illuminant_d65, extent_x=18 * mm, extent_y=18 * mm, Nx=1024, Ny=1024
)

rgb =F.get_colors()
F.plot_colors(rgb, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])

import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticSimulator, Lens, ApertureFromImage, cf, nm, mm, cm

def optical_setup(F):
    F = F.add(ApertureFromImage("./apertures/bahtinov_mask.jpg", image_size=(5. * mm, 5 * mm), simulation = F))

    F = F.add(Lens(f = 30*cm))
    F = F.propagate(z=30*cm)
    return F

F = PolychromaticSimulator(
    optical_setup=optical_setup,
    spectrum = 4*cf.illuminant_d65, 
    extent_x=15. * mm, extent_y=15. * mm, 
    Nx=1500, Ny=1500
)

rgb =F.get_colors()
F.plot_colors(rgb, xlim=[-6* mm, 6* mm], ylim=[-6* mm, 6* mm])

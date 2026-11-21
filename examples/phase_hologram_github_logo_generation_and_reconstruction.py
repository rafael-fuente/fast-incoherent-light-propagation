import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import SLM, load_phase_as_function, MonochromaticField, Lens, mm, nm, cm, FourierPhaseRetrieval


# Generate a Fourier plane phase hologram
PR = FourierPhaseRetrieval(target_amplitude_path = './apertures/github_logo.png', new_size= (400,400), pad = (200,200))
PR.retrieve_phase_mask(max_iter = 200, method = 'Conjugate-Gradient')
PR.save_retrieved_phase_as_image('github_logo_phase_hologram.png')
PR.save_retrieved_phase_as_file('github_logo_phase_hologram.npy')


#Add a plane wave
F = MonochromaticField(
    wavelength=532 * nm, extent_x=30 * mm, extent_y=30 * mm, Nx=2400, Ny=2400, intensity = 0.005
)


# add a Spatial Light Modulator
F = F.add(SLM(
      phase_mask_function = load_phase_as_function("github_logo_phase_hologram.npy", x_interval = (-5*mm, +5*mm),   y_interval = (-5*mm, +5*mm)), 
      pixel_dx = 9.99*mm/PR.Nx, pixel_dy = 9.99*mm/PR.Ny, Nx = PR.Nx, Ny = PR.Ny, 
      simulation = F)
)



# plot colors at z = 0
rgb = F.get_field()
F.plot_phase(rgb)



# propagate field to Fourier plane
F = F.add(Lens(f = 80*cm))
F = F.propagate(80*cm)


# plot colors (reconstructed image) at z = 80*cm (Fourier plane)
rgb = F.get_colors()
F.plot_colors(rgb)

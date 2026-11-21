"""Design a periodic phase cell for a chosen distribution of order powers."""

import numpy as np


def phase_grating_design(target, Nx=128, Ny=None, *, iterations=2000,
                         learning_rate=0.03, weight=0.5, seed=0,
                         initial_phase=None):
    """
    Optimize a periodic phase cell for chosen relative diffraction-order powers.

    Parameters
    ----------
    target: function target(m, n), or centred (Ny, Nx) array, giving finite
        nonnegative relative powers on integer orders. Positive values select
        orders and set their ratios; zeros exclude orders from the shape loss.
    Nx: horizontal number of phase samples, at least 2.
    Ny: vertical number of phase samples, at least 2; None uses Nx.
    iterations: positive number of Adam steps.
    learning_rate: initial positive Adam step in radians; cosine decay
        approaches 5% of this value.
    weight: nonnegative weight of the power-ratio error relative to efficiency.
    seed: random seed for initialization; does not change NumPy's global RNG.
    initial_phase: optional two-dimensional phase in radians; periodically
        resized to (Ny, Nx) if necessary. When supplied, seed is unused.

    Returns
    -------
    phase: JAX array of shape (Ny, Nx), in radians, with the lowest loss found.
    history: JAX array with one pre-update loss per iteration.

    Notes
    -----
    Select the JAX backend before calling. The incident amplitude is one.
    The loss on selected orders is -efficiency + weight * mean((power /
    mean(power) - target / mean(target))²). Equal target powers reduce this
    to -efficiency + weight * CV².

    Powers are Fourier-coefficient efficiencies. Choose a physical period
    that makes the desired orders propagate, then use get_modes_power for
    powers in the library power unit; divide them by W for watts. Initialization uses random target-coefficient phases
    or interpolates the supplied complex phasor. Adam restarts each call
    with beta1=0.9, beta2=0.999 and epsilon=1e-10. Target selection and grid
    sizes are fixed; the phase objective is differentiated by JAX.

    Examples
    --------
    phase, history = phase_grating_design(
        lambda m, n: (abs(m) <= 1) & (abs(n) <= 1), Nx=128)
    """
    from ..util.backend_functions import backend_name
    if backend_name != "jax":
        raise RuntimeError('Select diffractsim.set_backend("JAX") before designing a grating.')
    import jax
    import jax.numpy as jnp
    from ..util.phase_grating import periodic_phase_grating

    Ny = Nx if Ny is None else Ny
    if Nx < 2 or Ny < 2 or iterations < 1 or learning_rate <= 0 or weight < 0:
        raise ValueError("Use grid sizes >= 2, positive iterations/rate, and nonnegative weight.")
    m, n = np.meshgrid(np.arange(Nx)-Nx//2, np.arange(Ny)-Ny//2)
    target = np.asarray(target(m, n) if callable(target) else target, dtype=float)
    if target.shape != (Ny, Nx) or not np.isfinite(target).all() or np.any(target < 0) or not np.any(target > 0):
        raise ValueError("target must give a finite, nonnegative (Ny, Nx) array with at least one positive order.")
    selected = target > 0
    rows, cols = n[selected] % Ny, m[selected] % Nx
    desired = jnp.asarray(target[selected]/target[selected].mean())

    if initial_phase is None:
        coefficients = np.zeros((Ny, Nx), dtype=complex)
        angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, selected.sum())
        coefficients[rows, cols] = np.sqrt(target[selected])*np.exp(1j*angles)
        phase = jnp.asarray(np.angle(np.fft.ifft2(coefficients)))
    else:
        phase = jnp.asarray(initial_phase, dtype=float)
        if phase.ndim != 2:
            raise ValueError("initial_phase must be a two-dimensional array in radians.")
        if phase.shape != (Ny, Nx):
            phase = jnp.angle(periodic_phase_grating(
                phase, jnp.arange(Nx)/Nx, jnp.arange(Ny)/Ny, period=1))

    def loss_function(phase):
        coefficients = jnp.fft.fft2(jnp.exp(1j*phase))/phase.size
        power = jnp.abs(coefficients[rows, cols])**2
        efficiency = power.sum()
        error = jnp.mean((power/jnp.maximum(efficiency/power.size, 1e-30)-desired)**2)
        return -efficiency + weight*error

    value_and_gradient = jax.value_and_grad(loss_function)

    def step(carry, k):
        phase, first, second, best_phase, best_loss = carry
        loss, gradient = value_and_gradient(phase)
        best_phase = jnp.where(loss < best_loss, phase, best_phase)
        best_loss = jnp.minimum(loss, best_loss)
        first = .9*first + .1*gradient
        second = .999*second + .001*gradient**2
        first_hat, second_hat = first/(1-.9**(k+1)), second/(1-.999**(k+1))
        rate = learning_rate*(.05+.95*.5*(1+jnp.cos(jnp.pi*k/iterations)))
        phase = phase-rate*first_hat/(jnp.sqrt(second_hat)+1e-10)
        return (phase, first, second, best_phase, best_loss), loss

    @jax.jit
    def optimize(phase):
        carry = (phase, jnp.zeros_like(phase), jnp.zeros_like(phase), phase, jnp.inf)
        (phase, _, _, best_phase, best_loss), history = jax.lax.scan(step, carry, jnp.arange(iterations))
        best_phase = jnp.where(loss_function(phase) < best_loss, phase, best_phase)
        return best_phase, history

    return optimize(phase)

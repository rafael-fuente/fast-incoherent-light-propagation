"""Plot discrete diffraction orders as squares separated by a grid."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..util.constants import W


def plot_grating_modes(self, power, *, ax=None, xlim=None, ylim=None,
                       title="Diffraction-order powers", colorbar_label="Power [W]",
                       figsize=(5, 4), cmap="viridis", vmin=0, vmax=None, power_units=W):
    """
    Plot diffraction-order powers as squares surrounded by a grid.

    Parameters
    ----------
    power: centred (Ny, Nx) array from get_modes_power, in picowatts.
        The default display converts these values to watts.
    ax: Matplotlib axes; None creates and shows a figure.
    xlim: horizontal limits in order coordinates; half-integers such as
        (-2.5, 2.5) include whole squares.
    ylim: vertical limits in order coordinates, with the same convention.
    title: plot title.
    colorbar_label: label for the supplied values, default "Power [W]";
        change it when displaying percentages or other rescaled powers.
    power_units: positive divisor for displayed powers, default W. Use
        power_units=1 when supplying percentages or already normalized values.
    figsize: figure width and height in inches, when ax is None.
    cmap: Matplotlib colormap name or object.
    vmin: lower color limit in displayed units, default 0.
    vmax: upper color limit in displayed units; None uses the data maximum.

    Returns
    -------
    ax: Matplotlib axes with integer order centres and half-integer grid borders.
    """
    from ..util.backend_functions import backend_name
    if backend_name == "cupy":
        import cupy as cp
        power = cp.asnumpy(power)
    else:
        power = np.asarray(power)
    if power.shape != (self.Ny, self.Nx):
        raise ValueError("power must have shape (Ny, Nx), as returned by get_modes_power.")

    if not np.isscalar(power_units) or not np.isfinite(power_units) or power_units <= 0:
        raise ValueError("power_units must be a positive finite scalar.")
    power = power/power_units

    m_edges = np.arange(self.Nx+1)-self.Nx//2-0.5
    n_edges = np.arange(self.Ny+1)-self.Ny//2-0.5
    xlim = (m_edges[0], m_edges[-1]) if xlim is None else xlim
    ylim = (n_edges[0], n_edges[-1]) if ylim is None else ylim
    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    im = ax.imshow(power, origin="lower", interpolation="nearest", aspect="equal",
        extent=(m_edges[0], m_edges[-1], n_edges[0], n_edges[-1]),
        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, min_n_ticks=1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True, min_n_ticks=1))
    ax.set_xticks(m_edges[(m_edges >= xlim[0]) & (m_edges <= xlim[1])], minor=True)
    ax.set_yticks(n_edges[(n_edges >= ylim[0]) & (n_edges <= ylim[1])], minor=True)
    ax.grid(False, which="major")
    ax.grid(True, which="minor", color="0.5", linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(False)
    ax.set(title=title, xlabel="Order m", ylabel="Order n", xlim=xlim, ylim=ylim)
    ax.figure.colorbar(im, ax=ax, label=colorbar_label)
    if owns_figure:
        plt.show()
    return ax

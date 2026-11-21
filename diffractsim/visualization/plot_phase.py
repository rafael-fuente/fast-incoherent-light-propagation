import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import m, cm, mm, um, nm

"""
MPL 2.0 License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


def plot_phase(self, E, figsize=(7, 6), xlim=None, ylim=None, grid=False,
               text=None, max_val=0.5, units=mm, dark_background=False,
               *, cmap="hsv", ax=None):
    """
    Plot the phase of a complex field, wrapped to [-pi, pi].

    Parameters
    ----------
    E: complex field array of shape (Ny, Nx) on this simulation's grid.
    figsize: figure width and height in inches, when ax is None.
    xlim: optional horizontal limits, in micrometres.
    ylim: optional vertical limits, in micrometres.
    grid: show grid lines if True.
    text: plot title; None uses the simulation's propagation distance.
    max_val: positive amplitude threshold as a fraction of peak amplitude;
        values below it fade to black, default 0.5.
    units: positive divisor for displayed coordinates, default mm.
    dark_background: use a dark figure style when creating a new figure.
    cmap: Matplotlib colormap name or object, default "hsv".
    ax: Matplotlib axes; None creates and shows a figure. With an existing
        axes, figsize and dark_background do not change its figure.

    Returns
    -------
    ax: Matplotlib axes containing the phase and its colorbar in radians.
    """
    from ..util.backend_functions import backend_name
    E = E.get() if backend_name == "cupy" else np.asarray(E)
    if max_val <= 0:
        raise ValueError("max_val must be positive.")
    amplitude = np.abs(E)
    peak = amplitude.max()
    alpha = np.clip(amplitude/(max_val*peak), 0, 1) if peak > 0 else np.zeros_like(amplitude)
    extent = [float(self.x[0]-self.dx/2)/units, float(self.x[-1]+self.dx/2)/units,
              float(self.y[0]-self.dy/2)/units, float(self.y[-1]+self.dy/2)/units]

    owns_figure = ax is None
    style = ("dark_background" if dark_background else "default") if owns_figure else {}
    with plt.style.context(style):
        if owns_figure:
            _, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.set_facecolor("black")
        im = ax.imshow(np.angle(E), cmap=cmap, vmin=-np.pi, vmax=np.pi, alpha=alpha,
                       extent=extent, origin="lower", interpolation="nearest", aspect="equal")
        unit_name = {m: "m", cm: "cm", mm: "mm", um: "µm", nm: "nm"}.get(units, f"{units/m:g} m")
        title = f"Screen distance = {self.z/cm} cm" if text is None else text
        ax.set(title=title, xlabel=f"x [{unit_name}]", ylabel=f"y [{unit_name}]")
        if xlim is not None:
            ax.set_xlim(np.asarray(xlim)/units)
        if ylim is not None:
            ax.set_ylim(np.asarray(ylim)/units)
        if grid:
            ax.grid(alpha=0.2)
        colorbar = ax.figure.colorbar(im, ax=ax, label="Phase [rad]", ticks=[-np.pi, 0, np.pi])
        colorbar.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        if owns_figure:
            plt.show()
    return ax

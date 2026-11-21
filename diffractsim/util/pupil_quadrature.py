"""Fourier integration of physical pixels, including a circular pupil edge."""
import numpy as np
from .backend_functions import backend as bd, backend_name
from .bluestein_FFT import bluestein_fft


class PupilQuadrature:
    """Tensor Gauss nodes in each pixel; integrate circle cut cells separately.

    The weights integrate a local polynomial interpolant over the actual circle.
    The Fourier transform is a sum of shifted regular-grid FFTs, not an RS sum.
    """

    def __init__(self, extent_x, extent_y, nx, ny, radius, order=4):
        """
        Build Gauss integration nodes within pixels of a circular pupil.

        Parameters
        ----------
        extent_x: width of the original rectangular pixel grid, in micrometres.
        extent_y: height of the original rectangular pixel grid, in micrometres.
        nx: horizontal number of original pixels.
        ny: vertical number of original pixels.
        radius: positive circular pupil radius, in micrometres; must fit inside the grid.
        order: Gauss nodes per pixel axis, default 4.

        Notes
        -----
        x and y contain nx*order and ny*order nodes. weights has shape
        (ny*order, nx*order) and units of m². Edge-cell weights integrate
        only the part of each pixel inside the circle. Geometry is fixed;
        sampled field values remain differentiable.
        """
        self.nx, self.ny, self.order = nx, ny, order
        self.dx, self.dy = extent_x/nx, extent_y/ny
        nodes, weights = np.polynomial.legendre.leggauss(order)
        self.offsets = (nodes+1)/2
        x0 = (np.arange(nx)-nx//2)*self.dx
        y0 = (np.arange(ny)-ny//2)*self.dy
        self.x = bd.asarray((x0[:, None]+self.dx*self.offsets).ravel())
        self.y = bd.asarray((y0[:, None]+self.dy*self.offsets).ravel())
        if radius <= 0 or min(-x0[0], x0[-1]+self.dx, -y0[0], y0[-1]+self.dy) < radius:
            raise ValueError("The positive pupil radius must fit inside the pixel grid.")
        self.radius = radius
        xl, yl = np.meshgrid(x0, y0)
        xh, yh = xl+self.dx, yl+self.dy
        near = np.maximum(np.maximum(xl, -xh), 0)**2+np.maximum(np.maximum(yl, -yh), 0)**2
        far = np.maximum(abs(xl), abs(xh))**2+np.maximum(abs(yl), abs(yh))**2
        area = np.zeros((ny, nx, order, order))
        area[far <= radius**2] = self.dx*self.dy*np.outer(weights, weights)/4
        edge_y, edge_x = np.nonzero((near < radius**2) & (far > radius**2))
        g, w = np.polynomial.legendre.leggauss(max(18, 3*order))

        def basis(t):
            values = np.ones((len(t), order))
            for j in range(order):
                for k in range(order):
                    if j != k:
                        values[:, j] *= (t-nodes[k])/(nodes[j]-nodes[k])
            return values

        for iy, ix in zip(edge_y, edge_x):
            left, right, bottom, top = xl[iy, ix], xh[iy, ix], yl[iy, ix], yh[iy, ix]
            breaks = [max(bottom, -radius), min(top, radius)]
            for edge in (left, right):
                if abs(edge) < radius:
                    cross = np.sqrt(radius**2-edge**2)
                    breaks.extend(t for t in (-cross, cross) if breaks[0] < t < breaks[1])
            breaks = sorted(set(breaks))
            for lo, hi in zip(breaks[:-1], breaks[1:]):
                y = (lo+hi)/2+(hi-lo)*g/2
                half = np.sqrt(np.maximum(radius**2-y*y, 0))
                a, b = np.maximum(left, -half), np.minimum(right, half)
                x = (a[:, None]+b[:, None])/2+(b-a)[:, None]*g/2
                weight = w[:, None]*w[None, :]*(hi-lo)*np.maximum(b-a, 0)[:, None]/4
                bx = basis((2*(x-left)/self.dx-1).ravel()).reshape(len(g), len(g), order)
                by = basis(2*(y-bottom)/self.dy-1)
                area[iy, ix] += np.einsum('ab,abi,aj->ji', weight, bx, by)
        self.weights = bd.asarray(area.transpose(0, 2, 1, 3).reshape(ny*order, nx*order))

    def fourier_transform(self, field, fx, fy):
        """
        Integrate a sampled pupil against the continuous Fourier kernel.

        Parameters
        ----------
        field: complex values at the quadrature nodes, shape (ny*order, nx*order).
        fx: uniformly spaced horizontal spatial frequencies, in cycles per micrometre.
        fy: uniformly spaced vertical spatial frequencies, in cycles per micrometre.

        Returns
        -------
        spectrum: complex Fourier integral with shape (len(fy), len(fx));
            includes the physical area weights and exp(-2*pi*i*(fx*x+fy*y)).
        """
        n, m, q = self.nx, self.ny, self.order
        values = (field*self.weights).reshape(m, q, n, q).transpose(1, 0, 3, 2)
        origins = (-(n//2)+self.offsets)*self.dx
        phase = bd.exp(-2j*bd.pi*bd.asarray(origins)[:, None]*fx)
        def transform_rows(values):
            values = bluestein_fft(values, -1, fx[0], fx[-1], 1/self.dx, len(fx))
            return bd.sum(values*phase, axis=1)
        if backend_name == "jax":
            import jax
            values = jax.lax.map(jax.checkpoint(transform_rows), values)
        else:
            values = bd.stack([transform_rows(row) for row in values])
        values = bluestein_fft(values, 1, fy[0], fy[-1], 1/self.dy, len(fy))
        origins = (-(m//2)+self.offsets)*self.dy
        return bd.sum(values*bd.exp(-2j*bd.pi*fy[None, :, None]*bd.asarray(origins)[:, None, None]), axis=0)

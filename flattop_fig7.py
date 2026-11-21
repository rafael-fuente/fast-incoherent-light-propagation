"""Small, fixed-geometry calculations for the flat-top example in Fig. 7.

Lengths are in diffractsim's micrometres. Phase values stay differentiable;
only source quadrature, propagation matrices and interpolation are prepared.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import erf
from scipy.fft import next_fast_len
import jax
import jax.numpy as jnp
from diffractsim import IncoherentLightSimulation, mm, um, nm
from diffractsim.util.compact_source import lagrange_weights


def disk_source(z1, radial=3, azimuthal=8):
    """Integrate five equal-radiance disks, with unit total incident irradiance."""
    nodes, weights = np.polynomial.legendre.leggauss(radial)
    angle = 2*np.pi*np.arange(azimuthal)/azimuthal
    centres = np.array([[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]])*mm
    radii = np.array([.8]+[.8/np.sqrt(2)]*4)*mm
    xy, area = [], []
    for centre, radius in zip(centres, radii):
        r = radius*np.sqrt((nodes+1)/2)
        xy.append((centre+np.stack(np.broadcast_arrays(
            r[:, None]*np.cos(angle), r[:, None]*np.sin(angle)), axis=-1)).reshape(-1, 2))
        area.extend(np.repeat(np.pi*radius**2*weights/(2*azimuthal), azimuthal))
    xy = np.concatenate(xy)
    distance = np.sqrt(z1*z1+(xy*xy).sum(axis=1))
    mass = np.array(area)*(z1/distance)**4/z1**2
    return xy, xy/distance[:, None], mass/mass.sum()


def interpolation(x, y, x_axis, y_axis):
    """Prepare a six-point interpolation on a uniform rectangular grid."""
    ix, wx = lagrange_weights((np.ravel(x)-x_axis[0])/(x_axis[1]-x_axis[0]))
    iy, wy = lagrange_weights((np.ravel(y)-y_axis[0])/(y_axis[1]-y_axis[0]))
    if ix.min() < 0 or iy.min() < 0 or ix.max() >= len(x_axis) or iy.max() >= len(y_axis):
        raise ValueError("The interpolation grid needs a wider margin.")
    return tuple(jnp.asarray(a) for a in (ix, iy, wx, wy))


def sample(values, rule):
    """Interpolate values using a prepared rule; gradients pass through values."""
    ix, iy, wx, wy = rule
    result = jnp.zeros(ix.shape[0], dtype=values.dtype)
    for row in range(6):
        result += wy[:, row]*jnp.sum(values[iy[:, row, None], ix]*wx, axis=1)
    return result


def spectral_patch(axis, pitch, wavelength, z, x, y, refinement=1):
    """Prepare an angular-spectrum integral for one small monitor rectangle.

    The sinc factors integrate constant phase pixels. A tapered spectral
    margin includes the rays reaching this rectangle from the whole pupil.
    Increase refinement to widen that margin and reduce the frequency step.
    """
    xc, yc = (x[0]+x[-1])/2, (y[0]+y[-1])/2
    c = z/np.sqrt(z*z+xc*xc+yc*yc)
    pieces = []
    for output, centre in ((x, xc), (y, yc)):
        radius, half = len(axis)*pitch/2, (output[-1]-output[0])/2
        inner = refinement*(half+radius)*c/(z*wavelength)
        width = 1.25*inner
        reach = width*z*wavelength/c**3+radius
        df = 1/(2*(half+reach+2*radius)*.7*refinement)
        offset = np.linspace(-width, width, int(np.ceil(2*width/df))+1)
        frequency = centre*c/(z*wavelength)+offset
        taper = np.where(abs(offset) <= inner, 1.,
            .5*(1+np.cos(np.pi*(abs(offset)-inner)/(width-inner))))
        forward = pitch*np.sinc(pitch*frequency)[:, None]*np.exp(-2j*np.pi*np.outer(frequency, axis))
        inverse = np.exp(2j*np.pi*np.outer(output-centre, frequency))
        pieces.append((frequency, taper, forward, inverse))
    fx, wx, Fx, Gx = pieces[0]
    fy, wy, Fy, Gy = pieces[1]
    longitudinal = wavelength**-2-fy[:, None]**2-fx[None, :]**2
    if np.any(longitudinal <= 0):
        raise ValueError("This spectral patch extends outside the propagating disk.")
    transfer = np.exp(2j*np.pi*(z*np.sqrt(longitudinal)+xc*fx[None, :]+yc*fy[:, None]))
    transfer *= wy[:, None]*wx[None, :]*(fx[1]-fx[0])*(fy[1]-fy[0])
    return fx, fy, tuple(jnp.asarray(a) for a in (Fx, Fy, Gx, Gy, transfer))


class FlatTopDesign:
    """Keep the fixed source and monitor maps for the two Fig. 7 design models."""

    def __init__(self, screen_step=5*um, angular_step=4/2048,
                 radial=3, azimuthal=8, refinement=1):
        self.wavelength, self.radius, self.pitch = 940*nm, 50*um, .47*um
        self.z1, self.z2 = 300*mm, (50*um)**2/(940*nm)
        self.n = 213
        grid = IncoherentLightSimulation(self.wavelength, self.n*self.pitch,
                                        self.n*self.pitch, self.n, self.n)
        self.axis, self.xx, self.yy = map(np.asarray, (grid.x, grid.xx, grid.yy))
        self.mask = self.xx**2+self.yy**2 <= self.radius**2
        self.power = self.mask.sum()*self.pitch**2  # Unit incident irradiance.
        self.local = np.linspace(-210*um, 210*um, round(420*um/screen_step)+1)
        self.screen_step = self.local[1]-self.local[0]
        self.orders = np.array([(m, n) for n in (1, 0, -1) for m in (-1, 0, 1)])
        directions = .4*self.orders
        self.centres = self.z2*directions/np.sqrt(1-(directions**2).sum(axis=1))[:, None]
        ux, uy = np.meshgrid(self.local, self.local)
        self.x = self.centres[:, 0, None]+ux.ravel()
        self.y = self.centres[:, 1, None]+uy.ravel()
        rho = np.sqrt(self.z2**2+self.x**2+self.y**2)
        alpha, beta = self.x/rho, self.y/rho
        self.monitor_weight = jnp.asarray(self.z2**2/rho**4)
        profile = .5*(erf((160*um-self.local)/(np.sqrt(2)*20*um))+
                      erf((160*um+self.local)/(np.sqrt(2)*20*um)))
        self.target = jnp.asarray(np.outer(profile, profile))
        self.interior, self.collection = np.array(self.target > .95), jnp.asarray(self.target >= .5)
        self.source_xy, source_ab, self.source_mass = disk_source(self.z1, radial, azimuthal)
        self.curvature = jnp.exp(1j*np.pi*(self.xx**2+self.yy**2)/(self.wavelength*self.z1))

        # Deposit the 120 quadrature weights on a small reversed source grid.
        h = angular_step
        source_axis = np.arange(np.floor(source_ab.min()/h)-4, np.ceil(source_ab.max()/h)+5)*h
        ix, wx = lagrange_weights((-source_ab[:, 0]-source_axis[0])/h)
        iy, wy = lagrange_weights((-source_ab[:, 1]-source_axis[0])/h)
        mass = np.zeros((len(source_axis), len(source_axis)))
        for j in range(6):
            for k in range(6):
                np.add.at(mass, (iy[:, j], ix[:, k]), self.source_mass*wy[:, j]*wx[:, k])
        self.tiles = []
        for a, b in zip(alpha, beta):
            axes = [np.arange(np.floor((q.min()-source_axis[-1])/h)-4,
                              np.ceil((q.max()-source_axis[0])/h)+5)*h for q in (a, b)]
            ka, kb = axes
            aa, bb = np.meshgrid(ka, kb)
            cc = np.sqrt(1-aa*aa-bb*bb)
            xr, yr = self.z2*aa/cc, self.z2*bb/cc
            screen = [np.arange(np.floor(q.min()/screen_step)-4,
                                np.ceil(q.max()/screen_step)+5)*screen_step for q in (xr, yr)]
            _, _, matrices = spectral_patch(self.axis, self.pitch, self.wavelength,
                                             self.z2, *screen, refinement)
            response_rule = interpolation(xr, yr, *screen)
            shape = tuple(next_fast_len(n+len(source_axis)-1) for n in (len(kb), len(ka)))
            output_axes = [q[0]+source_axis[0]+np.arange(n)*h for q, n in zip(axes, shape[::-1])]
            output_rule = interpolation(a, b, *output_axes)
            fourier = [jnp.asarray(self.pitch*np.sinc(q*self.pitch/self.wavelength)[:, None]*
                       np.exp(-2j*np.pi*np.outer(q/self.wavelength, self.axis))) for q in axes]
            self.tiles.append((matrices, response_rule, jnp.asarray(self.z2**2/cc.ravel()**4),
                               aa.shape, shape, jnp.fft.rfft2(jnp.asarray(mass), s=shape),
                               output_rule, fourier))
        signature = np.concatenate([self.local, self.source_xy.ravel(), self.source_mass,
                                     [angular_step, refinement]])
        code = Path(__file__).read_bytes()
        self.signature = hashlib.sha256(signature.tobytes()+code).hexdigest()[:16]

    def irradiance(self, phase, model="eq24"):
        """Evaluate Eq. (22) or Eq. (24) for a new phase on the same physical pixels."""
        if model not in ("eq22", "eq24"):
            raise ValueError('Choose "eq22" or "eq24".')
        pupil = jnp.asarray(self.mask)*jnp.exp(1j*phase)*self.curvature
        values = []
        for matrices, rule, weight, shape, fft_shape, source_fft, output_rule, fourier in self.tiles:
            if model == "eq24":
                Fx, Fy, Gx, Gy, transfer = matrices
                field = Gy @ ((Fy @ pupil @ Fx.T)*transfer) @ Gx.T
                kernel = (sample(jnp.abs(field)**2, rule)*weight).reshape(shape)
            else:
                Fx, Fy = fourier
                kernel = jnp.abs(Fy @ pupil @ Fx.T/self.wavelength)**2
            combined = jnp.fft.irfft2(source_fft*jnp.fft.rfft2(kernel, s=fft_shape), s=fft_shape)
            values.append(sample(combined, output_rule))
        return (jnp.stack(values)*self.monitor_weight).reshape(9, len(self.local), len(self.local))

    def loss(self, phase, model="eq24"):
        """Match the square profiles and reward their collected power, as in S7."""
        E = self.irradiance(phase, model)
        mean = jnp.mean(E[:, self.interior])
        efficiency = jnp.sum(E*self.collection)*self.screen_step**2/self.power
        return jnp.mean((E/mean-self.target)**2)+.05*(1-efficiency)

    def metrics(self, irradiance):
        """Report shape error, interior nonuniformity and collection as fractions."""
        E = np.asarray(irradiance)
        mean = E[:, self.interior].mean()
        return dict(shape_error=float(np.linalg.norm(E/mean-np.asarray(self.target))/
                                      (3*np.linalg.norm(np.asarray(self.target)))),
                    nonuniformity=float(E[:, self.interior].std()/mean),
                    collection=float(np.sum(E*np.asarray(self.collection))*self.screen_step**2/self.power))

    def pixel_quadrature(self, order):
        """Place Gauss nodes inside every constant phase pixel, without changing it."""
        nodes, weights = np.polynomial.legendre.leggauss(order)
        axis = (self.axis[:, None]+self.pitch*nodes/2).ravel()
        weights = np.tile(self.pitch*weights/2, self.n)
        return axis, weights

    def rs_irradiance(self, phase, pixel_order=3, refinement=1):
        """Propagate each spherical source field and add its RS intensity.

        Angular-spectrum integration is the spectral form of the outgoing
        RS propagator. This reference does not shift a common intensity
        response. Gauss nodes integrate the same constant physical pixels.
        Increase pixel_order and refinement to check the numerical accuracy.
        """
        axis, weights = self.pixel_quadrature(pixel_order)
        xx, yy = map(jnp.asarray, np.meshgrid(axis, axis))
        pupil = jnp.repeat(jnp.repeat(jnp.asarray(self.mask)*jnp.exp(1j*phase),
                                     pixel_order, 0), pixel_order, 1)
        k = 2*np.pi/self.wavelength
        results = []
        for cx, cy in self.centres:
            fx, fy, matrices = spectral_patch(self.axis, self.pitch, self.wavelength,
                self.z2, cx+self.local, cy+self.local, refinement)
            _, _, Gx, Gy, transfer = matrices
            Fx = jnp.asarray(weights[None, :]*np.exp(-2j*np.pi*np.outer(fx, axis)))
            Fy = jnp.asarray(weights[None, :]*np.exp(-2j*np.pi*np.outer(fy, axis)))
            def add_source(total, source):
                sx, sy, mass = source
                r0 = jnp.sqrt(self.z1**2+sx*sx+sy*sy)
                r = jnp.sqrt(self.z1**2+(xx-sx)**2+(yy-sy)**2)
                path = (xx*xx+yy*yy-2*(xx*sx+yy*sy))/(r+r0)
                incident = (r0/r)**2*(1+1j/(k*r))*jnp.exp(1j*k*path)
                field = Gy @ ((Fy @ (pupil*incident) @ Fx.T)*transfer) @ Gx.T
                return total+mass*jnp.abs(field)**2, None
            source = jnp.asarray(np.column_stack((self.source_xy, self.source_mass)))
            total, _ = jax.lax.scan(jax.checkpoint(add_source),
                                    jnp.zeros((len(self.local), len(self.local))), source)
            results.append(total)
        return jnp.stack(results)

    def direct_rs(self, phase, points, pixel_order=4):
        """Check selected detector points by direct integration of both RS kernels."""
        axis, weights = self.pixel_quadrature(pixel_order)
        xx, yy = np.meshgrid(axis, axis)
        pupil = np.repeat(np.repeat(self.mask*np.exp(1j*np.asarray(phase)), pixel_order, 0), pixel_order, 1)
        active = np.abs(pupil) > 0
        x, y = xx[active], yy[active]
        pupil = pupil[active]*np.outer(weights, weights)[active]
        k = 2*np.pi/self.wavelength
        results = np.zeros(len(points))
        for (sx, sy), mass in zip(self.source_xy, self.source_mass):
            r0 = np.sqrt(self.z1**2+sx*sx+sy*sy)
            r = np.sqrt(self.z1**2+(x-sx)**2+(y-sy)**2)
            path = (x*x+y*y-2*(x*sx+y*sy))/(r+r0)
            incident = (r0/r)**2*(1+1j/(k*r))*np.exp(1j*k*path)
            for j, (mx, my) in enumerate(points):
                rho = np.sqrt(self.z2**2+mx*mx+my*my)
                R = np.sqrt(self.z2**2+(mx-x)**2+(my-y)**2)
                path = (x*x+y*y-2*(x*mx+y*my))/(R+rho)
                kernel = self.z2/(self.wavelength*R**2)*(1+1j/(k*R))*np.exp(1j*k*path)
                results[j] += mass*abs(np.sum(pupil*incident*kernel))**2
        return results

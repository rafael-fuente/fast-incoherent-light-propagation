from PIL import Image
import numpy as np
from pathlib import Path


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


def rescale_img_to_custom_coordinates(img, image_size, extent_x,extent_y, Nx, Ny):

    img_pixels_width, img_pixels_height = img.size

    if image_size != None:
        new_img_pixels_width, new_img_pixels_height = int(np.round(image_size[0] / extent_x  * Nx)),  int(np.round(image_size[1] / extent_y  * Ny))
    else:
        #by default, the image fills the entire aperture plane
        new_img_pixels_width, new_img_pixels_height = Nx, Ny

    img = img.resize((new_img_pixels_width, new_img_pixels_height))

    dst_img = Image.new("RGB", (Nx, Ny), "black" )
    dst_img_pixels_width, dst_img_pixels_height = dst_img.size

    Ox, Oy = (dst_img_pixels_width-new_img_pixels_width)//2, (dst_img_pixels_height-new_img_pixels_height)//2
    
    dst_img.paste( img , box = (Ox, Oy ))
    return dst_img


def convert_graymap_image_to_hsvmap_image(img):
    imgRGB = np.asarray(img) / 255.0
    imgR = imgRGB[:, :, 0]
    imgG = imgRGB[:, :, 1]
    imgB = imgRGB[:, :, 2]
    graymap_array = np.array(0.2990 * imgR + 0.5870 * imgG + 0.1140 * imgB)

    from matplotlib.colors import hsv_to_rgb

    h = graymap_array
    s = np.ones_like(h)
    v = np.ones_like(h)
    rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1))

    img_RGB = [Image.fromarray((np.round(255 * rgb[:,:,0])).astype(np.uint8), "L"),
               Image.fromarray((np.round(255 * rgb[:,:,1])).astype(np.uint8), "L"),
               Image.fromarray((np.round(255 * rgb[:,:,2])).astype(np.uint8), "L")]

    return Image.merge("RGB", img_RGB)



def resize_array(img_array, new_shape):

    Ny, Nx = img_array.shape
    
    from scipy.interpolate import RectBivariateSpline
    fun = RectBivariateSpline(
        np.linspace(0, 1, Nx),
        np.linspace(0, 1, Ny),
        img_array)
    resize_img_array = fun(np.linspace(0, 1, new_shape[1]), np.linspace(0, 1, new_shape[0]))
    return resize_img_array






def load_image_as_array(path):

    """Load *path* into a 2‑D float array.

    The function performs **three** independent preprocessing steps:

    1. Read the file with *Pillow* and convert to 8‑bit sRGB.
    2. Convert RGB to luminance (*Y*) using the ITU‑R BT.601 weights
       (0.299R+0.587G+0.114B).

    Parameters
    ----------
    path
        Path to any raster image supported by *Pillow* (PNG, JPEG, TIFF, …).
    image_size
        ``(width, height)`` tuple to which the image should be resized.  ``None``
        (default) keeps the original resolution.

    Returns
    -------
    np.ndarray
        2‑D array of shape ``(height, width)`` with values in the range
        :math:`[0, 1]`.

    Examples
    --------
    >>> from pathlib import Path
    >>> img = load_image("input/photo.jpg")
    
    """
    img = Image.open(Path(path))
    img = img.convert("RGB")
    imgRGB = np.asarray(img) / 255.0

    t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
    t = np.array(np.flip(t, axis = 0))
    
    return t





def load_image_as_function(
    path,
    *,
    x_interval=None,
    y_interval=None,
    method="linear",
    bounds_error=False,
    fill_value=0.0,
    invert=False,
    smoothing=0.0,
):
    """Load a grayscale image as a function of physical coordinates, f(x, y).

    Parameters
    ----------
    path : str or pathlib.Path
        Image file. Values are normalized to [0, 1], as in ``load_image``.
    x_interval, y_interval : pair of floats, optional
        Coordinates of the first and last pixel centres on each axis.
        Defaults to [0, width-1] and [0, height-1]. The image stays upright:
        its bottom row is at ymin and its top row at ymax. Rectangular and
        off-centre intervals are supported; use the same units as x and y.
    method : {"linear", "nearest"}
        Interpolation between image samples. Linear is the default.
    bounds_error : bool
        Raise for points outside the sampled image, including any smoothing
        margin. This eager check is not supported inside JAX jit or grad;
        leave False there to use fill_value instead.
    fill_value : float or None
        Value outside the image; default 0. None extends the edge values.
    invert : bool
        If True, turn dark image markings into bright values before filtering.
    smoothing : float
        Gaussian standard deviation in the same units as the intervals.
        Zero disables smoothing. Padding is added automatically so the blur
        extends beyond the original boundary, without moving the image.

    Returns
    -------
    callable
        f(x, y) accepts scalars or broadcastable arrays. f(points) also
        accepts points of shape (..., 2), ordered as (x, y).

    Notes
    -----
    Select the NumPy, JAX or CuPy backend before loading. File reading and
    Gaussian filtering happen once on the CPU. Evaluation uses the selected
    backend; linear interpolation supports JAX coordinate gradients away
    from pixel boundaries. Create the function outside JAX transformations.
    The intervals locate pixel centres exactly; older versions used only
    their widths and implicitly centred the image at the origin.

    Examples
    --------
    >>> from diffractsim import mm
    >>> logo = load_image_as_function("github_logo.png",
    ...     x_interval=(-110*mm, 110*mm), y_interval=(-110*mm, 110*mm),
    ...     smoothing=2.5*mm)
    >>> target_shape = logo(screen_x, screen_y)
    """
    from .backend_functions import backend as bd, backend_name

    if backend_name == "jax":
        import jax
        from jax.scipy.ndimage import map_coordinates
    elif backend_name == "cupy":
        from cupyx.scipy.ndimage import map_coordinates
    else:
        from scipy.ndimage import map_coordinates

    if method not in ("linear", "nearest"):
        raise ValueError("method must be 'linear' or 'nearest'")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be a finite, nonnegative standard deviation")

    data = _load_image_pixels(path, invert)
    ny, nx = data.shape

    def axis_geometry(interval, size, name):
        limits = np.asarray((0, size-1) if interval is None else interval, dtype=float)
        if limits.shape != (2,) or not np.all(np.isfinite(limits)):
            raise ValueError(f"{name} must contain two finite coordinates")
        lo, hi = limits
        if hi < lo or (hi == lo and size > 1):
            raise ValueError(f"{name} must be increasing")
        spacing = (hi-lo)/max(size-1, 1) if hi > lo else 1.0
        return lo, hi, spacing

    xmin, xmax, dx = axis_geometry(x_interval, nx, "x_interval")
    ymin, ymax, dy = axis_geometry(y_interval, ny, "y_interval")
    if smoothing:
        from scipy.ndimage import gaussian_filter

        sigma_pixels = np.array([smoothing/dy, smoothing/dx])
        # Cover the four-sigma filter plus one sample for interpolation.
        py, px = np.ceil(4*sigma_pixels).astype(int)+1
        padding = ((py, py), (px, px))
        if fill_value is None:
            data = np.pad(data, padding, mode="edge")
            data = gaussian_filter(data, sigma_pixels, mode="nearest")
        else:
            data = np.pad(data, padding, constant_values=fill_value)
            data = gaussian_filter(data, sigma_pixels, mode="constant", cval=fill_value)
        xmin, xmax = xmin-px*dx, xmax+px*dx
        ymin, ymax = ymin-py*dy, ymax+py*dy

    data = bd.asarray(data)

    def image_function(x, y=None):
        if y is None:
            points = bd.asarray(x, dtype=float)
            if points.ndim == 0 or points.shape[-1] != 2:
                raise ValueError("Provide x and y, or points with shape (..., 2)")
            x, y = points[..., 0], points[..., 1]
        x, y = bd.broadcast_arrays(bd.asarray(x, dtype=float), bd.asarray(y, dtype=float))
        outside = (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax)
        if bounds_error:
            if backend_name == "jax" and isinstance(outside, jax.core.Tracer):
                raise ValueError("Use bounds_error=False inside JAX jit or grad")
            if bool(bd.any(outside)):
                raise ValueError("Points outside the interpolation domain")

        ix, iy = (x-xmin)/dx, (y-ymin)/dy
        coordinates = bd.stack([iy.ravel(), ix.ravel()])
        values = map_coordinates(data, coordinates,
            order=1 if method == "linear" else 0, mode="nearest").reshape(x.shape)
        return values if fill_value is None else bd.where(outside, fill_value, values)

    return image_function


def _load_image_pixels(path, invert=False):
    """Read upright grayscale pixels once, before transfer to an array backend."""
    with Image.open(Path(path)) as image:
        pixels = np.asarray(image.convert("L"), dtype=float)/255
    if invert:
        pixels = 1-pixels
    return np.flipud(pixels).copy()


def load_image(path, invert=False):
    """Load normalized grayscale pixels into the active NumPy/JAX/CuPy backend.

    Rows follow physical y increasing upward. invert=True turns dark image
    markings into bright values. The original image resolution is preserved.
    """
    from .backend_functions import backend as bd

    return bd.asarray(_load_image_pixels(path, invert))

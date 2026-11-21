import numpy as np
from ..util.backend_functions import backend as bd
from .optical_element import OpticalElement
from ..util.image_handling import convert_graymap_image_to_hsvmap_image, rescale_img_to_custom_coordinates
from PIL import Image
from pathlib import Path

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


class SLM(OpticalElement):
    def __init__(self, phase_mask_function, pixel_dx, pixel_dy, Nx, Ny, simulation = None):

        """
        Class representing a spatial light modulator (SLM)
        Imparts a given phase profile specified as a function by phase_mask_function agument
        The SLM is centered on the plane and its physical size is specified in image_size parameter as size_x: float and size_y: float
        """

        
        global bd
        global backend_name
        from ..util.backend_functions import backend as bd
        from ..util.backend_functions import backend_name
        self.pixel_dx = pixel_dx
        self.pixel_dy = pixel_dy

        self.size_x = pixel_dx*Nx
        self.size_y = pixel_dy*Ny
        self.simulation = simulation
        self.phase_mask_function = phase_mask_function        

        
    def get_transmittance(self, xx, yy, λ):
        if backend_name == 'cupy':
            return bd.where((bd.abs(xx) < self.size_x/2)   &  (bd.abs(yy) < self.size_y/2),   bd.exp(1j *  bd.array(self.phase_mask_function(xx.get(),yy.get()))), bd.zeros(xx.shape))
        else:
            return bd.where((bd.abs(xx) < self.size_x/2)   &  (bd.abs(yy) < self.size_y/2),   bd.exp(1j *  self.phase_mask_function(xx,yy)), bd.zeros(xx.shape))


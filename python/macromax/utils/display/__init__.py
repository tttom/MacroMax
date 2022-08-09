"""
This package contains functionality to simplify the display of complex matrices.
"""
import logging

log = logging.getLogger(__name__)

from .grid2extent import grid2extent
from .complex2rgb import complex2rgb
from .hsv import hsv2rgb, rgb2hsv
from .colormap import InterpolatedColorMap

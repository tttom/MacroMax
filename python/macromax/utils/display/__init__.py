"""
This package contains functionality to simplify the display of complex matrices.
"""
from .grid2extent import grid2extent
from .complex2rgb import complex2rgb
from .hsv import hsv2rgb, rgb2hsv
from .colormap import InterpolatedColorMap
from .format_image_axes import format_image_axes
from macromax.utils import log

log = log.getChild(__name__)

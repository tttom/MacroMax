import numpy as np
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

from macromax.utils import round125


def format_image_axes(axes, title: str = None, scale: float | np.ndarray = 0, unit: str = "", white_background: bool = False):
    """
    Replaces the axes ticks on an image with a(n optional) scalebar and adds a(n optional) title.
    TODO: Integrate examples.complex_color_legend into this as well.

    :param axes: The Axes object to update.
    :param title: An optional title to be placed in the top left corner.
    :param scale: The width of the scale bar, typically grid.extent[0] / 8 works well. Set to 0 for no scalebar.
    :param unit: Optional unit to use. E.g. 'm' will translate to 'mm' for scales on the order of 1e-3.
    :param white_background: If True, use white annotations, otherwise black. (default: False).

    :return: The modified axes.
    """
    foreground = (0, 0, 0, 1) if white_background else (1, 1, 1, 1)

    axes.set(xticks=[], yticks=[])
    axes.axis('off')

    if title is not None:  # Add title
        if title is not None and len(title) > 0:
            anchored_text = AnchoredText(title, loc='upper left', borderpad=0.0, frameon=False,
                                         prop=dict(color=foreground, fontweight='normal', fontsize=14))
            axes.add_artist(anchored_text)

    if scale > 0:  # Add scale bar
        scale_bar_size = round125(scale)
        scale_bar_label = f"{scale_bar_size}{unit}"
        scale_bar = AnchoredSizeBar(axes.transData,
                                    scale_bar_size, scale_bar_label, "lower right",
                                    pad=0.5,  # In unit of font size
                                    color=foreground,
                                    frameon=False,
                                    label_top=True,
                                    size_vertical=scale_bar_size / 5,
                                    fontproperties=fm.FontProperties(size=12))
        axes.add_artist(scale_bar)

    return axes

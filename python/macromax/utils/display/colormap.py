import numpy as np

__all__ = ['InterpolatedColorMap']

from matplotlib.colors import LinearSegmentedColormap


class InterpolatedColorMap(LinearSegmentedColormap):
    """
    A custom colormap for use with imshow and colorbar.

    Example usage:

    ::
        cmap = colormap.InterpolatedColorMap('hsv', [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 0, 0), (1, 1, 1)])
        cmap = colormap.InterpolatedColorMap('rainbow', [(0, 0, 0), (1, 0, 0), (0.75, 0.75, 0), (0, 1, 0), (0, 0.75, 0.75), (0, 0, 1), (0.75, 0, 0.75), (1, 1, 1)],
                                             points=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1])

        fig, ax = subplots(1, 1)
        ax.imshow(intensity_array, cmap=cmap)
        from matplotlib import cm
        fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)

    """
    def __init__(self, name: str, colors, points=None):
        colors = np.asarray(colors)
        if points is None:  # Uniform spacing by default
            points = np.arange(colors.shape[0]) / (colors.shape[0] - 1)
        stack = lambda _: np.stack((points, _, _), axis=1)
        cdict = dict(red=stack(colors[:, 0]), green=stack(colors[:, 1]), blue=stack(colors[:, 2]))
        if colors.shape[1] > 3:
            cdict['alpha'] = stack(colors[:, 3])
        super().__init__(name, cdict)

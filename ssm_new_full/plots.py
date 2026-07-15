import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def white_to_color_cmap(color, nsteps=256):
    color = np.asarray(color)
    cdict = {
        "red": ((0.0, 1.0, 1.0), (1.0, color[0], color[0])),
        "green": ((0.0, 1.0, 1.0), (1.0, color[1], color[1])),
        "blue": ((0.0, 1.0, 1.0), (1.0, color[2], color[2])),
    }
    return LinearSegmentedColormap("white_color_colormap", cdict, nsteps)


def gradient_cmap(colors, nsteps=256, bounds=None):
    colors = np.asarray(colors)
    if bounds is None:
        bounds = np.linspace(0, 1, len(colors))
    cdict = {"red": [], "green": [], "blue": [], "alpha": []}
    for bound, color in zip(bounds, colors):
        cdict["red"].append((bound, color[0], color[0]))
        cdict["green"].append((bound, color[1], color[1]))
        cdict["blue"].append((bound, color[2], color[2]))
        alpha = color[3] if len(color) == 4 else 1.0
        cdict["alpha"].append((bound, alpha, alpha))
    return LinearSegmentedColormap("gradient_colormap", cdict, nsteps)

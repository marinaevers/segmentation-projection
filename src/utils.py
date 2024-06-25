import numpy as np
import os
import mapping.rendering as r


def draw(data, path, filename, img = None, colormap=None):
    r.visualize(img, os.path.join(path, filename + ".png"), data, minV=np.min(data), maxV=np.max(data), useCC=True, colormap=colormap)

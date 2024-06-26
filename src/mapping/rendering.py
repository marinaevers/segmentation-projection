import math

import numpy as np
from skimage.transform import resize
from PIL import Image
from skimage.segmentation import felzenszwalb
from matplotlib import cm


FACTOR = 12

def computeSurface(img, labelX, labelY, i, j, h, bw):
    """
    Computes the surface coefficients a, b, c, d
    that serve the same function as in the Cushion treemap
    paper.
    :param img: Image
    :param labelX: Label in x-direction
    :param labelY: Label in y-direction
    :param i: x-Coordinate of Pixel to consider
    :param j: y-Coordinate of Pixel to consider
    :param h: Maximum height of the surface
    :param bw: Boundary width
    :return: Coefficients a,b,c,d
    """
    y1 = np.where(labelX[i] == labelX[i,j])[0][0]
    x1 = np.where(labelY[:,j] == labelY[i,j])[0][0]
    y2 = np.where(labelX[i] == labelX[i,j])[0][-1]
    x2 = np.where(labelY[:,j] == labelY[i,j])[0][-1]
    right = min(img.shape[1]-1, y2+1)
    left = max(0, y1-1)
    top = min(img.shape[0]-1, x2+1)
    bottom = max(0, x1-1)
    # For checking the corners
    if i < bw or j < bw or i > img.shape[0]-bw - 1 or j > img.shape[1]-bw - 1:
        bottomRightCorner, topRightCorner, topLeftCorner, bottomLeftCorner = False, False, False, False
    else:
        bottomRightCorner = img[i, j] == img[i+bw, j] and img[i,j] == img[i, j-bw] and img[i,j] != img[i+bw, j-bw]
        topRightCorner = img[i, j] == img[i+bw, j] and img[i,j] == img[i, j+bw] and img[i,j] != img[i+bw, j+bw]
        topLeftCorner = img[i, j] == img[i-bw, j] and img[i,j] == img[i, j+bw] and img[i,j] != img[i-bw, j+bw]
        bottomLeftCorner = img[i, j] == img[i-bw, j] and img[i,j] == img[i, j-bw] and img[i,j] != img[i-bw, j-bw]
    a, b, c, d = 0, 0, 0, 0
    if (i < x1 + bw or bottomLeftCorner or topLeftCorner) and img[bottom, j] != -2:
        # Adapt value for x1
        if bottomLeftCorner:
            x1 = np.where(labelY[:,j-bw] == labelY[i,j-bw])[0][0]
        if topLeftCorner:
            x1 = np.where(labelY[:,j+bw] == labelY[i,j+bw])[0][0]
        a = -h/(bw*bw)
        c = 2*h/(bw*bw)*(x1+bw)
    elif (i > x2 - bw or bottomRightCorner or topRightCorner) and img[top, j] != -2:
        # Adapt value for x2
        if bottomRightCorner:
            x2 = np.where(labelY[:,j-bw] == labelY[i,j-bw])[0][-1]
        if topRightCorner:
            x2 = np.where(labelY[:,j+bw] == labelY[i,j+bw])[0][-1]
        a = -h/(bw*bw)
        c = 2*h/(bw*bw)*(x2-bw)
    if (j < y1 + bw or bottomRightCorner or bottomLeftCorner) and img[i, left] != -2:
        # Adapt value for y1
        if bottomRightCorner:
            y1 = np.where(labelX[i+bw] == labelX[i+bw,j])[0][0]
            dx = x2 - i
            dy = j - y1
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        if bottomLeftCorner:
            y1 = np.where(labelX[i-bw] == labelX[i-bw,j])[0][0]
            dx = i - x1
            dy = j - y1
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        b = -h/(bw*bw)
        d = 2*h/(bw*bw)*(y1+bw)
    elif (j > y2 - bw or topRightCorner or topLeftCorner) and img[i, right] != -2:
        # Adapt value for y1
        if topRightCorner:
            y2 = np.where(labelX[i+bw] == labelX[i+bw,j])[0][-1]
            dx = x2 - i
            dy = y2 - j
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        if topLeftCorner:
            y2 = np.where(labelX[i-bw] == labelX[i-bw,j])[0][-1]
            dx = i - x1
            dy = y2 - j
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        b = -h/(bw*bw)
        d = 2*h/(bw*bw)*(y2-bw)
    return a, b, c, d

def computeValley(img, labelX, labelY, i, j, h, bw):
    """
    Computes the surface coefficients for a valley in the
    boundaries that are real boundaries
    :param labelX: Label in x-direction
    :param labelY: Label in y-direction
    :param i: x-Coordinate of Pixel to consider
    :param j: y-Coordinate of Pixel to consider
    :param h: Maximum height of the surface
    :param bw: Boundary width
    :return: Coefficients a,b,c,d
    """
    y1 = np.where(labelX[i] == labelX[i, j])[0][0]
    x1 = np.where(labelY[:, j] == labelY[i, j])[0][0]
    y2 = np.where(labelX[i] == labelX[i, j])[0][-1]
    x2 = np.where(labelY[:, j] == labelY[i, j])[0][-1]
    # For checking the corners
    if i < bw or j < bw or i > img.shape[0]-bw - 1 or j > img.shape[1]-bw - 1:
        bottomRightCorner, topRightCorner, topLeftCorner, bottomLeftCorner = False, False, False, False
    else:
        bottomRightCorner = img[i, j] == img[i+bw, j] and img[i,j] == img[i, j-bw] and img[i,j] != img[i+bw, j-bw]
        topRightCorner = img[i, j] == img[i+bw, j] and img[i,j] == img[i, j+bw] and img[i,j] != img[i+bw, j+bw]
        topLeftCorner = img[i, j] == img[i-bw, j] and img[i,j] == img[i, j+bw] and img[i,j] != img[i-bw, j+bw]
        bottomLeftCorner = img[i, j] == img[i-bw, j] and img[i,j] == img[i, j-bw] and img[i,j] != img[i-bw, j-bw]
    a, b, c, d = 0, 0, 0, 0
    if (i < x1 + bw or bottomLeftCorner or topLeftCorner):
        # Adapt value for x1
        if bottomLeftCorner:
            x1 = np.where(labelY[:, j - bw] == labelY[i, j - bw])[0][0]
        if topLeftCorner:
            x1 = np.where(labelY[:, j + bw] == labelY[i, j + bw])[0][0]
        a = h / (bw * bw)
        c = -2 * h / (bw * bw) * (bw + x1)
    elif (i > x2 - bw or bottomRightCorner or topRightCorner):
        # Adapt value for x2
        if bottomRightCorner:
            x2 = np.where(labelY[:, j - bw] == labelY[i, j - bw])[0][-1]
        if topRightCorner:
            x2 = np.where(labelY[:, j + bw] == labelY[i, j + bw])[0][-1]
        a = h / (bw * bw)
        c = 2 * h / (bw * bw) * (bw - x2)
    if (j < y1 + bw or bottomRightCorner or bottomLeftCorner):
        # Adapt value for y1
        if bottomRightCorner:
            y1 = np.where(labelX[i + bw] == labelX[i + bw, j])[0][0]
            dx = x2 - i
            dy = j - y1
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        if bottomLeftCorner:
            y1 = np.where(labelX[i - bw] == labelX[i - bw, j])[0][0]
            dx = i - x1
            dy = j - y1
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        b = h / (bw * bw)
        d = -2 * h / (bw * bw) * (bw + y1)
    elif (j > y2 - bw or topRightCorner or topLeftCorner):
        # Adapt value for y1
        if topRightCorner:
            y2 = np.where(labelX[i + bw] == labelX[i + bw, j])[0][-1]
            dx = x2 - i
            dy = y2 - j
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        if topLeftCorner:
            y2 = np.where(labelX[i - bw] == labelX[i - bw, j])[0][-1]
            dx = i - x1
            dy = y2 - j
            if dx > dy:
                return a, 0, c, 0
            else:
                a = 0
                c = 0
        b = h / (bw * bw)
        d = 2 * h / (bw * bw) * (bw - y2)
    return a, b, c, d

def clostestSegment(img, labelX, labelY, i, j, h, bw):
    y1 = np.where(labelX[i] == labelX[i, j])[0][0]
    x1 = np.where(labelY[:, j] == labelY[i, j])[0][0]
    y2 = np.where(labelX[i] == labelX[i, j])[0][-1]
    x2 = np.where(labelY[:, j] == labelY[i, j])[0][-1]
    if min(i-x1, x2-i) < min(j-y1, y2-j):
        if i - x1 < x2 - i:  # 0.5*(y1+y2)-j < bw:#j-y1 < bw:
            if j - y1 < y2 - j and j - y1 - (i - x1) < bw:
                return img[x1-1, j], h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + x1 + j - y1), 0
            if y2 - j - (i - x1) < bw:
                return img[x1-1, j], h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + x1 + y2 - j), 0
        else:
            if j - y1 < y2 - j and j - y1 - (x2 - i) < bw:
                return img[x1-1, j], h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + x2 + y2 - j), 0
            if y2 - j - (x2 - i) < bw:
                return img[x1-1, j], h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + x2 + j - y1), 0
        return img[x1-1, j], 0,0,0,0
    else:
        # y direction opening
        if min(i-x1, x2-i) == min(j-y1, y2-j):
            return img[i, y1-1], 0, 0, 0.3, 0.3
            #return 0, True
        else:
            if j-y1 < y2-j:#0.5*(y1+y2)-j < bw:#j-y1 < bw:
                if i-x1 < x2-i and i-x1-(j-y1) < bw:
                    return img[i, y1-1], 0, h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + y1 + i-x1)
                if x2-i-(j-y1) < bw:
                    return img[i, y1-1], 0, h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + y1 + x2-i)
            else:
                if i-x1 < x2-i and i-x1-(y2-j) < bw:
                    return img[i, y1-1], 0, h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + y2 + x2-i)
                if x2-i-(y2-j) < bw:
                    return img[i, y1-1], 0, h / (bw * bw), 0, -2 * h / (bw * bw) * (-bw + y2 + i-x1)
        return img[i, y1 - 1], 0,0,0,0


def render(img, Ia = 40, Is = 215, L=None, h = 0.1, bw = 3, colorCoding=None):
    """
    Computes a rendering for the given input image.
    :param img: Image
    :param Ia: Intensity of ambient light
    :param Is: Intensity of directional light
    :param L: Vector that points towards the light source
    :param h: Maximum height of the surface
    :param bw: Boundary width
    :return: Shading image
    """
    if L is None:
        L = [1, 2, 10]
    out = np.empty(img.shape)
    img_out = np.empty(img.shape)
    # Normalize L vector
    L /= np.linalg.norm(L)
    # Create label image
    labelX = np.zeros(img.shape)
    labelY = np.zeros(img.shape)
    for i in range(len(img)):
        labelX[i] = felzenszwalb(np.array([img[i]]), min_size = 1, sigma=0.0)
    for i in range(len(img[0])):
        labelY[:,i] = felzenszwalb(np.array([img[:,i]]), min_size = 1, sigma=0.0)
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i,j] == 0:
                a, b, c, d = computeValley(img, labelX, labelY, i, j, h, bw)
            elif img[i,j] == -2:
                #a,b,c,d = 0,0,0,0
                img_out[i,j], a, b, c, d = clostestSegment(img, labelX, labelY, i,j, h, bw)
                # if equal:
                #     c, d = 0.5, 0.5
            else:
                a, b, c, d = computeSurface(img, labelX, labelY, i, j, h, bw)
            nx = -(2*a*i+c)
            ny = -(2*b*j+d)
            cosa = (nx*L[0] + ny*L[1] + L[2])/math.sqrt(nx*nx + ny*ny + 1)
            out[i,j] = Ia + max(0.0, Is*cosa)
    colorCoding[img == -2] = img_out[img == -2]
    img[img == -2] = img_out[img == -2]
    return out, img, colorCoding




def visualize(img, outPath, colorCoding, minV, maxV, useCC = True, colormap = None):
    img = resize(img, (FACTOR*img.shape[0], FACTOR*img.shape[1]), order=0)
    colorCoding = resize(colorCoding, (FACTOR*colorCoding.shape[0], FACTOR*colorCoding.shape[1]), order=0)
    out, img, colorCoding = render(img, bw=int(FACTOR/3), h = 1.0, colorCoding=colorCoding)
    maxC = maxV
    minC = minV
    if not useCC:
        colorCoding = img
    colorCoding = 1.0/(maxC-minC)*(colorCoding-minC)
    #img_c = cm.jet(colorCoding, bytes=True)
    # Some manual color coding
    img_c = np.zeros((*img.shape,4), dtype="uint8")
    # img_c[colorCoding==1] = [27, 158, 119, 255]
    # img_c[colorCoding==4] = [217, 95, 2, 255]
    # img_c[colorCoding==3] = [117, 112, 179, 255]
    # img_c[colorCoding==2] = [231, 41, 138, 255]
    if colormap:
        print("Color map: " + str(colormap))
        for s in colormap.keys():
            img_c[img==s] = 255*np.array(colormap[s])
    # Some other manual color map
    img_c[colorCoding == 0] = [255, 255, 255, 255]
    img_c[img != 0] = [255, 100, 100, 255]
    #img_c[img == 1] = [0.215686*255, 0.494118*255, 0.721569*255, 255]#[228, 26, 28, 255] # red
    #img_c[img == 2] = [0.596078*255, 0.305882*255, 0.639216*255, 255]#[55, 126, 184, 255] # blue
    #img_c[img == 3] = [0.894118*255, 0.101961*255, 0.109804*255, 255]#[152, 78, 163, 255] # purple
    #img_c[img == 4] = [0.301961*255, 0.686275*255, 0.290196*255, 255]#[228, 26, 28, 255]#[77, 175, 74, 255] # green
    # Colors Semiconductor
    # img_c[colorCoding == 1] = [255, 255, 51, 255]
    # img_c[colorCoding == 2] = [55, 126, 184, 255]
    # img_c[colorCoding == 3] = [255, 127, 0, 255]
    # img_c[colorCoding == 4] = [152, 78, 163, 255]
    # img_c[colorCoding == 5] = [77, 175, 74, 255]
    # img_c[colorCoding == 6] = [166, 86, 40, 255]
    # img_c[colorCoding == 7] = [228, 26, 28, 255]
    colorCoding = img_c
    out = out/255
    for i in range(3):
        colorCoding[:,:,i] = out*colorCoding[:,:,i]
    colorCoding[:,:,3] = 255
    colorCoding = Image.fromarray(colorCoding)
    colorCoding.save(outPath)
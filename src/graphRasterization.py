import numpy as np
import math

FACTOR = 1
NODESIZE = 21 * FACTOR
HALFNODESIZE = int(NODESIZE/2)
OFFSET = 5 + HALFNODESIZE/FACTOR
COUNTER = 1
XMIN = 0
YMIN = 0

def findEmbeddingSize(graph):
    """
    Finds out the extent of a given graph to use it for the embedding
    :param graph: Graph
    :param factor: Factor to multiply with size
    :return: Size
    """
    xMin = list(graph.values())[0].pos[0] * 10
    xMax = 0
    yMin = list(graph.values())[0].pos[1] * 10
    yMax = 0
    for n in graph.values():
        xMin = min(xMin, n.pos[0])
        yMin = min(yMin, n.pos[1])
        xMax = max(xMax, n.pos[0])
        yMax = max(yMax, n.pos[1])
        for e in n.edges.values():
            for b in e.bends:
                xMin = min(xMin, b[0])
                yMin = min(yMin, b[1])
                xMax = max(xMax, b[0])
                yMax = max(yMax, b[1])
    return (int(xMax + 2*OFFSET)*FACTOR, int(yMax + 2*OFFSET)*FACTOR)

def drawNode(img, pos, color):
    pos = (FACTOR*(np.array(pos) - [int(XMIN-OFFSET), int(YMIN-OFFSET)])).astype(int)
    img[pos[0]-HALFNODESIZE:pos[0]+HALFNODESIZE, pos[1]-HALFNODESIZE:pos[1]+HALFNODESIZE] = color

def colorPixel(img, x, y, color):
    """
    Tries to set the color if background or the same color
    :param img: Image
    :param x: x-Coordinate
    :param y: y-Coordinate
    :param color: Target value of the pixel
    :return: True if the color was set
    """
    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1]:
        return True
    if img[x,y] == 0 or img[x,y] == color or color in [-2, 0]:
        img[x,y] = color
        return True
    return False

def drawEdge(img, startO, endO, color):
    start = startO
    end = endO
    start = FACTOR*(np.array(start) - [int(XMIN-OFFSET), int(YMIN-OFFSET)])
    end = FACTOR*(np.array(end) - [int(XMIN-OFFSET), int(YMIN-OFFSET)])
    # Map to current region
    start[0] = max(0, start[0])
    start[0] = min(img.shape[0]-1, start[0])
    start[1] = max(0, start[1])
    start[1] = min(img.shape[1]-1, start[1])
    end[0] = max(0, end[0])
    end[0] = min(img.shape[0]-1, end[0])
    end[1] = max(0, end[1])
    end[1] = min(img.shape[1]-1, end[1])
    start = tuple(start.astype(int))
    end = tuple(end.astype(int))
    signX = 1
    signY = 1
    if start[0] > end[0]:
        signX = -1
    if start[1] > end[1]:
        signY = -1
    deltaX = abs(end[0]-start[0])
    deltaY = abs(end[1]-start[1])
    prevColor = 0
    if deltaX == 0:
        for i in range(deltaY+1):
            prevColorBefore = prevColor
            prevColor = img[start[0], start[1]+i*signY]
            if not colorPixel(img, start[0], start[1]+i*signY, color):
                if i == 0 or i == deltaY:
                    continue
                if img[start[0], start[1] + i * signY + 1 * signY] in [0, color] and img[start[0], start[1] + i * signY + 2 * signY] in [0, color] and i > 1:
                    colorPixel(img, start[0], start[1]+i*signY, -2)
                else:
                    if i < 2 or i > deltaY - 2:
                        if img[start[0]-2, start[1]] == color:
                            newStart = [start[0]-2, start[1]]
                            newStart = (1.0 / FACTOR * (np.array(newStart)) + [int(XMIN - OFFSET),
                                                                                          int(YMIN - OFFSET)])
                            return drawEdge(img, newStart, [endO[0] - 2, endO[1]], color)
                        return
                    curr = [start[0], start[1]+(i-2)*signY]
                    if (img[curr[0], curr[1]] in [0, color] and img[curr[0] - 1, curr[1]] in [0, color]
                            and img[curr[0] - 2, curr[1]] in [0, color] and img[curr[0] - 2, curr[1] + signY] in [0,
                                                                                                                  color]
                            and img[curr[0] - 2, curr[1] + 2 * signY] in [0, color]):
                        if prevColorBefore == 0:
                            img[curr[0], curr[1] + signY] = 0
                        img[curr[0] - 1, curr[1]] = color
                        newStart = [curr[0] - 2, curr[1]]
                        newStart = (1.0 / FACTOR * (np.array(newStart)) + [int(XMIN - OFFSET),
                                                                                      int(YMIN - OFFSET)])
                        newEnd = [end[0] - 2, end[1]]
                        newEnd = (1.0 / FACTOR * (np.array(newEnd)) + [int(XMIN - OFFSET),
                                                                                      int(YMIN - OFFSET)])
                        return drawEdge(img, newStart, newEnd, color)
                    pos = [start[0], start[1] + i * signY + 1 * signY]
                    img[tuple(pos)] = 0
                    return
    else:
        for i in range(deltaX + 1):
            prevColorBefore = prevColor
            prevColor = img[start[0] + i * signX, start[1]]
            if not colorPixel(img, start[0] + i * signX, start[1], color):
                if i == 0 or i == deltaX:
                    continue
                if img[start[0] + (i+1) * signX, start[1]] in [0, color] and img[start[0] + (i+2) * signX, start[1]] in [0, color] and i > 1:
                    # Edge Crossing
                    colorPixel(img, start[0]+i*signX, start[1], -2)
                else:
                    if i < 2 or i > deltaX - 2:
                        # Move whole line
                        if img[start[0], start[1]-2] == color:
                            newStart = [start[0], start[1]-2]
                            newStart = (1.0 / FACTOR * (np.array(newStart)) + [int(XMIN - OFFSET),
                                                                                          int(YMIN - OFFSET)])
                            return drawEdge(img, newStart, [endO[0], endO[1] - 2], color)
                        return
                    curr = [start[0]+(i-2)*signX, start[1]]
                    if (img[curr[0], curr[1]] in [0, color] and img[curr[0], curr[1] - 1] in [0, color]
                            and img[curr[0], curr[1] - 2] in [0, color] and img[curr[0] + signX, curr[1] - 2] in [0,
                                                                                                                  color]
                            and img[curr[0] + 2 * signX, curr[1] - 2] in [0, color]):
                        # Rerouting around small obstacle
                        if prevColorBefore == 0:
                            img[curr[0] + signX, curr[1]] = 0
                        img[curr[0], curr[1] - 1] = color
                        newStart = [curr[0], curr[1] - 2]
                        newStart = (1.0 / FACTOR * (np.array(newStart)) + [int(XMIN - OFFSET),
                                                                                      int(YMIN - OFFSET)])
                        newEnd = [end[0], end[1] - 2]
                        newEnd = (1.0 / FACTOR * (np.array(newEnd)) + [int(XMIN - OFFSET),
                                                                                      int(YMIN - OFFSET)])
                        return drawEdge(img, newStart, newEnd, color)
                    return
    return False

def drawSplitEdge(img, start, end, color1, color2):
    """
    Draws an edge that is color-coded in two colors
    :param img: Image
    :param start: Start pixel
    :param end: End pixel
    :param color1: First color
    :param color2: Second color
    """
    split = [start[0] + int((end[0]-start[0])/3), start[1] + int((end[1]-start[1])/3)]
    if split[0] == start[0]:
        split[1] += 1
    else:
        split[0] += 1
    split = tuple(split)
    first = drawEdge(img, start, split, color1)
    if not first:
        second = drawEdge(img, split, end, color2)
    return first or second

# Gebe boundaryNode ein und bestimmte nächste Grenze darüber
def getBoundaryPoints(prevP, inP, boundaryNode, img, s):
    imgSize = img.shape
    global COUNTER
    p1 = np.copy(inP)
    edgeDir = np.array(inP)-np.array(prevP)
    edgeDir = edgeDir/np.linalg.norm(edgeDir)
    if edgeDir[0] == 0:
        p1[1] -= 2*COUNTER * edgeDir[1] + 1
    else:
        p1[0] -= 2*COUNTER * edgeDir[0] + 1
    p1Access = FACTOR * (np.array(p1) - [int(XMIN - OFFSET), int(YMIN - OFFSET)])
    if not img[tuple(p1Access.astype(int))] in [0, s]:
        p1 -= edgeDir

    res = [p1[0], p1[1]]
    bPos = boundaryNode.pos
    if bPos[0] < imgSize[0]/FACTOR/2:
        if bPos[1] < imgSize[1]/FACTOR/2:
            if bPos[0] < bPos[1]:
                res[0] = -OFFSET
            else:
                res[1] = -OFFSET
        else:
            if bPos[0] < imgSize[1]/FACTOR - bPos[1]:
                res[0] = -OFFSET
            else:
                res[1] = imgSize[1]+OFFSET
    else:
        if bPos[1] < imgSize[1]/FACTOR/2:
            if imgSize[0]/FACTOR - bPos[0] < bPos[1]:
                res[0] = imgSize[0] + OFFSET
            else:
                res[1] = -OFFSET
        else:
            if imgSize[0]/FACTOR - bPos[0] < imgSize[1]/FACTOR - bPos[1]:
                res[0] = imgSize[0] + OFFSET
            else:
                res[1] = imgSize[1]+OFFSET
    COUNTER += 1
    return p1, res

def getMinBoundaryDis(p, imgShape):
    """
    Computes the minimum distance to the boundary
    :param p: coordinates of the point
    :param imgShape: Image Shape
    :return: Minimum distance
    """
    imgShape = np.array(imgShape)/FACTOR
    if p[0] < imgShape[0]/2:
        if p[1] < imgShape[1]/2:
            return min(p[0], p[1])
        else:
            return min(p[0], imgShape[1]-p[1])
    else:
        if p[1] < imgShape[1]/2:
            return min(imgShape[0]-p[0], p[1])
        else:
            return min(imgShape[0]-p[0], imgShape[1]-p[1])

def getBoundaryTarget(img, p, color):
    """
    Find boundary point with the smallest number of pixels in the way
    :param img: Image
    :param p: Starting point
    :return: Point on boundary
    """
    p = np.array(p, dtype=int)
    minNumPixels = np.infty
    target = []
    imgCopy = np.copy(img)
    imgCopy[imgCopy==color] = 0
    # x-direction
    px = np.count_nonzero(imgCopy[:p[0], p[1]])
    if px < minNumPixels:
        minNumPixels = px
        target = [-OFFSET, p[1]]
    px = np.count_nonzero(imgCopy[p[0]:, p[1]])
    if px < minNumPixels:
        minNumPixels = px
        target = [img.shape[0]+OFFSET, p[1]]
    # y-direction
    px = np.count_nonzero(imgCopy[p[0], :p[1]])
    if px < minNumPixels:
        minNumPixels = px
        target = [p[0], -OFFSET]
    px = np.count_nonzero(imgCopy[p[0], p[1]:])
    if px < minNumPixels:
        minNumPixels = px
        target = [p[0], img.shape[1]+OFFSET]
    return target

def rasterizeOrthoGraph(graph):
    """
    Creates a rasterization of an orthogonal graph embedding
    :param graph: The graph to draw
    :param size: The size of the image
    :param offset: The distance of the closest graph node to the boundary
    :return: Image containing a graph drawing
    """
    global FACTOR, NODESIZE, HALFNODESIZE, OFFSET
    print("Degree: " + str(len(graph[-1].edges)))
    FACTOR = max(2, int(math.sqrt(len(graph[-1].edges))))
    NODESIZE = 21 * FACTOR
    HALFNODESIZE = int(NODESIZE / 2)
    OFFSET = 5 + HALFNODESIZE / FACTOR
    size = findEmbeddingSize(graph)
    print("Resolution: " + str(size))
    img = np.zeros(size, dtype=np.int8)
    global COUNTER
    COUNTER = 1

    # Iterate about node
    for n in graph.values():
        if n.id == -1:
            continue
        drawNode(img, n.pos, n.id)

    boundaryEdges = []
    for n in graph.values():
        for e in n.edges.values():
            color1 = e.node1.id
            color2 = e.node2.id
            if color1 == color2:
                continue
            if color1 == -1 or color2 == -1:
                boundaryEdges.append(e)
                continue
            if len(e.bends) == 0:
                continue
            if len(e.bends) == 1:
                drawEdge(img, e.node1.pos, e.bends[0], color1)
                drawEdge(img, e.bends[0], e.node2.pos, color2)
            splitEdgeIndex = int((len(e.bends))/2)
            drawEdge(img, e.node1.pos, e.bends[0], color1)
            for i, b in enumerate(e.bends):
                if i == 0:
                    continue
                if i == splitEdgeIndex:# and i == len(e.bends)-1:
                    #drawEdge(img, e.bends[i - 1], b, color2)
                    drawSplitEdge(img, e.bends[i-1], b, color1, color2)
                else:
                    if i < splitEdgeIndex:
                        drawEdge(img, e.bends[i - 1], b, color1)
                    else:
                        drawEdge(img, e.bends[i - 1], b, color2)
    print("Edge crossings before boundary: " + str(np.count_nonzero(img==-2)))
    # Draw boundary edges
    for e in boundaryEdges:
        color1 = e.node1.id
        color2 = e.node2.id
        if color1 == -1:
            color1 = e.node2.id
        if color2 == -1:
            color2 = e.node1.id
        finished = False
        boundaryNode = e.node1
        if boundaryNode.id != -1:
            boundaryNode = e.node2
        for i, b in enumerate(e.bends):
            if i == 0 or finished:
                continue
            minBoundaryDis = getMinBoundaryDis(b, img.shape)
            # Test layout
            if minBoundaryDis < 2 * NODESIZE / FACTOR + OFFSET:
                p1, p2 = getBoundaryPoints(e.bends[i - 1], b, boundaryNode, img, color1)
                drawEdge(img, e.bends[i - 1], p1, color1)
                drawEdge(img, p1, p2, color1)
                finished = True
                continue
            if i == len(e.bends) - 1:
                p1, p2 = getBoundaryPoints(e.bends[i - 1], b, boundaryNode, img, color1)
                drawEdge(img, e.bends[i - 1], p1, color1)
                drawEdge(img, p1, p2, color1)
                finished = True
                continue
            pAccess = FACTOR * (np.array(b) - [int(XMIN - OFFSET), int(YMIN - OFFSET)])
            if not (img[int(pAccess[0]), int(pAccess[1])] in [0, color1, color2]):
                p1, p2 = getBoundaryPoints(e.bends[i - 1], b, boundaryNode, img, color1)
                drawEdge(img, e.bends[i - 1], p1, color1)
                drawEdge(img, p1, p2, color1)
                finished = True
                continue
            drawEdge(img, e.bends[i - 1], b, color1)
    return img
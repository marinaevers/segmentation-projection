import numpy as np
from skimage import measure

import graphAlgorithms

NEAREST_SEC_FACTOR = 3
DIAGONAL_NEAREST_SEC_FACTOR = 1
SECOND_NEAREST_SEC_FACTOR = 0

def securityFactor(img):
    """
    Compute security factors. Number of point-wise neighbors + 3*number of edge-wise neighbors
    :param img: Image
    :return: Image with security factor for each pixel
    """
    # Edge-wise
    # right
    sec = np.vstack(((img[:-1] == img[1:]).astype(int), np.zeros((1, len(img[0]))))) * NEAREST_SEC_FACTOR
    # left
    sec += np.vstack((np.zeros((1, len(img[0]))), (img[:-1] == img[1:]).astype(int))) * NEAREST_SEC_FACTOR
    # top
    sec += np.hstack((np.zeros((len(img), 1)), (img[:, :-1] == img[:, 1:]).astype(int))) * NEAREST_SEC_FACTOR
    # bottom
    sec += np.hstack(((img[:, :-1] == img[:, 1:]).astype(int), (np.zeros((len(img), 1))))) * NEAREST_SEC_FACTOR
    # Diagonals
    # top-left
    tmp = np.hstack((np.zeros((len(img) - 1, 1)), (img[:-1, :-1] == img[1:, 1:]).astype(int)))
    sec += np.vstack((np.zeros((1, len(img[0]))), tmp)) * DIAGONAL_NEAREST_SEC_FACTOR
    # top-right
    tmp = np.hstack((np.zeros((len(img) - 1, 1)), (img[:-1, 1:] == img[1:, :-1]).astype(int)))
    sec += np.vstack((tmp, np.zeros((1, len(img[0]))))) * DIAGONAL_NEAREST_SEC_FACTOR
    # lower-left
    tmp = np.hstack(((img[:-1, 1:] == img[1:, :-1]).astype(int), (np.zeros((len(img) - 1, 1)))))
    sec += np.vstack((np.zeros((1, len(img[0]))), tmp)) * DIAGONAL_NEAREST_SEC_FACTOR
    # lower-right
    tmp = np.hstack(((img[:-1, :-1] == img[1:, 1:]).astype(int), (np.zeros((len(img) - 1, 1)))))
    sec += np.vstack((tmp, np.zeros((1, len(img[0]))))) * DIAGONAL_NEAREST_SEC_FACTOR
    # Second nearest edgewise
    # right
    sec += np.vstack(((img[:-2] == img[2:]).astype(int), np.zeros((2, len(img[0]))))) * SECOND_NEAREST_SEC_FACTOR
    # left
    sec += np.vstack((np.zeros((2, len(img[0]))), (img[:-2] == img[2:]).astype(int))) * SECOND_NEAREST_SEC_FACTOR
    # top
    sec += np.hstack((np.zeros((len(img), 2)), (img[:, :-2] == img[:, 2:]).astype(int))) * SECOND_NEAREST_SEC_FACTOR
    # bottom
    sec += np.hstack(((img[:, :-2] == img[:, 2:]).astype(int), (np.zeros((len(img), 2))))) * SECOND_NEAREST_SEC_FACTOR
    return sec


def calcDensityBinaryField(dens):
    """
    Determine cells where at least one direct neighbor has higher density
    :param dens: Image with densities
    :return: Binary image where true means that at least on direct neighbor has a higher density
    """
    # Edge-wise
    # right
    grad = np.vstack(((dens[:-1] < dens[1:]), np.zeros((1, len(dens[0])), dtype=bool)))
    # left
    grad = np.logical_or(grad, np.vstack((np.zeros((1, len(dens[0])), dtype=bool), (dens[:-1] > dens[1:]))))
    # top
    grad += np.logical_or(grad, np.hstack((np.zeros((len(dens), 1), dtype=bool), (dens[:, :-1] > dens[:, 1:]))))
    # bottom
    grad += np.logical_or(grad, np.hstack(((dens[:, :-1] < dens[:, 1:]), (np.zeros((len(dens), 1), dtype=bool)))))
    return grad

def checkTopology2(compareTopBottom, compareLeftRight, img):
    # Check for last pixels of connected component (all neighbors need to be different)
    singleLeftover = np.logical_not((img == np.roll(img, -1, axis=1)) + (img == np.roll(img, -1, axis=0)) + \
                                    (img == np.roll(img, 1, axis=1)) + (img == np.roll(img, 1, axis=0)))
    singleLeftover[img == 0] = 0
    changeCounter = np.zeros(img.shape)
    # Count general changes
    # right to lower-right
    changeCounter += np.roll(np.vstack((compareTopBottom[1:], np.ones((1, len(img[0]))))), -1, axis=1)
    changeCounter[-1, -1] -= 1
    # lower-right to bottom
    changeCounter += np.roll(np.hstack((compareLeftRight[:, 1:], np.ones((len(img), 1)))), -1, axis=0)
    changeCounter[-1, -1] -= 1
    # bottom to bottom-left
    changeCounter += np.hstack((compareLeftRight[:, 1:], np.ones((len(img), 1))))
    changeCounter[0, -1] -= 1
    # bottom-left to left
    changeCounter += np.roll(np.vstack((np.ones((1, len(img[0]))), compareTopBottom[:-1])), -1, axis=1)
    changeCounter[0, -1] -= 1
    # left to top-left
    changeCounter += np.vstack((np.ones((1, len(img[0]))), compareTopBottom[:-1]))
    changeCounter[0, 0] -= 1
    # top-left to top
    changeCounter += np.hstack((np.ones((len(img), 1)), compareLeftRight[:, :-1]))
    changeCounter[0, 0] -= 1
    # top to top-right
    changeCounter += np.roll(np.hstack((np.ones((len(img), 1)), compareLeftRight[:, :-1])), -1, axis=0)
    changeCounter[-1, 0] -= 1
    # top-right to right
    changeCounter += np.vstack((compareTopBottom[1:], np.ones((1, len(img[0])))))
    changeCounter[-1, 0] -= 1
    critical = np.logical_or(singleLeftover, changeCounter > 3)
    return critical

# Identify cells that would destroy the topology
def checkTopology(img):
    compareTopBottom = img != np.roll(img, 1, axis=1)
    compareTopBottom = np.hstack((np.zeros((len(img), 1)), compareTopBottom[:, 1:]))
    compareLeftRight = img != np.roll(img, 1, axis=0)
    compareLeftRight = np.vstack((np.zeros((1, len(img[0]))), compareLeftRight[1:]))
    critical = checkTopology2(compareTopBottom, compareLeftRight, img)
    return critical

def checkPairwiseCriticality(g, s1, s2):
    if s1 in [0, -2] or s2 in [0, -2] or s1 == s2:
        return False
    else:
        return not graphAlgorithms.jointEdge(g, s1, s2)

def checkCriticality(g, padded, t, x, y):
    """
    Checks if changing the corresponding pixel
    :param g: Graph
    :param padded: Image with padding
    :param t: Target group
    :param x: x-Coordinate
    :param y: y-Coordinate
    :return: true if critical
    """
    targetPixelCritical =  checkPairwiseCriticality(g, padded[x-1, y], t) or \
        checkPairwiseCriticality(g, padded[x+1, y], t) or \
        checkPairwiseCriticality(g, padded[x, y-1], t) or \
        checkPairwiseCriticality(g, padded[x, y+1], t) or \
        checkPairwiseCriticality(g, padded[x+1, y+1], t) or \
        checkPairwiseCriticality(g, padded[x+1, y-1], t) or \
        checkPairwiseCriticality(g, padded[x-1, y-1], t) or \
        checkPairwiseCriticality(g, padded[x-1, y+1], t)
    if not targetPixelCritical:
        return targetPixelCritical, t
    else:
        candidates = np.unique([padded[x-1, y], padded[x+1, y], padded[x, y-1], padded[x, y+1]])
        candidates = candidates[candidates!=0]
        candidates = candidates[candidates!=t]
        candidates = candidates[candidates!=-2]
        for c in candidates:
            if c not in [t, -1, -2]:
                targetPixelCritical = checkPairwiseCriticality(g, padded[x - 1, y], c) or \
                                      checkPairwiseCriticality(g, padded[x + 1, y], c) or \
                                      checkPairwiseCriticality(g, padded[x, y - 1], c) or \
                                      checkPairwiseCriticality(g, padded[x, y + 1], c) or \
                                        checkPairwiseCriticality(g, padded[x+1, y+1], c) or \
                                        checkPairwiseCriticality(g, padded[x+1, y-1], c) or \
                                        checkPairwiseCriticality(g, padded[x-1, y-1], c) or \
                                        checkPairwiseCriticality(g, padded[x-1, y+1], c)
                if not targetPixelCritical:
                    return targetPixelCritical, c
    return targetPixelCritical, t

def checkTopologyPreservation(critical, target, img, g):
    """
    Remove critical flag from non-critical background
    :param critical: Binary image for critical voxels
    :param target: Target image
    :param img: Image
    :param g: Graph
    :return: Binary image for critical voxels
    """
    indices = np.array(np.nonzero(np.logical_and(img == 0, critical))).T
    padded = np.pad(img.astype(int), pad_width=1, constant_values=-1)
    for [x,y] in indices:
        critical[x, y], target[x, y] = checkCriticality(g, padded, target[x,y], x+1, y+1)
    return critical, target

def calcTargetDens(img, dens):
    # Change to neighbor with maximum density, only consider next neighbors
    maxDens = np.hstack((dens[:, 1:], np.zeros((len(dens), 1))-1))
    target = np.hstack((img[:, 1:], np.zeros((len(img), 1))))
    densRot = np.hstack((np.zeros((len(dens), 1))-1, dens[:, :-1]))
    imgRot = np.hstack((np.zeros((len(img), 1)), img[:, :-1]))
    toChange = densRot > maxDens
    maxDens[toChange] = densRot[toChange]
    target[toChange] = imgRot[toChange]
    densRot = np.vstack((np.zeros((1, len(dens[0])))-1, dens[:-1]))
    imgRot = np.vstack((np.zeros((1, len(img[0]))), img[:-1]))
    toChange = densRot > maxDens
    maxDens[toChange] = densRot[toChange]
    target[toChange] = imgRot[toChange]
    densRot = np.vstack((dens[1:], np.zeros((1, len(dens[0])))-1))
    imgRot = np.vstack((img[1:], np.zeros((1, len(img[0])))))
    toChange = densRot > maxDens
    maxDens[toChange] = densRot[toChange]
    target[toChange] = imgRot[toChange]
    return target

def calcTargetBoundary(img, toChange, axis, dir=-1):
    target = np.copy(img)
    target[toChange] = np.roll(img, dir, axis=axis)[toChange]
    return target

def applyDamping(toChange, maxError, img, dampingFactor = 1):
    mask = np.ones(img.shape)
    mask[img==0] = 0
    probability = min(1, maxError*dampingFactor)
    # The probability of changing should be the defined number
    selection = np.random.rand(*toChange.shape) > 1-probability
    return np.logical_and(selection, toChange)

def calculateDensityField(img, densities):
    """
    Computes the density of the region for each pixel
    :param img: Current image
    :param densities: Dictionary with density for each node
    :return: density image
    """
    # - 1 for distance-based density
    densityField = np.zeros(img.shape) - 1
    for n in densities.keys():
        if n == -1:
            continue
        densityField[img == n] = densities[n]
    return densityField

def updateCheckerboard(checkerboard):
    """
    Moves the adapted checkerboard pattern one step further
    :param checkerboard: The original pattern
    :return: The updated pattern
    """
    if np.sum(checkerboard[0][0] == 0 or checkerboard[1,1] == 0):
        checkerboard = np.roll(checkerboard, 1, axis=0)
    else:
        checkerboard = np.roll(checkerboard, 1, axis=1)
    return checkerboard

def getNeighbors(imgIn, pos, label, id = 1, otherCrossing = []):
    """
    Apply a region growing approach to identify the neighbors
    :param img: Image
    :param pos: Starting position
    :return: list of neighbors
    """
    # Visit all neighbors and if same group but not labeled: call recursive
    numDimensions = len(imgIn.shape)
    s = imgIn[tuple(pos)]
    neighbors = []
    pixels = [tuple(pos)]
    while pixels:
        pos = pixels.pop()
        label[tuple(pos)] = id
        for d in range(numDimensions):
            left = np.copy(pos)
            if pos[d] == 0:
                neighbors += [-1]
            left[d] = max(0, pos[d]-1)
            right = np.copy(pos)
            if pos[d] == imgIn.shape[d]-1:
                neighbors += [-1]
            right[d] = min(imgIn.shape[d]-1, pos[d]+1)
            if imgIn[tuple(left)] == s and label[tuple(left)] == 0 and tuple(left) not in pixels:
                pixels.append(tuple(left))
            else:
                if imgIn[tuple(left)] == -2:
                    otherCrossing.append(left)
                neighbors += [imgIn[tuple(left)]]
            if imgIn[tuple(right)] == s and label[tuple(right)] == 0 and tuple(right) not in pixels:
                pixels.append(tuple(right))
            else:
                if imgIn[tuple(right)] == -2:
                    otherCrossing.append(right)
                neighbors += [imgIn[tuple(right)]]
    return neighbors

COUNTERR1 = 0
COUNTERR2 = 0
def checkAndRemoveDuplicate(img, crossing, s1, s2):
    """
    Checks if the current edge crossing is a duplicate
    :param img: Image
    :param crossing: Coordinates of the crossing
    :param s1: Segment
    :param s2: Other segment
    :return: Image
    """
    global COUNTERR1, COUNTERR2
    otherCrossing = []
    label = np.zeros(img.shape)
    neighbors = getNeighbors(img, [crossing[0]-1, crossing[1]], label, otherCrossing=otherCrossing)
    if len(otherCrossing) == 2:
        neighbors += [img[crossing[0]-1, crossing[1]]]
        for i in range(2):
            neighbors += [img[otherCrossing[i][0]-1, otherCrossing[i][1]]]
            neighbors += [img[otherCrossing[i][0], otherCrossing[i][1]-1]]
        neighbors = list(np.unique(neighbors))
        if sorted(neighbors) == sorted([s1, s2, -2]):
            img[tuple(crossing)] = img[crossing[0]-1, crossing[1]]
            return img, True
    label = np.zeros(img.shape)
    otherCrossing = []
    neighbors = getNeighbors(img, [crossing[0]+1, crossing[1]], label, otherCrossing=otherCrossing)
    if len(otherCrossing) == 2:
        neighbors += [img[crossing[0]+1, crossing[1]]]
        for i in range(2):
            neighbors += [img[otherCrossing[i][0]-1, otherCrossing[i][1]]]
            neighbors += [img[otherCrossing[i][0], otherCrossing[i][1]-1]]
        neighbors = list(np.unique(neighbors))
        if sorted(neighbors) == sorted([s1, s2, -2]):
            img[tuple(crossing)] = img[crossing[0]+1, crossing[1]]
            return img, True
    label = np.zeros(img.shape)
    otherCrossing = []
    neighbors = getNeighbors(img, [crossing[0], crossing[1]-1], label, otherCrossing=otherCrossing)
    if len(otherCrossing) == 2:
        neighbors += [img[crossing[0], crossing[1]-1]]
        for i in range(2):
            neighbors += [img[otherCrossing[i][0]-1, otherCrossing[i][1]]]
            neighbors += [img[otherCrossing[i][0], otherCrossing[i][1]-1]]
        neighbors = list(np.unique(neighbors))
        if sorted(neighbors) == sorted([s1, s2, -2]):
            img[tuple(crossing)] = img[crossing[0], crossing[1]-1]
            return img, True
    label = np.zeros(img.shape)
    otherCrossing = []
    neighbors = getNeighbors(img, [crossing[0], crossing[1]+1], label, otherCrossing=otherCrossing)
    if len(otherCrossing) == 2:
        neighbors += [img[crossing[0], crossing[1]+1]]
        for i in range(2):
            neighbors += [img[otherCrossing[i][0]-1, otherCrossing[i][1]]]
            neighbors += [img[otherCrossing[i][0], otherCrossing[i][1]-1]]
        neighbors = list(np.unique(neighbors))
        if sorted(neighbors) == sorted([s1, s2, -2]):
            img[tuple(crossing)] = img[crossing[0], crossing[1]+1]
            return img, True
    return img, False

def createSegmentation(img):
    """
    Creates a segmentation where each segment has a unique ID,
    background and edge crossings are not included and marked as -1
    :param img: Image
    :return: Segmentation and list of segments with ID, ID in the original image and IDs
    of neighbors (in the original image)
    """
    segmentation = np.zeros(img.shape)
    segmentation[img == -2] = -1
    segmentation[img == 0] = -1
    segmentationList = {}
    id = 1
    while np.count_nonzero(segmentation == 0) > 0:
        pixel = (np.array(np.nonzero(segmentation == 0)).T)[0]
        neighbors = getNeighbors(img, pixel, segmentation, id)
        crossingNumber = np.count_nonzero(np.array(neighbors) == -2)
        segmentationList[id] = (img[tuple(pixel)], crossingNumber,list(np.unique(neighbors)))
        id += 1
    return segmentation, segmentationList

def segmentIsUnnecessary(s, segmentList, img, segmentation):
    """
    Checks if the neighbors are already covered by other segments and this segments only contains
    a single adjacent edge crossing
    :param s: ID of unique segment
    :param segmentList: List of segments
    :param img: Image
    :return: True if the segment can be removed without changing topology
    """
    # Only check if single edge crossing
    if segmentList[s][1] == 1:
        # Check neighbors
        sID = img[segmentation == s][0]
        # Get all neighbors of the other segments
        coveredNeighbors = []
        for t in segmentList.keys():
            if segmentList[t][0] == sID and t != s:
                coveredNeighbors += segmentList[t][2]
        # Check if neighbor list is a subset of the other list
        if not False in np.in1d(segmentList[s][2], coveredNeighbors):
            return True
    return False
def findCrossing(imgIn, pos):
    """
    Determines the coordinate of the edge crossing
    :param img: Image
    :param s: ID of the segment in segmentation
    :param segmentation: Segmentation
    :return: Coordinate of the edge crossing
    """
    label = np.zeros(imgIn.shape)
    label[tuple(pos)] = 1
    # Visit all neighbors and if same group but not labeled: call recursive
    numDimensions = len(imgIn.shape)
    s = imgIn[tuple(pos)]
    queue = [pos]
    label[tuple(pos)] = 1
    while queue:
        q = queue.pop()
        for d in range(numDimensions):
            left = np.copy(q)
            left[d] = max(0, q[d]-1)
            right = np.copy(q)
            right[d] = min(imgIn.shape[d]-1, q[d]+1)
            if imgIn[tuple(left)] == s and label[tuple(left)] == 0:
                label[tuple(left)] = 1
                queue.append(left)
            if imgIn[tuple(left)] < 0:
                return left
            if imgIn[tuple(right)] == s and label[tuple(right)] == 0:
                label[tuple(right)] = 1
                queue.append(right)
            if imgIn[tuple(right)] < 0:
                return right
    print("Reached end of function")

def removeCrossing(img, s, segmentation, id, g):
    """
    Remove segment s and change the crossing
    :param img: Image
    :param s: ID of the segment in segmentation
    :param segmentation: Segmentation
    """
    global COUNTERR2
    startPos = (np.array(np.nonzero(segmentation == s)).T)[0]
    # Remove crossing
    pos = findCrossing(img, startPos)
    a = img[pos[0]-1, pos[1]]
    b = img[pos[0], pos[1]-1]
    if a == id:
        img[tuple(pos)] = b
        if b not in g[a].edges:
            img[pos[0]-1, pos[1]] = 0
            img[pos[0]+1, pos[1]] = 0
    else:
        if b != id:
            print("Problem!")
        img[tuple(pos)] = a
        if a not in g[b].edges:
            img[pos[0], pos[1]-1] = 0
            img[pos[0], pos[1]+1] = 0
    img[segmentation == s] = 0

def removeUnnecessaryCrossings(img, g):
    """
    Remove double edge crossings between two segments if this does not change the topology
    :param img: Image
    :return: Image with removed edge crossings
    """
    indices = np.array(np.nonzero(img == -2)).T
    crossingList = {}
    for i, crossing in enumerate(indices):
        # I can assume that edge crossings are never at the boundary
        s1 = img[crossing[0]-1, crossing[1]]
        s2 = img[crossing[0], crossing[1]-1]
        if (s1, s2) in crossingList or (s2, s1) in crossingList:
            # It is enough to check the neighbours of the newly found edge crossing
            img, removed = checkAndRemoveDuplicate(img, crossing, s1, s2)
            if removed:
                continue
        crossingList[(s1, s2)] = [crossing]
    # Remove unnecessary segments
    label = np.ones(img.shape)
    label[img==0] = 0
    label[img==-2] = 0
    segmentation, segmentList = createSegmentation(img)
    while np.count_nonzero(label) > 0:
        idx = tuple(np.argwhere(label==1)[0])
        s = segmentation[idx]
        if s in segmentList and segmentIsUnnecessary(s, segmentList, img, segmentation) and np.count_nonzero(img ==img[idx]) > 2:
            removeCrossing(img, s, segmentation, segmentList[s][0], g)
            segmentation, segmentList = createSegmentation(img)
        label[segmentation==s] = 0
    return img

def moveCrossing(img):
    """
    Try to move the crossings closer to the barycenter of the segment
    :param img: Image
    :return: Image
    """
    # Iterate over crossings
    indices = np.array(np.nonzero(img == -2)).T
    props = measure.regionprops(img.astype(int))
    for i, crossing in enumerate(indices):
        s1 = int(img[crossing[0]-1, crossing[1]])
        s2 = int(img[crossing[0], crossing[1]-1])
        # Compute barycenters and direction of motion
        if s1 != props[s1-1].label:
            print("Error!")
            print(s1)
            print(props[s1-1].label)
            return img
        dir1 = props[s1-1].centroid - crossing
        dir2 = props[s2-1].centroid - crossing
        if dir1[0] != 0:
            dir1[0]/=abs(dir1[0])
        if dir1[1] != 0:
            dir1[1]/=abs(dir1[1])
        if dir2[0] != 0:
            dir2[0]/=abs(dir2[0])
        if dir2[1] != 0:
            dir2[1]/=abs(dir2[1])
        dir1 = dir1.astype(int)
        dir2 = dir2.astype(int)
        # If possible: move
        if crossing[0] > 1 and crossing[0] < img.shape[0]-2 and \
            img[crossing[0]+2*dir1[0], crossing[1]] == img[crossing[0]+dir1[0], crossing[1]] and \
            img[crossing[0]+dir1[0], crossing[1]-1] == img[crossing[0], crossing[1]-1] and \
            img[crossing[0]+dir1[0], crossing[1]+1] == img[crossing[0], crossing[1]+1]:
            img[crossing[0], crossing[1]] = img[crossing[0]+dir1[0], crossing[1]]
            img[crossing[0]+dir1[0], crossing[1]] = -2
        elif crossing[1] > 1 and crossing[1] < img.shape[1]-2 and \
            img[crossing[0], crossing[1]+2*dir2[1]] == img[crossing[0], crossing[1]+dir2[1]] and \
            img[crossing[0]-1, crossing[1]] == img[crossing[0]-1, crossing[1]+dir2[1]] and \
            img[crossing[0]+1, crossing[1]] == img[crossing[0]+1, crossing[1]+dir2[1]]:
            img[crossing[0], crossing[1]] = img[crossing[0], crossing[1]+dir2[1]]
            img[crossing[0], crossing[1]+dir2[1]] = -2
    return img

def computeBoundarySums(graph, newGraph):
    # Compute normalization values
    newBoundarySum = 0
    oldBoundarySum = 0
    for n in graph.keys():
        for m in graph[n].edges.keys():
            if n == -1:
                continue
            if m < n:
                continue
            try:
                newBoundarySum += newGraph[n].edges[m].boundarySize
                oldBoundarySum += graph[n].edges[m].boundarySize
            except Exception as e:
                print(n)
                print(m)
                continue
    return newBoundarySum, oldBoundarySum

def calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, axis, dir=-1):
    out = np.zeros(img.shape)
    # Iterate over edges
    imgNew = np.pad(img.astype(int), pad_width=1, constant_values=-1)
    rolled = np.roll(imgNew, dir, axis=axis)
    for n in graph.keys():
        if n == -1:
            continue
        for m in graph[n].edges.keys():
            # For each edge: if end node < start node: do nothing
            if m < n:
                continue
            # Otherwise: Calculate if improvement necessary
            newNormalized = newGraph[n].edges[m].boundarySize / newBoundarySum
            oldNormalized = graph[n].edges[m].boundarySize / oldBoundarySum
            #dens = newNormalized/oldNormalized
            dens = oldNormalized - newNormalized
            mask = np.logical_and(imgNew == m, rolled==n)
            mask = np.logical_or(mask, np.logical_and(imgNew == n, rolled==m))
            out[mask[1:-1,1:-1]] = dens
    return out, newBoundarySum

def numNeighbors(img, imgRef):
    """
    Get the number of neighbors of group n
    :param img: Image
    :return: Image where each pixel stores the number of neighbors
    """
    numSame = np.zeros(img.shape)
    numSame[:-1] += (img == np.roll(imgRef, -1, axis=0)).astype(int)[:-1]
    # left
    numSame[1:] += (img == np.roll(imgRef, 1, axis=0)).astype(int)[1:]
    # top
    numSame[:,:-1] += (img == np.roll(imgRef, -1, axis=1)).astype(int)[:,:-1]
    # bottom
    numSame[:,1:]  += (img == np.roll(imgRef, 1, axis=1)).astype(int)[:,1:]
    return numSame
def calculateBoundary(img, boundaryDensity, axis, i = 0, dir=-1):
    delta = numNeighbors(img, img)-numNeighbors(np.roll(img, dir, axis=axis), img)
    return boundaryDensity*delta > 0

def weightTargets(targetDens, densityField, targetBx1, boundaryDensityX1, toChangeBx1, targetBy1, boundaryDensityY1, toChangeBy1,\
                  targetBx2, boundaryDensityX2, toChangeBx2, targetBy2, boundaryDensityY2, toChangeBy2):
    # compute density deviations
    densityFieldD = np.abs(densityField)
    boundaryDensityXD1 = np.abs(boundaryDensityX1)
    boundaryDensityYD1 = np.abs(boundaryDensityY1)
    boundaryDensityXD2 = np.abs(boundaryDensityX2)
    boundaryDensityYD2 = np.abs(boundaryDensityY2)
    target = targetDens
    bd = [boundaryDensityXD1, boundaryDensityYD1, boundaryDensityXD2, boundaryDensityYD2]
    change = [toChangeBx1, toChangeBy1, toChangeBx2, toChangeBy2]
    t = [targetBx1, targetBy1, targetBx2, targetBy2]
    for i in range(len(bd)):
        maxD = bd[i] > densityFieldD
        for j in range(len(bd)):
            if i != j:
                maxD = np.logical_and(maxD, bd[i] > bd[j])
        mask = np.logical_and(maxD, change[i])
        target[mask] = t[i][mask]
    return target

def automaton(img, graph, iterations=300, maxIterWithoutChange=10, dampingFactor=5, secNearest = 3, secSecond=2, moving=True):
    """
    Uses a celular automaton to create a graph visualization
    :param img: Image with graph rasterization
    :param graph: Graph (needed for sizes)
    :param iterations: Number of iterations to execute
    :return: ???
    """
    # Create checkerboard pattern for updates (binary mask)
    np.random.seed(0)
    checkerboard = np.tile(np.array([[0, 1], [1, 1]]), img.shape).astype(bool)
    checkerboard = checkerboard[:len(img), :len(img[0])]
    changed = True # If no changes: algorithm converged
    # Data for densities
    totalAreaTarget = np.count_nonzero(img>0)
    totalAreaGraph = 0
    for n in graph.keys():
        if n == -1:
            continue
        totalAreaGraph += graph[n].area
    densities = {}
    for n in graph.keys():
        if n == -1:
            continue
        densities[n] = graph[n].area / totalAreaGraph - np.count_nonzero(img == n) / totalAreaTarget
    newGraph = graphAlgorithms.createGraph(img)
    newBoundarySum, oldBoundarySum = computeBoundarySums(graph, newGraph)
    boundaryDensityX1, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 0, -1)
    boundaryDensityY1, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 1, -1)
    boundaryDensityX2, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 0, 1)
    boundaryDensityY2, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 1, 1)
    counter = 0
    errors = []
    i = 0
    maxError = 1
    while changed and i < iterations:
        if i % 1000 == 0:
            print(i)
        if np.count_nonzero(img == -2) > 0 and moving:
            img = moveCrossing(img)
        # Calculate the security factors
        # Number of point-wise neighbors + 3*number of edge-wise neighbors
        securityFactors = securityFactor(img)
        # Array with densities
        densityField = calculateDensityField(img, densities)
        # Identify the cells that need change
        # Makes no difference
        toChange = calcDensityBinaryField(densityField)
        # Boundary preservation x-direciton
        toChangeBx1 = calculateBoundary(img, boundaryDensityX1, axis=0, i=i, dir=-1)
        toChangeBx2 = calculateBoundary(img, boundaryDensityX2, axis=0, i=i, dir=1)
        toChangeBy1 = calculateBoundary(img, boundaryDensityY1, axis=1, dir=-1)
        toChangeBy2 = calculateBoundary(img, boundaryDensityY2, axis=1, dir=1)
        toChange = np.logical_or(toChange, toChangeBx1)
        toChange = np.logical_or(toChange, toChangeBx2)
        toChange = np.logical_or(toChange, toChangeBy1)
        toChange = np.logical_or(toChange, toChangeBy2)
        sec = securityFactors < secNearest * NEAREST_SEC_FACTOR + secSecond * DIAGONAL_NEAREST_SEC_FACTOR
        toChange = np.logical_and(toChange, sec)
        # Apply damping
        toChange = applyDamping(toChange, maxError, img, dampingFactor=dampingFactor)
        targetDens = calcTargetDens(img, densityField)
        targetBx1 = calcTargetBoundary(img, toChangeBx1, axis=0, dir=-1)
        targetBy1 = calcTargetBoundary(img, toChangeBy1, axis=1, dir=-1)
        targetBx2 = calcTargetBoundary(img, toChangeBx2, axis=0, dir=1)
        targetBy2 = calcTargetBoundary(img, toChangeBy2, axis=1, dir=1)
        target = weightTargets(targetDens, densityField, targetBx1, boundaryDensityX1, toChangeBx1, targetBy1, boundaryDensityY1, toChangeBy1,
                               targetBx2, boundaryDensityX2, toChangeBx2, targetBy2, boundaryDensityY2, toChangeBy2)
        # Preserve topology
        criticalCells = checkTopology(img)
        criticalCells, target = checkTopologyPreservation(criticalCells, target, img, graph)
        toChange[criticalCells] = False
        checkerboard = updateCheckerboard(checkerboard)
        toChange[checkerboard] = False
        # Change
        changes = np.count_nonzero(img[toChange]!=target[toChange])
        img[toChange] = target[toChange]
        # Update the densities
        totalAreaTarget = np.count_nonzero(img > 0)
        for n in graph.keys():
            if n == -1:
                continue
            densities[n] = graph[n].area / totalAreaGraph - np.count_nonzero(img == n) / totalAreaTarget
        newGraph = graphAlgorithms.createGraph(img)
        newBoundarySum, oldBoundarySum = computeBoundarySums(graph, newGraph)
        boundaryDensityX1, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 0, -1)
        boundaryDensityY1, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 1, -1)
        boundaryDensityX2, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 0, 1)
        boundaryDensityY2, totalBoundaryTarget = calculateBoundaryDensity(img, graph, newGraph, newBoundarySum, oldBoundarySum, 1, 1)
        # Check for convergence
        changed = changes > 0
        if (changed and counter > 0):
            counter = 0
        densityField = calculateDensityField(img, densities)
        maxError = np.max(np.abs(np.unique(densityField[img>0])))
        maxError = max(maxError, np.max(np.abs(np.unique(boundaryDensityX1))))
        maxError = max(maxError, np.max(np.abs(np.unique(boundaryDensityY1))))
        maxError = max(maxError, np.max(np.abs(np.unique(boundaryDensityX2))))
        maxError = max(maxError, np.max(np.abs(np.unique(boundaryDensityY2))))
        errors.append(maxError)
        if maxError < max(1.0/totalAreaTarget, 1.0/totalBoundaryTarget) and np.count_nonzero(img == 0) == 0:
            print("Converged!")
            changed = False
            continue
        if not changed and counter < maxIterWithoutChange:
            counter += 1
            changed = True
        if np.count_nonzero(img == -2) > 0 and (i % 300) == 0 and np.count_nonzero(img == 0) < 0.1*img.size:
            print("Try removing at i = " + str(i))
            img = removeUnnecessaryCrossings(img, graph)
        # Increase counter
        i = i + 1
    print("Area: " + str(1.0 / totalAreaTarget))
    print("Boundary: " + str(1.0 / totalBoundaryTarget))
    return errors, i
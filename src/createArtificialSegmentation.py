import numpy as np

np.random.seed(0)

def calculateDensityField(img, densities):
    """
    Computes the density of the region for each pixel
    :param img: Current image
    :param densities: Dictionary with density for each node
    :return: density image
    """
    densityField = np.zeros(img.shape)
    for i in range(len(densities)):
        densityField[img == i+1] = densities[i]
    return densityField

def calcTarget(img, dens):
    # Change to neighbor with maximum density, only consider next neighbors
    numDimensions = len(img.shape)
    maxDens = np.zeros(img.shape)
    target = np.zeros(img.shape)
    for d in range(numDimensions):
        sliceShape = np.array(img.shape)
        sliceShape[d] = 1
        sliceShape = tuple(sliceShape)
        densRot = np.concatenate((dens.take(np.arange(1, img.shape[d]), axis=d), np.zeros(sliceShape)), axis = d)
        targetRot = np.concatenate((img.take(np.arange(1, img.shape[d]), axis=d), np.zeros(sliceShape)), axis = d)
        toChange = densRot > maxDens
        maxDens[toChange] = densRot[toChange]
        target[toChange] = targetRot[toChange]
        densRot = np.concatenate((np.zeros(sliceShape), dens.take(np.arange(0, img.shape[d]-1), axis=d)), axis = d)
        targetRot = np.concatenate((np.zeros(sliceShape), img.take(np.arange(0, img.shape[d]-1), axis=d)), axis = d)
        toChange = densRot > maxDens
        maxDens[toChange] = densRot[toChange]
        target[toChange] = targetRot[toChange]
    return target

def createSegmentation(numDims, numSegments, res = 64):
    """
    Creates a segmentations in a domain of the given dimension
    with the selected number of segments
    :param numDims: Number of dimensions
    :param numSegments: Number of segments
    :param res: Resolution per dimension
    :return: Dataset
    """
    shape = tuple(np.ones(numDims, dtype=int)*res)
    data = np.zeros(shape)
    # Sampling positions for creation of segments
    seeds = (np.random.rand(numSegments, numDims)*res).astype(int)
    # Probability to grow for each segment
    probs = np.random.rand(numSegments)
    # Seed data
    for i, s in enumerate(seeds):
        data[tuple(s)] = i+1
    # Grow regions
    while np.count_nonzero(data == 0):
        # Create target volume
        dens = calculateDensityField(data, probs)
        target = calcTarget(data, dens)
        # Assign probabilities to neighboring voxels
        probVol = np.zeros(data.shape)
        for i in range(numSegments):
            probVol[target == (i+1)] = probs[i]
        # Create random numbers, if they are larger than probability: assign value of target volume
        toChange = np.logical_and(probVol > np.random.rand(*data.shape), data == 0)
        data[toChange] = target[toChange]
    return data

def createOctree(numDims, res, randomize=False):
    """
    Creates a segmentations in a domain of the given dimension
    in form of an octree. The first two dimensions are partitioned,
    the others not. In case of 2D, use a quadtree. The size of the segments
    can be set randomly
    :param numDims: Number of dimensions
    :param res: Resolution per dimension
    :return: Dataset
    """
    shape = tuple(np.ones(numDims, dtype=int) * res)
    data = np.zeros(shape)
    if randomize:
        bound = (np.random.rand(7)*res).astype(int)
    else:
        bound = np.ones(7, dtype=int)*int(res/2)
    print(bound)
    if numDims == 2:
        data[:bound[0],bound[1]:] = 1
        data[bound[0]:,:bound[2]] = 2
        data[bound[0]:,bound[2]:] = 3
    else:
        data[:bound[0],bound[1]:,:bound[3]] = 1
        data[:bound[0],bound[1]:,bound[3]:] = 4
        data[bound[0]:,:bound[2],:bound[4]] = 2
        data[bound[0]:,:bound[2],bound[4]:] = 5
        data[bound[0]:,bound[2]:,:bound[5]] = 3
        data[bound[0]:,bound[2]:,bound[5]:] = 6
        data[:bound[0],:bound[1],bound[6]:] = 7
    return data+1
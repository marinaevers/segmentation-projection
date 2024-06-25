import numpy as np
import graphAlgorithms
import matplotlib.pyplot as plt

def areaDeviations(emb, g):
    """
    Computes the area deviation for each segment
    :param emb: Embedding
    :param g: Graph
    :return: Deviation of areas
    """
    totalAreaTarget = np.count_nonzero(emb>0)
    totalAreaGraph = 0
    for n in g.keys():
        if n == -1:
            continue
        totalAreaGraph += g[n].area
    densities = []
    for n in g.keys():
        if n == -1:
            continue
        densities.append(g[n].area / totalAreaGraph - np.count_nonzero(emb == n) / totalAreaTarget)
    return np.array(densities)

def boundaryDeviations(emb, g):
    """
    Computes the relative boundary deviation for each edge
    :param emb: Embedding
    :param g: Graph
    :return: Deviation of boundaries
    """
    gEmb = graphAlgorithms.createGraph(emb)
    totalBoundaryTarget = 0
    totalBoundaryGraph = 0
    for n in g.keys():
        for e in g[n].edges.keys():
            if n == -1:
                continue
            if e < n:
                continue
            totalBoundaryGraph += g[n].edges[e].boundarySize
            totalBoundaryTarget += gEmb[n].edges[e].boundarySize
    deviations = []
    for n in g.keys():
        if n == -1:
            continue
        for e in g[n].edges:
            if e < n:
                continue
            deviations.append(g[n].edges[e].boundarySize/totalBoundaryGraph - gEmb[n].edges[e].boundarySize/totalBoundaryTarget)
    return np.array(deviations)

def totalBoundaryLength(emb, g):
    gEmb = graphAlgorithms.createGraph(emb)
    totalBoundaryTarget = 0
    for n in g.keys():
        for e in g[n].edges.keys():
            if n == -1:
                continue
            if e < n:
                continue
            totalBoundaryTarget += gEmb[n].edges[e].boundarySize
    return totalBoundaryTarget


def maxAreaDeviation(emb, g):
    """
    Computes the maximum deviation of the area
    :param emb: Embedding
    :param g: Graph
    :return: Maximum deviation of area
    """
    deviations = areaDeviations(emb, g)
    maxVal = deviations[np.argmax(np.abs(deviations))]
    return maxVal

def meanAreaDeviation(emb, g):
    """
    Computes the mean deviation of the area
    :param emb: Embedding
    :param g: Graph
    :return: Mean deviation of area
    """
    deviations = areaDeviations(emb, g)
    return np.mean(np.abs(deviations))

def numberOfEdgeCrossings(emb):
    """
    Computes the number of edge crossings
    :param emb: Embedding
    :return: Number of edge crossings
    """
    return np.count_nonzero(emb == -2)

def averageBoundarySizePerSegment(emb):
    """
    Compute the average boundary size per segment
    :param emb: Embedding
    :return: Average boundary size
    """
    g = graphAlgorithms.createGraph(emb)
    boundarySize = 0
    counter = 0
    for n in g:
        for e in g[n].edges:
            boundarySize += g[n].edges[e].boundarySize
            counter += 1
    return boundarySize/counter

def maxBoundaryDeviation(emb, g):
    """
    Compute the maximum relative deviation of the boundary
    :param emb: Embedding
    :param g: Graph
    :return: Maximum deviation of boundary
    """
    deviations = boundaryDeviations(emb, g)
    maxVal = deviations[np.argmax(np.abs(deviations))]
    return maxVal

def meanBoundaryDeviation(emb, g):
    """
    Compute the mean relative deviation of the boundary
    :param emb: Embedding
    :param b: Graph
    :return: Mean deviation of boundary
    """
    deviations = boundaryDeviations(emb, g)
    return np.mean(np.abs(deviations))

def numberOfBackgroundVoxels(emb):
    return np.count_nonzero(emb == 0)

def visualizeQualityEvaluation(x, metrics, filename):
    """
    Create a visual overview about the given metrics
    Currently, the visualization does support 9 metrics at most
    :param x: Varied parameter that should be shown on the x-axis
    :param metrics: Dictionary of metrics
    :param filename: Filename to save the result
    :return: None
    """
    if len(metrics) < 3:
        xDim = len(metrics)
        yDim = 1
    elif len(metrics) <= 6:
        xDim = 3
        yDim = 2
    elif len(metrics) <= 9:
        xDim = 3
        yDim = 3
    else:
        xDim = 3
        yDim = 4
    fig, axs = plt.subplots(xDim, yDim, figsize=(3*xDim, 3*yDim))
    for i, m in enumerate(metrics):
        axs[i%3, int(i/3)].set_title(m)
        axs[i%3, int(i/3)].plot(x, metrics[m])
    fig.tight_layout()
    plt.savefig(filename)
    plt.show()

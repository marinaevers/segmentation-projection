import time

import graphAlgorithms
import graphRasterization
import automaton
import numpy as np
import utils
import qualityMetrics
import matplotlib.pyplot as plt

def remove_consecutive_duplicates(arr):
    #f = 5
    # Compute pairwise differences between consecutive rows
    diff_rows = np.diff(arr, axis=0)

    # Add a row of True values to the top of the diff_rows array
    diff_rows = np.vstack((np.ones((1, arr.shape[1]), dtype=bool), diff_rows))

    # Compute the row indices to keep
    keep_rows = np.logical_or.reduce(diff_rows, axis=1)
    #keep_rows[1::f] = True

    # Compute pairwise differences between consecutive columns
    diff_cols = np.diff(arr, axis=1)

    # Add a column of True values to the left of the diff_cols array
    diff_cols = np.hstack((np.ones((arr.shape[0], 1), dtype=bool), diff_cols))

    # Compute the column indices to keep
    keep_cols = np.logical_or.reduce(diff_cols, axis=0)
    #keep_cols[1::f] = True

    # Use boolean indexing to select the rows and columns to keep
    result = arr[keep_rows, :]
    result = result[:, keep_cols]

    # Make sure that the image dimension is not uneven as the automaton will not work in this case
    # utils.draw(result, "img", "resultBefore")
    if result.shape[0] % 2 != 0:
        result = np.vstack(([result[0]], result))
    if result.shape[1] % 2 != 0:
        result = np.hstack((np.array([result[:,0]]).T, result))
    return result

def graphEmbedding(g, v1, minEdgeCrossings, img):
    # 2. Embed graph
    embedding = graphAlgorithms.embedGraph(g, v1)
    # 3. Rasterize embedding
    imgTry = graphRasterization.rasterizeOrthoGraph(embedding)
    imgTry = remove_consecutive_duplicates(imgTry)
    gTest = graphAlgorithms.createGraph(imgTry)
    preserved = graphAlgorithms.topologyTest(g, gTest)
    crossings = np.count_nonzero(imgTry == -2)
    if crossings > 0 and preserved:
        automaton.removeUnnecessaryCrossings(imgTry, g)
    crossings = np.count_nonzero(imgTry == -2)
    print("Crossings: " + str(crossings))
    if preserved and crossings <= minEdgeCrossings:
        minEdgeCrossings = crossings
        img = imgTry
    return img, minEdgeCrossings, embedding

def embedSegmentation(data, iter=200, dampingFactor = 3, secNearest = 3, secSecond = 2):
    # 1. Create graph datastructure
    startGraph = time.time()
    g = graphAlgorithms.createGraph(data)
    print("Time for graph: " + str(time.time()-startGraph))
    startEmbedding = time.time()
    i = 0
    minEdgeCrossings = np.inf
    img = []
    key = list(g[-1].edges.keys())[-1]
    v1 = key-1
    img, minEdgeCrossings, e = graphEmbedding(g, v1, minEdgeCrossings, img)
    while i < len(g[-1].edges) and minEdgeCrossings > 0:
        key = list(g[-1].edges.keys())[i]
        v1 = key-1
        try:
            img, minEdgeCrossings, e = graphEmbedding(g, v1, minEdgeCrossings, img)
        except Exception as e:
            print(e)
        i += 1
    print("Time for embedding: " + str(time.time()-startEmbedding))
    print("Final resolution: " + str(img.shape))
    # 3B: Check if graph embedding is correct (just a double check)
    gTest = graphAlgorithms.createGraph(img)
    if (not graphAlgorithms.topologyTest(g, gTest)):
        print("Something went wrong with the graph embedding!")
        exit()

    # 4. Cellular Automaton
    startAutomaton = time.time()
    errors, i = automaton.automaton(img, g, iter, dampingFactor=dampingFactor, secNearest=secNearest, secSecond=secSecond)
    print("Time for automaton: " + str(time.time()-startAutomaton))
    crossingsAfter = np.count_nonzero(img==-2)
    gTest = graphAlgorithms.createGraph(img)
    if (not graphAlgorithms.topologyTest(g, gTest)):
        utils.draw(img, "img", "graphWrong")
        print("Something went wrong with the graph embedding!")
        exit()

    print("--- Statistics ---")
    print("Number of iterations:" + str(i))
    print("Maximum Area Deviation: " + str(qualityMetrics.maxAreaDeviation(img, g)))
    print("Mean Area Deviation: " + str(qualityMetrics.meanAreaDeviation(img, g)))
    print("Maximum Boundary Deviation: " + str(qualityMetrics.maxBoundaryDeviation(img, g)))
    print("Mean Boundary Deviation: " + str(qualityMetrics.meanBoundaryDeviation(img, g)))
    print("Edge Crossings: " + str(qualityMetrics.numberOfEdgeCrossings(img)))
    print("Number of boundary pixels: " + str(np.count_nonzero(img==0)))
    return img, minEdgeCrossings, crossingsAfter, g, i
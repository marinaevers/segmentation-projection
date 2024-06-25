from graph import Node, Edge
import graphDrawing as gd
import numpy as np
from ctypes import *

lib1 = cdll.LoadLibrary('libraries/libOGDF.so')
lib2 = cdll.LoadLibrary('libraries/libCOIN.so')
import cppimport
funcs = cppimport.imp("embedGraph")

def createGraph(field):
    graph = {}
    # add boundary as set of voxels in all directions
    field = np.pad(field.astype(int), pad_width=1, constant_values=-1)

    # Add nodes
    segments, counts = np.unique(field, return_counts=True)
    for c, s in enumerate(segments):
        if s == -2 or s == 0:
            continue
        graph[s] = Node(s, counts[c])
    # Remove size of boundary node
    graph[-1].area = 0

    # Add edges
    numDimensions = len(field.shape)
    for d in range(numDimensions):
        diff = np.abs(np.diff(field, axis=d))
        it = np.nditer(diff, flags=['multi_index'])
        for x in it:
            if (x > 0):
                first = it.multi_index
                second = list(it.multi_index)
                second[d] += 1
                second = tuple(second)
                firstID = field[first]
                secondID = field[second]
                if(firstID == 0 or firstID == -2 or secondID == 0 or secondID == -2):
                    continue
                graph[firstID].addEdge(secondID, graph[secondID])
                graph[secondID].addEdge(firstID, graph[firstID])
    return graph

def printGraph(graph):
    for node in graph.values():
        print("Node: " + str(node.id) + " with area " + str(node.area))
        for edge in node.edges.values():
            print("Edge: From " + str(edge.node1.id) + " to " + str(edge.node2.id) + " with size " + str(edge.boundarySize))

def printGraphWithEmbedding(graph):
    for node in graph.values():
        print("Node: " + str(node.id) + " with position " + str(node.pos))

def createAdjacencyMatrix(graph):
    """
    Forms an adjacency matrix from the graph.
    The diagonal contains the information if the corresponding node has a connection to the domain boundary
    :param graph: The input graph
    :return: Adjacency matrix
    """
    ad = np.zeros((len(graph)-1, len(graph)-1), dtype=int)
    mapping = {}
    i = 0
    for node in graph.values():
        if node.id == -1:
            continue
        mapping[node.id] = i
        i += 1
    for node in graph.values():
        if node.id == -1:
            continue
        for edge in node.edges.values():
            if edge.node1.id != -1 and edge.node2.id != -1:
                ad[mapping[edge.node1.id], mapping[edge.node2.id]] = 1
            if edge.node1.id == -1:
                ad[mapping[edge.node2.id], mapping[edge.node2.id]] = 1
            if edge.node2.id == -1:
                ad[mapping[edge.node1.id], mapping[edge.node1.id]] = 1
    return ad

def embedGraph(graph, v1):
    adjacencyMatrix = createAdjacencyMatrix(graph)
    embedding = funcs.embed(adjacencyMatrix, v1)
    embeddedGraph = gd.embedOrthoLayout(graph, embedding)
    return embeddedGraph


def equal(g, h):
    """
    Compares to graphs
    :param g: first graph
    :param h: second graph
    :return: True if the two graphs are equal
    """
    if(len(g.keys()) != len(h.keys())):
        print("Unequal number of nodes")
        return False
    for n in g.keys():
        if not n in h.keys():
            print("Node missing: " + str(n))
            return False
        if(len(g[n].edges) != len(h[n].edges)):
            print("Unequal number of edges for node " + str(n))
            return False
        for e in g[n].edges:
            if not e in h[n].edges:
                print("Edge between " + str(g[n].id) + " and " + str(g[e].id) + " missing.")
                return False
    return True

def jointEdge(graph, s1, s2):
    """
    Determines if the segments s1 and s2 have a joint boundary
    :param graph: Graph
    :param s1: Segment 1
    :param s2: Segment 2
    :return: True if they share a joint boundary
    """
    if s1 in [0,-2] or s2 in [0,-2]:
        return False
    return s2 in graph[s1].edges.keys() or s1 in graph[s2].edges.keys()

def removeNode(g, n):
    """
    Removes the node including all edges
    :param g: Graph
    :param n: Index of the node
    """
    if n in g:
        del g[n]
    for v in g.keys():
        edgesToDelete = []
        for eId in g[v].edges.keys():
            e = g[v].edges[eId]
            if e.node1.id == n or e.node2.id == n:
                edgesToDelete.append(eId)
        for id in edgesToDelete:
            del g[v].edges[id]

def topologyTest(g1, g2):
    """
    Tests if the graphs have the same topological information
    :param g1: Graph 1
    :param g2: Graph 2
    :return: True if graphs describe the same topology
    """
    # Remove edge crossings
    removeNode(g1, -2)
    removeNode(g2, -2)
    # Remove background
    removeNode(g1, 0)
    removeNode(g2, 0)
    return equal(g1, g2)

def countEdges(g):
    sum = 0
    for v in g.keys():
        sum += len(g[v].edges)
    return sum/2
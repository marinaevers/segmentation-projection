class Node:
    id = 0
    edges = {}
    area = 0
    boundaryDirection = []

    def __init__(self, id, area):
        self.id = int(id)
        self.area = area
        self.edges = {}
        self.pos = [0,0]

    def addEdge(self, targetID, targetNode):
        if targetID in self.edges:
            self.edges[targetID].boundarySize += 1
        else:
            # print("Add edge: " + str(self.id) + "-" +  str(targetID))
            self.edges[targetID] = Edge(self, targetNode)

    def addExistingEdge(self, edge):
        targetID = edge.node2.id
        if targetID in self.edges:
            self.edges[targetID].boundarySize += edge.boundarySize
        else:
            if targetID != self.id:
                self.edges[targetID] = Edge(self, edge.node2, edge.boundarySize)

class Edge:
    node1 = None
    node2 = None
    boundarySize = 0
    bends = []

    def __init__(self, node1, node2, size=1):
        self.node1 = node1
        self.node2 = node2
        self.boundarySize = size

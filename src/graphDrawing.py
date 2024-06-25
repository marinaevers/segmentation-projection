def embedOrthoLayout(graph, embedding):
    """
    Takes a given embedding and adds the information to the graph data structure
    :param graph: Graph to embed
    :param embedding: Embedding of the graph
    :return: Graph with embedding
    """
    # Create same mapping as for adjacency matrix
    mapping = {}
    i = 0
    for node in graph.values():
        if node.id == -1:
            mapping[node.id] = len(graph.values())-1
            continue
        mapping[node.id] = i
        i += 1

    for n in graph.values():
        index = mapping[n.id]
        n.pos = list(embedding[index][index][0])
        for edge in n.edges.values():
            i1 = mapping[edge.node1.id]
            i2 = mapping[edge.node2.id]
            edge.bends = list(embedding[i1][i2])
    return graph


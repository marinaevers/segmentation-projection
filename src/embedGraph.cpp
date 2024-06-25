#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#ifdef OGDF_DEBUG
    #undef OGDF_DEBUG
#endif

#include <ogdf/planarity/PlanarizerMixedInsertion.h>
#include <ogdf/planarity/PlanarSubgraphTriangles.h>
#include <ogdf/planarity/MaximumPlanarSubgraph.h>
#include <ogdf/fileformats/GraphIO.h>
#include <ogdf/planarity/PlanarizationLayout.h>
#include <ogdf/planarity/EmbedderMaxFace.h>
#include <ogdf/orthogonal/OrthoLayout.h>
#include <ogdf/planarity/EmbedderBoundaryNode.h>
#include <ogdf/planarity/SubgraphPlanarizer.h>
#include <ogdf/planarity/PlanarSubgraphFast.h>
#include "EmbedderBoundaryCycle.h"

#include <random>

namespace py = pybind11;

using namespace pybind11::literals;

// Input: adjacency matrix, store on diagonal if node has a connection to the boundary
std::vector<std::vector<std::vector<std::pair<double, double>>>> embed(std::vector<std::vector<int>> adjacencyMatrix, int v1) {
    // Create graph from adjacency matrix
    srand(0);
    std::cout << "Start C++ Part" << std::endl;
    ogdf::Graph G;
    std::vector<ogdf::node> nodes;
    for(size_t i = 0; i < adjacencyMatrix.size(); i++) {
        nodes.push_back(G.newNode());
    }
    for(size_t i = 0; i < adjacencyMatrix.size(); i++) {
        for(size_t j = 0; j < i; j++) {
            // undirected graph, adding the edges once is sufficient
            if(adjacencyMatrix[i][j] == 1 || adjacencyMatrix[j][i] == 1) {
                G.newEdge(nodes[i], nodes[j]);
            }
        }
    }
    // Add boundary node and edges to boundary
    ogdf::node boundaryNode = G.newNode();
    size_t numNodes = adjacencyMatrix.size();
    for(size_t i = 0; i < numNodes; i++) {
        if(adjacencyMatrix[i][i] == 1)
            G.newEdge(nodes[i], boundaryNode);
    }

    std::cout << "Start embedding" << std::endl;
    std::vector<std::vector<std::vector<std::pair<double, double>>>> result;
    try
    {
        // Duplicate boundary node to ensure in embedding that it is on the outside
        ogdf::GraphAttributes GA(G, ogdf::GraphAttributes::nodeGraphics | ogdf::GraphAttributes::edgeGraphics);// | ogdf::GraphAttributes::edgeType |
        // Stores attributes like positions
        ogdf::PlanarizationLayout PL;
        // Planarization layout with best performance in Chimani 2021 paper
        ogdf::PlanarizerMixedInsertion *mim = new ogdf::PlanarizerMixedInsertion;
        // Algorithm by Chalermsook and Schmid (used in evaluation paper)​
        ogdf::PlanarSubgraphFast<int>* psf = new ogdf::PlanarSubgraphFast<int>;
       mim->setSubgraph(psf);
        PL.setCrossMin(mim);
        ogdf::EmbedderBoundaryNode *emb = new ogdf::EmbedderBoundaryNode;
        emb->setTargets(boundaryNode, nodes[v1]);
        PL.setEmbedder(emb);

        ogdf::OrthoLayout *OL = new ogdf::OrthoLayout;
        OL->cOverhang(0.0);
        PL.setPlanarLayouter(OL);
        PL.call(GA);

        std::cout << "Embedding done" << std::endl;

        // Fill result data with empty vectors
        for(size_t i = 0; i < adjacencyMatrix.size()+1; i++) {
            result.push_back(std::vector<std::vector<std::pair<double, double>>>());
            for(size_t j = 0; j < adjacencyMatrix.size()+1; j++) {
                result[i].push_back(std::vector<std::pair<double, double>>());
            }
        }
        int i = 0;
        for(ogdf::node n : G.nodes) {
            result[i][i].push_back(std::pair<double, double>(GA.x(n), GA.y(n)));
            i++;
        }

        std::cout << "Created result" << std::endl;


        // Find bending points of the edges
        for(ogdf::edge e : G.edges) {
            int sourceIndex = e->source()->index();
            int targetIndex = e->target()->index();

            ogdf::DPolyline line = GA.bends(e);
            for(ogdf::ListConstIterator<ogdf::DPoint> it = line.begin(); it.valid(); it++) {
                ogdf::DPoint p1 = *it;
                result[sourceIndex][targetIndex].push_back(std::pair<double, double>(p1.m_x, p1.m_y));
            }
        }

    }
    catch (const std::exception& e) // reference to the base of a polymorphic object
    {
        std::cout << e.what(); // information from length_error printed
    }
    return result;
}


PYBIND11_MODULE(embedGraph, m) {
    m.def("embed", &embed, "A function that embeds a graph");
}


<%
cfg['include_dirs'] = ['../ogdf/include/']
cfg['extra_link_args'] = ['-Wl,-rpath,$ORIGIN','-L/home/marina/Documents/Forschung/Münster/AbstractParameterSpaceVisualization/segmentation-projection/ogdf/']
cfg['libraries'] = ['OGDF', 'COIN']
setup_pybind11(cfg)
%>

//'-L/home/marina/Dokumente/Promotion/EigenePaper/AbstractParameterSpaceVisualization/segmentation-projection/ogdf/'
//cfg['extra_link_args'] = ['-Wl,-rpath,$ORIGIN','-L/data2/Sciebo/Promotion/EigenePaper/AbstractParameterSpaceVisualization/segmentation-projection/ogdf/buildDesktop/']
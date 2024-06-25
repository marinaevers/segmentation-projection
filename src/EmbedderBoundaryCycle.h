#include <ogdf/planarity/EmbedderModule.h>

namespace ogdf {
class OGDF_EXPORT EmbedderBoundaryCycle : public EmbedderModule {
private:
    ogdf::node targetNode;
    node v1;
    node v2;
public:
    void setTargets(node& boundary, node& v1, node& v2) {
        this->targetNode = boundary;
        this->v1 = v1;
        this->v2 = v2;
    }

    virtual void doCall(Graph& G, adjEntry& adjExternal) {
        PlanRep pr = static_cast<PlanRep&>(G);
        node newTarget = targetNode;
        node newV1 = v1;
        node newV2 = v2;
        for(node n : G.nodes) {
            if(pr.v(n->index()) == targetNode)
                newTarget = n;
            if(pr.v(n->index()) == v1)
                newV1 = n;
            if(pr.v(n->index()) == v2)
                newV2 = n;
        }
        targetNode = newTarget;
        v1 = newV1;
        v2 = newV2;

        // Create combinatorial embedding
        if(!G.representsCombEmbedding())
            planarEmbed(G);

        CombinatorialEmbedding CE(G);
        PlanRep PR(G);

        // Find face with all three nodes
        for(face f : CE.faces) {
            adjExternal = f->firstAdj();
            bool v1Found = false;
            bool v2Found = false;
            adjEntry curr = adjExternal;
            for(int i = 0; i < f->size(); i++) {
                if(curr->theNode() == v1) {
                    v1Found = true;
                }
                if(curr->theNode() == v2) {
                    v2Found = true;
                }
                if(v1Found && v2Found) {
                    std::cout << adjExternal << std::endl;
                    return;
                }
                std::cout << curr << std::endl;
                std::cout << curr->theNode() << std::endl;
                curr = f->nextFaceEdge(curr);
            }
        }

        std::cout << "Not found" << std::endl;
    }
};
}
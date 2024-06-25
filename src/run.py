import numpy as np

import segmentationEmbedding
import utils
import createArtificialSegmentation

np.random.seed(0)

data = createArtificialSegmentation.createSegmentation(numDims=3, numSegments=7, res=50)
img, _, _, _, _ = segmentationEmbedding.embedSegmentation(data, dampingFactor=7,
                                                                          secNearest=3,
                                                                          secSecond=2, iter=1000)
print("Embedding done")
utils.draw(img, "../output/", "debug", img=img)
print("Done")

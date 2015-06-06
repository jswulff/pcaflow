#! /usr/bin/env python2

from pcaflow import PCAFlow

# To read images
from scipy.misc import imread

# To display
from matplotlib import pyplot as plt
from pcaflow.utils.viz_flow import viz_flow

PATH_PC_U = 'data/PC_U.npy'
PATH_PC_V = 'data/PC_V.npy'
PATH_COV = 'data/COV_SINTEL.npy'
PATH_COV_SUBLAYER = 'data/COV_SINTEL_SUBLAYER.npy'


### Compute using PCA-Flow.

#P = PCAFlow.PCAFlow(
#        pc_file_u=PATH_PC_U,
#        pc_file_v=PATH_PC_V,
#        covfile=PATH_COV,
#        preset='pcaflow_sintel',
#        )
#

### Compute using PCA-Layers.
P = PCAFlow.PCAFlow(
        pc_file_u=PATH_PC_U,
        pc_file_v=PATH_PC_V,
        covfile=PATH_COV,
        covfile_sublayer=PATH_COV_SUBLAYER,
        preset='pcalayers_sintel',
        )


### Once the object is created, it can be used like this:
I1 = imread('image1.png')
I2 = imread('image2.png')

P.push_back(I1)
P.push_back(I2)

# Compute flow
u,v = P.compute_flow()

### Use this if you want to just get the motion descriptor
#u,v,data = P.compute_flow(return_additional=['weights',])
#descriptor = data['weights']

I_flow = viz_flow(u,v)

plt.figure()
plt.subplot(221)
plt.imshow(I1)
plt.title('First image')
plt.subplot(222)
plt.imshow(I_flow)
plt.title('Flow colormap')
plt.subplot(223)
plt.imshow(u)
plt.title('Horizontal component')
plt.subplot(224)
plt.imshow(v)
plt.title('Vertical component')

plt.show()



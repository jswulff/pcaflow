#! /usr/bin/env python2

import numpy as np
import os,sys

# To read images
from scipy.misc import imread,imsave

from pcaflow import PCAFlow
from pcaflow.utils.viz_flow import viz_flow
from pcaflow.utils.flow_io import flow_read,flow_write

def main():
    if len(sys.argv) < 4:
        print('\tUSAGE: compute_flow.py [PARAMETER] IMAGE1 IMAGE2 OUTFILE.')
        print('\t\tSee readme.md for details.')
        sys.exit(1)

    if '-kitti' in sys.argv:
        use_dataset = 'kitti'
    else:
        use_dataset = 'sintel'

    if '-pcaflow' in sys.argv:
        use_algorithm = 'pcaflow'
    else:
        use_algorithm = 'pcalayers'
    
    preset = '{}_{}'.format(use_algorithm,use_dataset)

    outfile = sys.argv[-1]
    img2 = sys.argv[-2]
    img1 = sys.argv[-3]
    
    
    PATH_PC_U = 'data/PC_U.npy'
    PATH_PC_V = 'data/PC_V.npy'
    if use_dataset == 'sintel':
        PATH_COV = 'data/COV_SINTEL.npy'
        PATH_COV_SUBLAYER = 'data/COV_SINTEL_SUBLAYER.npy'
    else:
        PATH_COV = 'data/COV_KITTI.npy'
        PATH_COV_SUBLAYER = None

    P = PCAFlow.PCAFlow(
        pc_file_u=PATH_PC_U,
        pc_file_v=PATH_PC_V,
        covfile=PATH_COV,
        covfile_sublayer=PATH_COV_SUBLAYER,
        preset=preset,
        )

    I1 = imread(img1)
    I2 = imread(img2)

    # Compute flow
    P.push_back(I1)
    P.push_back(I2)
    u,v = P.compute_flow()

    # Save output files
    I_flow = viz_flow(u,v)
    flow_write(outfile,u,v)
    imsave(outfile + '.png', I_flow)

if __name__ == '__main__':
    main()

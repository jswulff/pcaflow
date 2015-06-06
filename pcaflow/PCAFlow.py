#! /usr/bin/env python2

import numpy as np

from scipy import interpolate

import cv2

import sys,os
import time

# Local imports
import parameters as defaults

cpath = os.path.split(os.path.abspath(__file__))[0]
print(cpath)
sys.path.append(cpath)

from utils import pcautils
from utils.cprint import cprint

libviso_available=True
try:
    from features.FeatureMatcherLibviso import FeatureMatcherLibviso as FeatureMatcherLibviso
except:
    print('*** ERROR *** : Libviso features are not available, falling back to FAST.')
    print('                Please see README for instructions on how to install Libviso.')
    libviso_available=False
    FeatureMatcherLibviso = None

from features.FeatureMatcherFast import FeatureMatcherFast as FeatureMatcherFast
from features.FeatureMatcherORB import FeatureMatcherORB as FeatureMatcherORB
from features.FeatureMatcherAKAZE import FeatureMatcherAKAZE as FeatureMatcherAKAZE

from solver.RobustQuadraticSolverCython import RobustQuadraticSolverCython as RobustQuadraticSolver
from solver.EMSolver import EMSolver

import homographytools as ht

from collections import deque



class PCAFlow:
    """
    Basic PCAFlow class.

    """
    def __init__(self,pc_file_u,pc_file_v,
                 covfile,
                 covfile_sublayer=None,
                 pc_size=-1,
                 params={},
                 preset=None):
        """
        Initialize PCAFlow object.

        Parameters
        ----------
        pc_file_u, pc_file_v : string
            Files containing the principal components in horizontal and
            vertical direction, respectively.
            These files should be .npy files, in which each row is a flattened
            principal component (i.e., the total size of these principal
            component matrices is NUM_PC x (WIDTH*HEIGHT).

        cov_file : string
            File containing the covariance matrix of size NUM_PC x NUM_PC for 
            PCA-Flow.

        covfile_sublayer : string, optional
            File containing the covariance matrix for the layers (usually
            biased towards the first PCs).
            If PCA-Layers is used and this file is not given, use cov_file.

        pc_size : tuple, optional
            Size of principal components. Only required if PCs are not of size
            512x256 or 1024x436.

        params : dict, optional
            Parameters. See parameters.py for documentation of parameters.

        preset : string
            Preset with useful parameter values for different datasets.
            Can be one of
                'pcaflow_sintel'
                'pcalayers_sintel'
                'pcaflow_kitti'
                'pcalayers_kitti'

        """

        np.random.seed(1)

        self.params = defaults.get_parameters(params,preset)

        cprint('[PCAFlow] Initializing.', self.params)

        NC = int(self.params['NC'])
        self.NC = NC

        pc_u = np.load(pc_file_u)
        pc_v = np.load(pc_file_v)
        cov_matrix = np.load(covfile).astype('float32')

        if covfile_sublayer is not None:
            cov_matrix_sublayer = np.load(covfile_sublayer).astype('float32')
        else:
            cov_matrix_sublayer = None
       
        pc_w = 0
        pc_h = 0

        if pc_size==-1:
            # Try to guess principal component dimensions
            if pc_u.shape[1] == 1024*436:
                cprint('[PCAFLOW] Using PC dimensionality 1024 x 436', self.params)
                pc_w = 1024
                pc_h = 436
            elif pc_v.shape[1] == 512*256:
                cprint('[PCAFLOW] Using PC dimensionality 512 x 256', self.params)
                pc_w = 512
                pc_h = 256
            else:
                print('[PCAFLOW] *** ERROR *** ')
                print('[PCAFLOW] Could not guess dimensionality of principal components.')
                print('[PCAFLOW] Please provide as parameter.')
                sys.exit(1)


        self.PC = []

        # Smooth principal components.
        self.pc_u = self.filter_pcs(pc_u,(pc_w,pc_h)).astype('float32')
        self.pc_v = self.filter_pcs(pc_v,(pc_w,pc_h)).astype('float32')

        self.cov_matrix = cov_matrix
        
        self.pc_w = pc_w
        self.pc_h = pc_h

        self.reshape_features=True

        ###############################
        # Feature matcher
        ###############################
        if self.params['features'].lower() == 'libviso' and libviso_available:
            self.feature_matcher = FeatureMatcherLibviso(self.params)
        elif self.params['features'].lower() == 'orb':
            self.feature_matcher = FeatureMatcherORB(self.params)
        elif self.params['features'].lower() == 'fast':
            self.feature_matcher = FeatureMatcherFast(self.params)
        elif self.params['features'].lower() == 'akaze' or not libviso_available:
            self.feature_matcher = FeatureMatcherAKAZE(self.params)
        else:
            print('[PCAFLOW] *** ERROR ***')
            print('[PCAFLOW] Unknown feature type {}. Please use "libviso" or "fast".'.format(self.params['features']))
            sys.exit(1)

        if self.params['n_models'] <= 1:
            ##############################
            # Solver for PCA-Flow
            ##############################
            self.solver = RobustQuadraticSolver(self.pc_u,
                                                self.pc_v,
                                                self.cov_matrix,
                                                pc_size=(pc_w,pc_h),
                                                params=self.params)


        else:
            ############################## 
            # Solver for PCA-Layers
            ##############################  
            self.solver = EMSolver(self.pc_u, self.pc_v,
                                   self.cov_matrix,
                                   pc_size = (pc_w,pc_h),
                                   params=self.params,
                                   cov_matrix_sublayer=cov_matrix_sublayer)

        self.images = deque(maxlen=2)

        cprint('[PCAFLOW] Finished initializing.',self.params)



    def filter_pcs(self,matrix,size):
        """
        Apply Gaussian filter to principal components.
        This makes them somewhat better behaved.

        """

        matrix_out = np.zeros_like(matrix)

        #pdb.set_trace()

        for i,m in enumerate(matrix):
            m_ = m.reshape((size[1],size[0]))
            matrix_out[i,:] = cv2.GaussianBlur(m_,
                                               ksize=(0,0),
                                               sigmaX=size[0]/200.0).flatten()
        return matrix_out
        
        
    def push_back(self,I):
        """
        Push back frame.
        When processing a streaming video, this allows to pre-compute
        features only once per frame.

        Parameters
        ----------
        I : array_like
            Image, usually given as H x W x 3 color image.

        """
        cprint('[PCAFLOW] Adding image...', self.params)

        if not (I.shape[0] == self.pc_h and I.shape[1] == self.pc_w):
            self.reshape_features = True
            self.shape_I_orig = I.shape

        if self.params['image_blur'] > 0:
            I = cv2.GaussianBlur(
                    I,
                    ksize=(int(self.params['image_blur']),int(self.params['image_blur'])),
                    sigmaX=-1)

        cprint('[PCAFLOW] Adding image to feature matcher.', self.params)
        self.feature_matcher.push_back(I)
        self.images.append(I)
        cprint('[PCAFLOW] Done adding image.',self.params)

    def compute_flow(self,
                       kp1=None,kp2=None,
                       return_additional=[],
                       **kwargs
                      ):
        """
        Compute the flow.

        Parameters
        ----------
        kp1, kp2 : array_like, shape (NUM_KP,2), optional
            Matrices containing keypoints in image coordinates for
            first and second frame, respectively.
            The first column of both matrices contains the x coordinates,
            the second contains the y coordinates.
            If kp1 and kp2 are given, no additional feature matching is
            performed.
        
        return_additional: array of strings, optional.
            If set, return additional data. Possible entries are:
        
                'weights'   : Return flow coefficients
                'keypoints' : Return matched feature points
                'keypoint_labels' : Return assigned layers for keypoints
                                    (PCA-Layers only).
                'segments'  : Return segmentation map
                              (PCA-Layers only)
                'segment_flows' : For each layer, return flow.
                                  (PCA-Layers only)
        
            The additional data is returned as a dict with the same keys.
        
            Example:
                u,v,data = pcaflow.compute_flow(return_additional=['weights',])
                weights = data['weights']



        Returns
        -------
        u, v : array_like
            U and V flow fields.

        data_additional : dict, optional
            See above for details. The return formats are:

                'weights' : array_like, shape (NUM_PC,)
                'keypoints' : tuple (array_like, array_like)
                              Each array has shape (NUM_KP,2).
                'keypoint_labels' : array_like, shape (NUM_KP,)
                'segments' : array_like, shape (WIDTH,HEIGHT)
                'segment_flows' : array_like, shape (WIDTH, HEIGHT, 2, NUM_LAYERS)

        """

        # Parse return_additional.
        return_weights = False
        return_keypoints = False
        return_keypoint_labels = False
        return_segments = False
        return_segment_flows = False
        
        if 'weights' in return_additional:
            return_weights = True
        if 'keypoints' in return_additional:
            return_keypoints = True
        if 'keypoint_labels' in return_additional:
            return_keypoint_labels = True
        if 'segments' in return_additional:
            return_segments = True
        if 'segment_flows' in return_additional:
            return_segment_flows = True
            

        if kp1 is not None and kp2 is not None:
            # We got some initial features.
            kp1_ = kp1.copy()
            kp2_ = kp2.copy()

        else:
            kp1_,kp2_ = self.feature_matcher.get_features()

        if len(kp1_) == 0:
            print('[PCAFlow] Warning: No features found. Setting flow to 0.')
            u = np.zeros(self.shape_I_orig[:2])
            v = np.zeros_like(u)
            return (u,v)

        if self.params['remove_homography'] == 1:
            cprint('[PCAFlow] Removing homography...', self.params)

            kp1_h, kp2_h, H, H_inv, inliers_ = ht.remove_homography_from_points(kp1_,kp2_)

            dists_new = np.sqrt(np.sum((kp1_h - kp2_h)**2,axis=1))
            inliers = dists_new < 2
            kp1_ = kp1_h
            kp2_ = kp2_h
            #kp1[inliers,:] = kp0[inliers,:]
            I1_warped = cv2.warpPerspective(self.images[1],
                    H,
                    (self.images[1].shape[1],self.images[1].shape[0]),
                    flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                    )
        elif self.params['remove_homography'] == 2:
            cprint('[PCAFlow] Computing homography...', self.params)

            kp1_h, kp2_h, H, H_inv, inliers_ = ht.remove_homography_from_points(kp1_,kp2_)

            dists_new = np.sqrt(np.sum((kp1_h - kp2_h)**2,axis=1))
            inliers = dists_new < 2
            I1_warped = self.images[1]

        else:
            inliers = None
            I1_warped = self.images[1]
            H = None

        kp1_orig = kp1_.copy()
        kp2_orig = kp2_.copy()

        if self.reshape_features:
            h_orig,w_orig = self.shape_I_orig[:2]
            h_orig_f = float(h_orig)
            w_orig_f = float(w_orig)
            scale = [self.pc_w / w_orig_f, self.pc_h / h_orig_f]
            kp1_ *= scale
            kp2_ *= scale
            I0_ = cv2.resize(self.images[0],(self.pc_w,self.pc_h))
            I1_ = cv2.resize(I1_warped,(self.pc_w,self.pc_h))
        else:
            I0_ = self.images[0]
            I1_ = I1_warped

        cprint('[PCAFLOW] %s features detected...'%kp1_.shape[0], self.params)

        # Solve
        if self.params['n_models'] > 1:
            u_,v_,weights,data_additional_em = self.solver.solve(kp1_,kp2_,
                    I0=I0_,
                    I1=I1_,
                    inliers=inliers,
                    H=H,
                    shape_I_orig=self.shape_I_orig,
                    return_additional=return_additional,
                    **kwargs)
        else:
            if return_weights:
                u_,v_,weights = self.solver.solve(kp1_,kp2_,return_coefficients=True)
            else:
                u_,v_ = self.solver.solve(kp1_,kp2_)
            data_additional_em = {}

        if self.reshape_features:
            u = cv2.resize(u_,(w_orig,h_orig))
            v = cv2.resize(v_,(w_orig,h_orig))

            u *= w_orig_f / self.pc_w
            v *= h_orig_f / self.pc_h

        if self.params['remove_homography']==1:
            cprint('[PCAFlow] Re-applying homography...', self.params)
            u2,v2 = ht.apply_homography_to_flow(u,v,H)
            u = u2
            v = v2

        if len(return_additional) == 0:
            return u,v

        else:
            # Return more additional data
            data_additional = {}
            if return_weights:
                data_additional['weights'] = weights
            if return_keypoints:
                data_additional['keypoints'] = (kp1_orig,kp2_orig)

            # Get additional data from EMSolver
            for key,value in data_additional_em.items():
                data_additional[key] = value

            return u, v, data_additional    

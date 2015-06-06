#! /usr/bin/env python2

import numpy as np
import cv2

import clahe

from collections import deque
import time

class FeatureMatcherORB:
    def __init__(self,params):
        """
        Create new structure to extract and match ORB features,
        using the ORB descriptor.

        """
        self.params = params
        self.I_ar = []

        self.keypoints = deque(maxlen=2)
        self.descriptors = deque(maxlen=2)

        self.h = 0
        self.w = 0

        # Initialize machinery
        self.orb = cv2.ORB_create(5000)
        #self.brisk_descriptor_extractor = cv2.BRISK_create()

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)


    def push_back(self,I):
        """
        Compute features for current image, push to stack

        """
        h,w = I.shape[:2]
        self.h = h
        self.w = w

        if I.ndim > 2:
            I_ = clahe.convert_color(I)
        else:
            I_ = clahe.convert_bw(I)

        kp_in_opencv,desc = self.orb.detectAndCompute(I_,None)
        kp_in = np.array([P.pt for P in kp_in_opencv])

        if self.params['features_prune_border'] > 0:
            prune = self.params['features_prune_border']
            border_x = w * prune
            border_y = h * prune

            ind_valid = np.logical_and(np.logical_and(kp_in[:,0] >= border_x,
                                                      kp_in[:,0] <= w-border_x),
                                       np.logical_and(kp_in[:,1] >= border_y,
                                                      kp_in[:,1] <= h-border_y))
            kp_in = kp_in[ind_valid,:]

            # Clunky filtering of python list
            # (Doing this via list(array(...)[inds]) is much slower).
            kp_in_opencv = [kp_in_opencv[i] for i in xrange(len(kp_in_opencv)) if ind_valid[i]]
            desc = desc[ind_valid,:]

        
        #kp,desc = self.brisk_descriptor_extractor.compute(I_,kp_in_opencv)

        #if I_.ndim > 2:
        #    kp,desc = self.opponentbrief_descriptor_extractor.compute(I_,kp_in)
        #else:
        #    kp,desc = self.brief_descriptor_extractor.compute(I_,kp_in)
       
        self.keypoints.append(kp_in_opencv)
        self.descriptors.append(desc)


    def get_features(self):
        """
        Computes and returns feature matches.

        """
        t0 = time.time()

        max_displacement = 200

        if len(self.keypoints)<2 or len(self.keypoints[0]) < 1 or len(self.keypoints[1]) < 1:
            cx = self.w/2
            cy = self.h / 2
            return np.array([[cx,cy]]),np.array([[cx,cy]])


        matches = self.matcher.match(self.descriptors[0],self.descriptors[1])

        matches = sorted(matches, key=lambda x:x.distance)

        limit = self.params['features_fast_limit']

        if limit >= 0 and limit <= 1:
            # We use a fixed percentage of matches
            matches = matches[:int(len(matches )* limit)]
        elif limit > 1:
            # We use a given number of matches
            matches = matches[:max(len(matches),int(limit))]
        elif limit == -1:
            # Use all matches better than mean + std
            m_dists = np.array([m.distance for m in matches])
            mean = m_dists.mean()
            std_r = m_dists.std()

            invalid = m_dists>(mean+std_r)
            if np.any(invalid):
               cutoff = np.nonzero(m_dists>(mean+std_r))[0][0]
               matches = matches[:cutoff]
        elif limit == -2:
            # Use all matches better than the median
            m_dists = np.array([m.distance for m in matches])
            cutoff = np.nonzero(m_dists>np.median(m_dists))[0][0]
            matches = matches[:cutoff]

        queryIndices = [m.queryIdx for m in matches]
        trainIndices = [m.trainIdx for m in matches]

        kp1 = self.keypoints[0]
        kp2 = self.keypoints[1]

        q_x = np.array([kp1[T].pt[0] for T in queryIndices])
        q_y = np.array([kp1[T].pt[1] for T in queryIndices])
        t_x = np.array([kp2[T].pt[0] for T in trainIndices])
        t_y = np.array([kp2[T].pt[1] for T in trainIndices])

        feats_I1 = np.vstack((q_x,q_y)).T.astype('float32')
        feats_I2 = np.vstack((t_x,t_y)).T.astype('float32')

        if max_displacement > 0:
            # Exclude features that travelled further than a given limit.
            # Usually, at least in video, the displacements are still relatively
            # small, so "far" matches are very likely to be false.
            uv_norm = np.linalg.norm(feats_I2-feats_I1,axis=1)

            med_d = np.median(uv_norm)
            std_r_d = np.median(np.abs(uv_norm-med_d))
            cutoff = max(max_displacement,med_d+std_r_d)
            ind_valid = uv_norm <= max_displacement
            feats_I1 = feats_I1[ind_valid,:]
            feats_I2 = feats_I2[ind_valid,:]

            #if return_scores:
            #    matches = [k for i,k in enumerate(matches) if ind_valid[i]]
            

        t1 = time.time()
        if self.params['debug']:
            print('Feature matching took %2.6f secs'%(t1-t0))
            print('\t %s features found.'%feats_I1.shape[0])

        #if return_scores:
        #    dists = np.array([m.distance for m in matches])
        #    return feats_I1,feats_I2,dists
        #else:
        #    return feats_I1,feats_I2
        return feats_I1,feats_I2


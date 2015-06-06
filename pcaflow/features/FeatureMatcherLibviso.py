#! /usr/bin/env python2

import numpy as np
import cv2

import clahe

import time
import sys

import parameters

try:
    from extern.libvisomatcher import LibvisoMatcher
except:
    print('ERROR:: COULD NOT LOAD LIBVISOMATCHER.\n\n')
    sys.exit(1)


class FeatureMatcherLibviso:
    def __init__(self,params,image_size=None):
        """
        Create new structure to extract and match LIBVISO features.

        Optional parameter:
        image_size: To pre-allocate scales. Otherwise, the smaller
                    dimension of the image is assumed to be 512.

        """
        self.params = dict(params)
        #self.keypoints = deque(maxlen=2)
        #self.descriptors = deque(maxlen=2)

        self.h = 0
        self.w = 0

        # Initialize machinery

        #
        # Check if we want bucketing
        #
        if self.params['features_libviso_bucket_size'] > 0:
            use_buckets = True
            bucket_size = self.params['features_libviso_bucket_size']
            buckets_x = self.params['features_libviso_buckets_x']
            buckets_y = self.params['features_libviso_buckets_y']
        else:
            use_buckets = False
            bucket_size = 1
            buckets_x = 4
            buckets_y = 4

        #
        # If we want a pyramid, set parameters
        #
        if self.params['features_libviso_multiscale'] > 0:
            
            # Basis scaling of pyramid.
            sc_basis = 2.0

            if self.params['features_libviso_n_scales'] > 0:
                max_scale = int(self.params['features_libviso_n_scales']-1)
            else:
                if image_size is None:
                    w_ = 512
                else:
                    w_ = min(image_size)

                max_scale = int(np.floor(np.log(w_/32.0)/np.log(sc_basis)))

            self.scale_ranges = [float(sc_basis**i) for i in range(max_scale+1)]

        else:
            self.scale_ranges = [1,]

        #
        # Initialize matchers
        #
        self.matchers = []
        for s in self.scale_ranges:
            L = LibvisoMatcher(use_buckets=use_buckets,
                    buckets_x=buckets_x,
                    buckets_y=buckets_y,
                    bucket_size=bucket_size,
                    nms_n=self.params['features_libviso_nms_n'],
                    nms_tau=self.params['features_libviso_nms_tau'],
                    match_binsize=self.params['features_libviso_match_binsize'],
                    match_radius=self.params['features_libviso_match_radius'],
                    match_disp_tolerance=self.params['features_libviso_match_disp_tolerance'],
                    outlier_flow_tolerance=self.params['features_libviso_outlier_flow_tolerance'],
                    multi_stage=self.params['features_libviso_multi_stage'],
                    half_resolution=self.params['features_libviso_half_resolution'],
                    refinement=self.params['features_libviso_refinement'],
                    )
            self.matchers.append(L)

    def push_back(self,I):
        """
        Compute features for current image, push to stack

        """
        h,w = I.shape[:2]
        self.h = h
        self.w = w

        if self.params['features_clahe'] > 0:
            Ibw = clahe.convert_bw(I)
        else:
            if I.ndim > 2:
                Ibw = cv2.cvtColor(I,7)
            else:
                Ibw = I

        for i,s in enumerate(self.scale_ranges):
            hn = int(np.round(h/s))
            wn = int(np.round(w/s))
            
            # WORKAROUNDS for some quirks in libviso2.
            # We need to make sure that the image has an odd number of rows, and an
            # even number of columns.
            hn = hn - (hn % 2 == 0)
            wn = wn - (wn % 2 == 1)

            I_ = cv2.resize(Ibw,(wn,hn))

            self.matchers[i].pushBack(I_)


    def get_features(self):
        """
        Computes and returns feature matches.

        """
        t0 = time.time()

        feats_I1 = np.empty((0,2))
        feats_I2 = np.empty((0,2))

        for i,s in enumerate(self.scale_ranges):
            feats1_,feats2_ = self.matchers[i].getMatches()

            feats1_ *= s #[sx,sy]
            feats2_ *= s #[sx,sy]
            
            if self.params['debug']:
                print(' [Matcher] : Adding {0} features on scale {1}'.format(len(feats1_),s))

            feats_I1 = np.vstack((feats_I1,feats1_))
            feats_I2 = np.vstack((feats_I2,feats2_))
     


        if len(feats_I1) < 1 or len(feats_I2) < 1: 
            cx = self.w/2
            cy = self.h / 2
            return np.array([[cx,cy]]),np.array([[cx,cy]])

        # Optional pruning.
        if self.params['features_prune_border'] > 0:
            prune = self.params['features_prune_border']
            w = self.w
            h = self.h
            border_x = w * prune
            border_y = h * prune
            kp_in = feats_I1

            ind_valid = np.logical_and(np.logical_and(kp_in[:,0] >= border_x,
                                                      kp_in[:,0] <= w-border_x),
                                       np.logical_and(kp_in[:,1] >= border_y,
                                                      kp_in[:,1] <= h-border_y))
            feats_I1 = feats_I1[ind_valid,:]
            feats_I2 = feats_I2[ind_valid,:]

  
        feats_I1,feats_I2 = self.prune_second_stage(feats_I1.copy(),feats_I2.copy(),self.w,self.h) 

        t1 = time.time()
        if self.params['debug']:
            print('Feature matching took %2.6f secs'%(t1-t0))
            print('\t %s features found.'%feats_I1.shape[0])

        return feats_I1.astype('float32'),feats_I2.astype('float32')

    def prune_second_stage(self,kp0,kp1,w,h):
        xbins = 200
        ybins = 100

        bw = np.ceil(w / float(xbins))
        bh = np.ceil(h / float(ybins))

        kp0_ = np.zeros_like(kp0)
        kp1_ = np.zeros_like(kp1)
        bin_occupied = np.zeros((xbins*ybins))

        j=0
        for i,k in enumerate(kp0):
            b = xbins * (k[1] // bh) + k[0]//bw
            if not bin_occupied[b]:
                bin_occupied[b]=1
                kp0_[j,:] = kp0[i,:] 
                kp1_[j,:] = kp1[i,:]
                j+=1

        kp0_ = kp0_[:j,:]
        kp1_ = kp1_[:j,:]

        return kp0_,kp1_

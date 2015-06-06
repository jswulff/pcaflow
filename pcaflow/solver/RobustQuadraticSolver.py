#! /usr/bin/env python2

import numpy as np
import cv2

import ipdb
from quick_robust.quick_robust import quick_robust
import time

class RobustQuadraticSolver:
    def __init__(self,
                 flow_bases_u,  # U basis (n_bases x (width*height))
                 flow_bases_v,  # V basis (n_bases x (widght*height))
                 cov_matrix,    # Covariance matrix
                 pc_size,       # (width,height) tuple
                 params         # params
                 ):
        if params['debug']:
            print('Initializing ParameterUpdateSparse...')
        t0 = time.time()

        self.params = params

        self.flow_bases_u = flow_bases_u.astype('float32')
        self.flow_bases_v = flow_bases_v.astype('float32')

        self.flow_bases_u_t = self.flow_bases_u.T.astype('float32').copy()
        self.flow_bases_v_t = self.flow_bases_v.T.astype('float32').copy()

        self.pc_w = pc_size[0]
        self.pc_h = pc_size[1]
        
        len_bases = len(self.flow_bases_u)

        # Generate Tikhonov matrix
        self.Q = np.linalg.inv(cov_matrix + np.identity(cov_matrix.shape[0])).astype('float32')

        t1 = time.time()
        if params['debug']:
            print('Initialization took %2.6f secs'%(t1-t0))


    def update_parameters(self,params):
        for k,v in params.items():
            self.params[k] = v

    #@profile
    def get_system(self,kp0,kp1,max_bases=0):
        """
        Compute system to solve from keypoints.
        """

        # Generate differences
        u = kp1[:,0] - kp0[:,0]
        v = kp1[:,1] - kp0[:,1]

        b = np.hstack((u,v))

        #ipdb.set_trace()

        len_kp = kp0.shape[0]        
        len_bases = len(self.flow_bases_u)
        if max_bases > 0:
            len_bases = max_bases

        # Generate remap into all flow bases
        A = np.zeros((2*len_kp,2*len_bases),dtype='float32')

        kp0x_r = np.floor(kp0[:,0]).astype('int')
        kp0y_r = np.floor(kp0[:,1]).astype('int')
        indices = kp0y_r * self.pc_w + kp0x_r
        
#        A[:len_kp,:len_bases] = self.flow_bases_u_t[indices,:len_bases]
#        A[len_kp:,:len_bases] = self.flow_bases_v_t[indices,:len_bases]
        for i in range(len_bases):
            A[:len_kp,i] = self.flow_bases_u[i,indices]
            A[len_kp:,len_bases+i] = self.flow_bases_v[i,indices]
#                
        return A,b

    
   
            
        
    #@profile
    def solve(self,kp0,kp1,initial_weights=None,return_flow=True,return_coefficients=False,return_weights=False):
        """
        Compute flow fields from sparse tracks.

        Return u,v,weights

        """
        debug = self.params['debug']

        t0 = time.time()

        len_bases = len(self.flow_bases_u)


        ##################################################
        # Robust stuff
        ##################################################
        
        sigma = self.params['sigma']

        epssq_charb=1e-6
        robust_charb = lambda x: 1.0/np.sqrt(x**2 + 1e-6)                
        robust_gemanmcclure = lambda x: ((2*sigma))/((2*sigma+x**2)**2)
        robust_lorentzian = lambda x: 2.0 / (2*sigma**2 + x**2)
        robust_cauchy = lambda x : 1.0 / (1+(x/sigma)**2)

        robust_function = robust_cauchy

        ##################################################
        # Joint separate.
        ##################################################
        
        if debug:
            print('\t Estimating separate PCs jointly...')

            
        def update_weights_joint(res,robust_function):
            res_uv = np.sqrt(np.sum(res.reshape((2,-1))**2,axis=0))
            res = np.hstack((res_uv,res_uv))
            n_inliers = (res<4).sum()
            return robust_function(res).astype('float32'),n_inliers

        len_bases_max = len(self.flow_bases_u)
        red_fac = self.params['NC_reduction_factor']

        len_bases = max(1,min(len_bases_max, int(round(kp0.shape[0]/float(red_fac))))) # dividing by 4, since we retain /2 per component
        #len_bases = min(len_bases_max, kp0.shape[0]/4) # dividing by 4, since we retain /2 per component

        len_bases = min(len_bases,self.params['NC'])

        # Here we have to do some fiddling to exclude the right PCS...
        retain = np.ones_like(self.Q)

        len_component = retain.shape[0]/2
        retain[len_bases:len_component,:] = 0
        retain[:,len_bases:len_component] = 0
        retain[len_bases+len_component:,:] = 0
        retain[:,len_bases+len_component:] = 0
        
        Q = self.Q[retain==1].reshape((2*len_bases,2*len_bases)).copy()

           
        if debug:
            print('Pruning the bases to %s'%len_bases)

        Q = (Q*self.params['lambda']).astype('float32')

        weights_u = np.zeros(self.Q.shape[0]/2,dtype='float32')
        weights_v = np.zeros(self.Q.shape[0]/2,dtype='float32')

        #ipdb.set_trace()
        A,b = self.get_system(kp0,kp1,max_bases=len_bases)

        if initial_weights is not None:
            A *= np.r_[initial_weights,initial_weights][:,np.newaxis]
            b *= np.r_[initial_weights,initial_weights]
        
        weights_,robust_weights = self.solve_robust(A,b,update_weights_joint,
                                                    robust_function,Q=Q,
                                                    )
        weights_u[:len_bases] = weights_[:len_bases]
        weights_v[:len_bases] = weights_[len_bases:]
       

        ret = []
        if return_flow:

            u = self.flow_bases_u_t.dot(weights_u).reshape((self.pc_h,self.pc_w))
            v = self.flow_bases_v_t.dot(weights_v).reshape((self.pc_h,self.pc_w))

            ret.append(u)
            ret.append(v)

        if return_coefficients:
            ret.append(np.r_[weights_u,weights_v])

        if return_weights:
            ret.append(robust_weights)
        
        t1 = time.time()
        if debug:
            print('Computation took %2.6f secs'%(t1-t0))

        return ret
        

#        max_bases = self.params['NC']
#        weights_u = weights_u[:max_bases]
#        weights_v = weights_v[:max_bases]
#
#        weights = np.r_[weights_u,weights_v]
#            
#           
#        return u,v,weights

    #@profile
    def solve_robust(self,A,b,update_weights,robust_function,Q,robustiter=5):

        """
        Solve a robust system, using the given function to
        update weights.

        """
        inliers_before = 0            
        robust_weights = np.ones_like(b)
        weights_prev = np.zeros(A.shape[1],dtype='float32')

        #ipdb.set_trace()
        
        for it in range(robustiter):                
            
#            wA = robust_weights[:,np.newaxis] * A
#            A_T_dot = A.T.dot(wA)
#            A_T_dot_Q = A_T_dot + Q
#            wA_T_dot = wA.T.dot(b)
#            inv_A_T_dot_Q = np.linalg.inv(A_T_dot_Q)
#            weights = inv_A_T_dot_Q.dot(wA_T_dot)

            sw = np.sqrt(robust_weights)

            wA = sw[:,np.newaxis] * A
            wB = sw * b

            inner,rhs = self.get_parts_cy(wA.astype('float32'),Q.astype('float32'),wB.astype('float32'))
            
            weights = np.linalg.inv(inner).dot(rhs)

            #weights = np.linalg.inv(A.T.dot(wA)+Q).dot(wA.T.dot(b))
            res = A.dot(weights) - b

            if not robust_function == None:
                robust_weights,n_inliers = update_weights(res,robust_function)
            else:
                robust_weights = update_weights(res)
                n_inliers = (res<4).sum()

        return weights,robust_weights

    def get_parts_np(self,A,Q,b):
        inner = A.T.dot(A) + Q
        rhs = A.T.dot(b)
        return inner,rhs

    def get_parts_cy(self,A,Q,b):
        inner,rhs = quick_robust(A,Q.copy(),b)
        return inner,rhs

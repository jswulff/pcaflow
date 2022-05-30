#! /usr/bin/env python2

import numpy as np
import cv2
import sys,os
import time

import parameters as dp
import homographytools as ht
from utils import pcautils
from utils.cprint import cprint

# For debugging
have_plt=False
try:
    from matplotlib import pyplot as plt
    have_plt=True
except:
    have_plt=False
    plt = None

from sklearn import mixture
from sklearn.cluster import KMeans


# Local
#from RobustQuadraticSolver import RobustQuadraticSolver
from RobustQuadraticSolverCython import RobustQuadraticSolverCython as RobustQuadraticSolver

from extern import pygco


def pullback_opencv(u,v,I):
    """
    Simple warping method, using OpenCV for speed.

    """
    x,y = np.meshgrid(range(u.shape[1]),range(u.shape[0]))
    xn = (x + u).astype('float32')
    yn = (y + v).astype('float32')
    I_warped = cv2.remap(I,
            xn,yn,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)
    I_valid = cv2.remap(np.ones(I.shape[:2],dtype='uint8'),
            xn,yn,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT) == 1
    return I_warped,I_valid

    
class EMSolver:
    def __init__(self,
                 flow_bases_u,  # U basis (n_bases x (width*height))
                 flow_bases_v,  # V basis (n_bases x (widght*height))
                 cov_matrix,    # Covariance matrix
                 pc_size,       # (width,height) tuple
                 params,        # params
                 cov_matrix_sublayer=None, # Optional different covariance matrix for sublayer solver
                 ):
        cprint('[EMSolver] Initializing ...', params)
        t0 = time.time()

        self.params = dict(params)

        self.flow_bases_u = flow_bases_u.astype('float32')
        self.flow_bases_v = flow_bases_v.astype('float32')

        self.pc_w = pc_size[0]
        self.pc_h = pc_size[1]

        self.cov_matrix = cov_matrix.astype('float32').copy()
        
        len_bases = len(self.flow_bases_u)

        n_components_max = self.flow_bases_u.shape[0]

        self.n_models = params['n_models']

        # Define additional solution
        self.n_models += 1

        # Section to add QuadraticSolver as additional solution
        params_inner = dict(self.params)            

        cprint('[EMSolver] Initializing additional RobustQuadraticSolver...', self.params)
        self.sub_solver_additional = RobustQuadraticSolver(
                flow_bases_u,
                flow_bases_v,
                cov_matrix,
                pc_size,
                params_inner,
                n_iters=10)
        self.use_additional = 1
                        
        self.models = np.zeros((params['n_models']+self.use_additional,2*n_components_max),dtype='float32')
        self.model_medians = np.ones((params['n_models']+self.use_additional,2)) * -1

        # If no separate covariance matrix for sublayer is provided, use the full one.
        if cov_matrix_sublayer is not None:
            self.cov_sublayer = cov_matrix_sublayer.copy()
        else:
            self.cov_sublayer = cov_matrix.copy()

        params_sublayer = dict(dp.get_sublayer_parameters(self.params))

        self.sub_solver = RobustQuadraticSolver(flow_bases_u,
                flow_bases_v,
                self.cov_sublayer,
                pc_size,
                params_sublayer,
                n_iters=10)

        self.debug_path = './output'

        t1 = time.time()

        cprint('[EMSolver] Done. Initialization took %2.6f secs'%(t1-t0), self.params)


    def update_parameters(self,params):
        for k,v in params.items():
            self.params[k] = v

            
    def get_system(self,kp0,kp1):
        """
        Compute system to solve from keypoints.

        In this context, we use it to be able to quickly estimate the per-point
        errors for different estimated models
        """

        # Generate differences
        u = kp1[:,0] - kp0[:,0]
        v = kp1[:,1] - kp0[:,1]

        b = np.hstack((u,v))

        #ipdb.set_trace()

        len_kp = kp0.shape[0]        
        len_bases = self.flow_bases_u.shape[0]

        # Generate remap into all flow bases
        A = np.zeros((2*len_kp,2*len_bases),dtype='float32')

        kp0x_r = np.floor(kp0[:,0]).astype('int')
        kp0y_r = np.floor(kp0[:,1]).astype('int')
        indices = kp0y_r * self.pc_w + kp0x_r
        
        for i in range(len_bases):
            A[:len_kp,i] = self.flow_bases_u[i,indices]
            A[len_kp:,len_bases+i] = self.flow_bases_v[i,indices]
                
        return A,b

    #@profile
    def solve(self,kp0,kp1,soft=False,I0=None,I1=None,inliers=None,H=None,
              shape_I_orig=None,
              return_additional=[],
              **kwargs):
        """
        Solve using EM.

        This is the main entry function.
        """

        # Parse return_additional
        return_keypoint_labels=False
        return_segments=False
        return_segment_flows=False
        if 'keypoint_labels' in return_additional:
            return_keypoint_labels=True
        if 'segments' in return_additional:
            return_segments = True
        if 'segment_flows' in return_additional:
            return_segment_flows = True
        
        kp0_ = kp0.copy()
        kp1_ = kp1.copy()

        # Compute system in order to evaluate models.
        A,b = self.get_system(kp0_,kp1_)
        n_points = kp0_.shape[0]
        n_models = self.params['n_models']
        n_components_max = self.flow_bases_u.shape[0]

        # ownership indicates which keypoint belongs to which model.
        ownership = np.zeros((n_models,n_points),dtype='bool')
        ownership_previous = ownership.copy()

        # Distance of each keypoint to the model
        dists = np.zeros((n_models,n_points),dtype='float32')
        weights_all = np.zeros_like(dists)

        
        if kwargs.has_key('debug_path'):
            self.debug_path = kwargs['debug_path']
        
        if self.use_additional:
            self.models = np.zeros((n_models+1,2*n_components_max),dtype='float32')

            # Defining a "median" for all points does not make sense (this would
            # cause center pixels to be more likely to belong to the PCA-Flow
            # solution
            self.model_medians = np.ones((n_models,2)) * -1

            # For the additional (=PCAFlow) model, solve using all keypoints.
            model_additional,weights_features = self.sub_solver_additional.solve(
                    kp0,
                    kp1,
                    return_flow=False,
                    return_coefficients=True,
                    return_weights=True,
                    )
            
            self.models[-1,:] = model_additional

            
        else:
            self.models = np.zeros((n_models,2*n_components_max),dtype='float32')
            self.model_medians = np.ones((n_models,2)) * -1


        

        ############################## 
        # Initialize
        ##############################
        IDs = np.arange(n_points)
        block_width = self.pc_w / n_models

        uv = kp1_-kp0_
        data_clustering = np.c_[kp0_,uv].astype('float32')
        #data_clustering = kp0_.astype('float32')
        data_clustering -= data_clustering.mean(axis=0)
        data_clustering /= data_clustering.std(axis=0)

        # Weigh down the location features
        data_clustering[:,:2] *= self.params['em_init_loc_weight']
        
        # Use KMeans from scikit-image, since OpenCV's sklearn cannot be
        # used with a given random seed.
        L = KMeans(n_clusters=n_models,
                max_iter=100,
                tol=0.1,
                precompute_distances=False,
                random_state=12345).fit_predict(data_clustering)

        # For each cluster, extract the points belonging to this cluster,
        # and recompute the model.
        for m in range(n_models):
            weights = self.sub_solver.solve(
                    kp0_[L==m,:],
                    kp1_[L==m,:],
                    return_flow=False,
                    return_coefficients=True,
                    )[0]
            self.models[m,:] = weights
            ownership[m,L==m] = True

            # Robustly compute median of model locations
            kp0_cur = kp0_[ownership[m,:],:]
            self.model_medians[m] = np.median(kp0_cur,axis=0)
            
        
        USE_MEDIAN = True
        MED_FACTOR = self.params['model_factor_dist_to_median']

        ##############################
        # Iterate to get ownership
        ##############################
        for iter in range(20):
            
            #
            # M-Step: Determine distances and ownerships
            #
            for m in range(n_models):
                # Compute distance of all points to current model
                err = (A.dot(self.models[m,:])-b)**2
                dists[m,:] = np.sqrt(err[:n_points]+err[n_points:])

                # Add median to distance
                if USE_MEDIAN and self.model_medians[m,0] > -1:
                    dists_median = np.sqrt(np.sum((kp0_ - self.model_medians[m])**2,axis=1))
                    dists[m,:] += dists_median * MED_FACTOR

            # Set correct ownerships (=binary mask)
            mn = np.argmin(dists,axis=0)
            ownership[:] = False
            ownership[mn,IDs] = True

            weights_all = np.exp(-dists)
            weights_all /= np.maximum(1e-9,weights_all.sum(axis=0))

            # Check how many entries changed. If no change, exit.
            n_change = np.sum(np.logical_xor(ownership,ownership_previous))/2
            cprint('Iter {0}. {1} entries changed...\n'.format(iter,n_change),self.params)
            if n_change == 0:
                break


            # Remove models with < 10 points
            small_models = ownership.sum(axis=1) < 10
            if np.any(small_models):
                # Prune the empty models
                m_remove = np.nonzero(small_models)[0]
                
                print('[EMSolver] Removing models {} because they became too small.'.format(m_remove))

                weights_all = np.delete(weights_all,m_remove,axis=0)
                ownership = np.delete(ownership,m_remove,axis=0)
                ownership_previous = np.delete(ownership_previous,m_remove,axis=0)
                self.models = np.delete(self.models,m_remove,axis=0)
                self.model_medians = np.delete(self.model_medians,m_remove,axis=0)
                dists = np.delete(dists,m_remove,axis=0)
                n_models -= len(m_remove)
                continue

            # 
            # E-Step: Re-compute models
            # 
            for m in range(n_models):
                kp0_cur = kp0_[ownership[m,:],:]
                kp1_cur = kp1_[ownership[m,:],:]

                self.models[m,:] = self.sub_solver.solve(kp0_cur,kp1_cur,
                                             return_flow=False,
                                             return_coefficients=True)[0]
                
                self.model_medians[m] = np.median(kp0_cur,axis=0)

            ownership_previous = ownership.copy()

            
        # Determine ownerships one last time
        for m in range(n_models):
            err = (A.dot(self.models[m,:])-b)**2
            dists[m,:] = np.sqrt(err[:n_points]+err[n_points:])

            if USE_MEDIAN and self.model_medians[m,0] > -1:
                dists_median = np.sqrt(np.sum((kp0_ - self.model_medians[m])**2,axis=1))
                dists[m,:] += dists_median * MED_FACTOR

        mn = np.argmin(dists,axis=0)
        ownership[:] = False
        ownership[mn,IDs] = True

        weights_all = np.exp(-dists)
        weights_all /= np.maximum(1e-9,weights_all.sum(axis=0))


        if I0 is None or I1 is None:
            print('[EMSolver] :: ERROR. No images given.')
            u = None
            v = None
        else:
            if inliers is None:
                u,v,segments,flow_u_all,flow_v_all = self.get_flow_GC(kp0_,kp1_,weights_all,I0,I1)
            else:
                u,v,segments,flow_u_all,flow_v_all = self.get_flow_GC(kp0_,kp1_,weights_all,
                        I0,I1,
                        inliers,H,shape_I_orig)

        if len(return_additional) == 0:
            return u,v,self.models[0],{}
            
        else:
            data_additional_em = {}
            if return_keypoint_labels:
                data_additional_em['keypoint_labels'] = mn
            if return_segments:
                data_additional_em['segments'] = segments
            if return_segment_flows:
                segment_flows = np.zeros((self.pc_h,
                                          self.pc_w,
                                          2,
                                          len(flow_u_all)))

                for i in range(len(flow_u_all)):
                    segment_flows[:,:,0,i] = flow_u_all[i].reshape((self.pc_h,self.pc_w))
                    segment_flows[:,:,1,i] = flow_v_all[i].reshape((self.pc_h,self.pc_w))

                data_additional_em['segment_flows'] = segment_flows

            return u,v,self.models[0],data_additional_em
                

        
    def get_flow_GC(self,kp0,kp1,weights,I0,I1,inliers=None,H=None,shape_I_orig=None):
        """
        Given models, densify using graph cut (i.e., solve labeling problem).

        """


        # Determine ownership of points
        use_zero_layer = False
        
        # At this point, n_models also contains the PCA-Flow model.
        n_models = self.models.shape[0]
        
        point_models = np.argmax(weights,axis=0)

        # If inliers is not zero, we want to compute a "zero" layer using the
        # homography
        if inliers is not None:
            n_models += 1
            use_zero_layer = True

        use_homography = self.params['remove_homography']

        n_coeffs = self.flow_bases_u.shape[0]
        n_pixels = self.flow_bases_u.shape[1]


        # Define general cost structures
        log_unaries = np.zeros((self.pc_h,self.pc_w,n_models),dtype='int32')


        # Warping takes the images into account.
        # Thus, we need to rescale them to the size of the principal components.
        I_ndim = I0.ndim
        if shape_I_orig is None:
            Ih,Iw = I0.shape[:2]
        else:
            Ih,Iw = shape_I_orig[:2]
            
        if I_ndim > 2:
            I0_ = cv2.resize(cv2.cvtColor(I0,45),(self.pc_w,self.pc_h))
            I1_ = cv2.resize(cv2.cvtColor(I1,45),(self.pc_w,self.pc_h))
        else:
            I0_ = cv2.resize(I0,(self.pc_w,self.pc_h))
            I1_ = cv2.resize(I1,(self.pc_w,self.pc_h))
            
        if I_ndim > 2:
            I0_bw = I0_[:,:,0]
            I1_bw = I1_[:,:,0]
        else:
            I0_bw = I0_
            I1_bw = I1_

        x,y = np.meshgrid(range(self.pc_w),range(self.pc_h))


        # Build basis flow models
        flow_u_all = np.zeros((n_models,n_pixels))
        flow_v_all = np.zeros((n_models,n_pixels))

        # Save indices for PCA-Flow and homography models.
        # If unset, set to invalid indices to catch errors
        pcaflow_model = n_models+1
        homography_model = n_models+1
        if self.params['remove_homography']:
            homography_model = n_models-1
            pcaflow_model = n_models - 2
        else:
            pcaflow_model = n_models - 1


        # For each model / layer, generate flow fields from coefficients.
        for m in range(n_models):
            if m == homography_model:
                # If we are on the homography layer, generate from from H.
                # (We generate the flow from H before downscaling it to the
                # size of the PCs.)
                ud = np.zeros((Ih,Iw),dtype='float32')
                vd = np.zeros((Ih,Iw),dtype='float32')
                if H is None:
                    H = np.eye(3)
                ud,vd = ht.apply_homography_to_flow(ud,vd,H)
                u,v = pcautils.scale_u_v(ud,vd,(self.pc_w,self.pc_h))
                flow_u_all[m] = u.flatten()
                flow_v_all[m] = v.flatten()
            else:
                # Simply create flow by weighting.
                flow_u_all[m] = self.models[m,:n_coeffs].dot(self.flow_bases_u)
                flow_v_all[m] = self.models[m,n_coeffs:].dot(self.flow_bases_v)



        # Step 1: Color models

        if self.params['model_gamma_c'] > 0:
            log_color = self._compute_unaries_color(kp0,
                                                    kp1,
                                                    I0_,
                                                    n_models,
                                                    pcaflow_model,
                                                    homography_model,
                                                    inliers,
                                                    point_models)
            log_unaries += log_color

            

        if self.params['model_gamma_warp'] > 0:
           log_warp = self._compute_unaries_warp(I0_bw.astype('float32'),
                                                 I1_bw.astype('float32'),
                                                 n_models,
                                                 use_homography,
                                                 homography_model,
                                                 flow_u_all,
                                                 flow_v_all)
           log_unaries += log_warp
           


        if self.params['model_gamma_l'] > 0:
            log_dist = self._compute_unaries_location(kp0,
                                                      n_models,
                                                      homography_model,
                                                      pcaflow_model,
                                                      point_models,
                                                      inliers)

            log_unaries += log_dist

        cprint('\n',self.params)


        #
        # Compute pairwise terms
        #

        # This is a simple 0/1 error. All the weighting is done through the
        # weight variables w_x, w_y.
        cprint('[GC] Computing edgeweights...',self.params)
        
        gamma = self.params['model_gamma']        

        log_pairwise = (-np.eye(n_models)).astype('int32')

        # Compute weights according to GrabCut
        gy,gx = np.gradient(I0_bw.astype('float32'))
        beta = 1.0 / ((gy**2).mean() + (gx**2).mean())
        w_y_gc = np.exp(- beta * gy**2)
        w_x_gc = np.exp(- beta * gx**2)
        w_x = (w_x_gc * 100 * gamma).astype('int32')
        w_y = (w_y_gc * 100 * gamma).astype('int32')

        
        cprint('done.\n',self.params)
        cprint('[GC] Solving...',self.params)
        try:
            res_ = pygco.cut_simple_vh(log_unaries,log_pairwise,w_y,w_x)
        except:
            cprint('[GC] *** Alpha expansion failed. Using alpha-beta swap.', self.params)
            res_ = pygco.cut_simple_vh(log_unaries,log_pairwise,w_y,w_x,algorithm='swap')

        res = cv2.medianBlur(res_.astype('uint8'),ksize=3).astype('int32')
        cprint('done.\n',self.params)

        if self.params['debug']>1:
            self.output_debug2(kp0,point_models,res,flow_u_all,flow_v_all)

        u_all = flow_u_all[res.ravel(),np.arange(n_pixels)].reshape((self.pc_h,self.pc_w))
        v_all = flow_v_all[res.ravel(),np.arange(n_pixels)].reshape((self.pc_h,self.pc_w))

        return u_all,v_all,res,flow_u_all,flow_v_all



    def _compute_unaries_color(self,
                               kp0,
                               kp1,
                               I0_,
                               n_models,
                               pcaflow_model,
                               homography_model,
                               inliers,
                               point_models):
        """
        Compute unaries based on the color distributions of assigned features.

        """
        # Build color models
        cprint('[GC] Computing color models',self.params)
        kp0_ = np.floor(kp0).astype('int')

        I_ndim = I0_.ndim
       
        # Colors at points
        point_surround = 1
        if I_ndim > 2:
            colors_points_ = I0_[kp0_[:,1],kp0_[:,0],:].reshape((-1,3))
        else:
            colors_points_ = I0_[kp0_[:,1],kp0_[:,0]].flatten()

        # Colors of whole image
        if I_ndim > 2:
            color_all_ = I0_.reshape((-1,3))
        else:
            color_all_ = I0_.flatten()

        colors_points = colors_points_.astype('float32')
        color_all = color_all_.astype('float32')

        # Normalize to mean / std of matched points.
        color_all -= colors_points.mean(axis=0)
        color_all /= colors_points.std(axis=0)

        colors_points -= colors_points.mean(axis=0)
        colors_points /= colors_points.std(axis=0)

        scores_colors = np.zeros((self.pc_h,self.pc_w,n_models))

        for m in range(n_models):
            # Compute indices into features at current model
            if m == homography_model:
                # For homography layer, use only inliers
                ind = np.tile(inliers==1,point_surround)
            elif m == pcaflow_model:
                # For PCAFlow layer, use all points
                ind = np.ones(colors_points.shape[0])==1
            else:
                # Otherwise, use current ownerships
                ind = np.tile(point_models==m,point_surround)

            # Extract colors of selected features
            if I_ndim > 2:
                P = colors_points[ind,:]
            else:
                P = colors_points[ind]

            cprint('[GC] Model {0}: Num points: {1}'.format(m,P.shape[0]),self.params)
            
            if P.shape[0] > 1:
                if P.shape[0] < 10:
                    nc = 1
                else:
                    # Currently, this is always one. Mixtures were of no
                    # advantage.
                    nc = self.params['model_color_n_mixtures']

                # Fit Gaussian to selected color points, and compute score for
                # all pixels.
                G = mixture.GaussianMixture(n_components=nc,covariance_type='full').fit(P)
                #score = G.score(color_all)
                score = G.score_samples(color_all)
                
            else:
                # Simple fallback.
                score = np.ones(color_all.shape[0]) * -100000

            S = score.reshape((self.pc_h,self.pc_w))
            S = cv2.GaussianBlur(S,ksize=(5,5),sigmaX=-1)
            scores_colors[:,:,m] = S 

        log_colors = - (self.params['model_gamma_c'] * 100 * (scores_colors - scores_colors.max(axis=2)[:,:,np.newaxis])).astype('int32')

        cprint('done\n',self.params)
        
        return log_colors


        
    def _compute_unaries_warp(self,
                              I0_bw,
                              I1_bw,
                              n_models,
                              use_homography,
                              homography_model,
                              flow_u_all,
                              flow_v_all
                              ):
        """
        Compute warping based unaries, i.e. the violation of brightness
        constancy given a particular flow.

        """

        #import ipdb;from matplotlib import pyplot as plt; ipdb.set_trace()
        cprint('[GC] Computing warp models',self.params)

        l_warp = np.zeros((self.pc_h,self.pc_w, n_models))

        dI0dy,dI0dx = np.gradient(I0_bw)
        dI1dy,dI1dx = np.gradient(I1_bw)

        dx_weight = 1.0

        I1_stacked = np.dstack((I1_bw,dx_weight*dI1dx,dx_weight*dI1dy))
        I0_stacked = np.dstack((I0_bw,dx_weight*dI0dx,dx_weight*dI0dy))

        I_valid_homography = np.ones_like(I1_bw)

        for m in range(n_models):
            if use_homography==1 and m == homography_model:
                # I1 has the homography already removed. Nothing to do here.
                I1_warped = I1_stacked
                I_valid = np.ones(I1_warped.shape[:2],dtype='uint8')
                
            else:
                # Warp back by flow for current model
                u = flow_u_all[m].reshape((self.pc_h,self.pc_w))
                v = flow_v_all[m].reshape((self.pc_h,self.pc_w))

                I1_warped,I_valid = pullback_opencv(u,v,I1_stacked)

                if m == homography_model:
                    I_valid_homography = I_valid



            df = np.abs(I0_stacked - I1_warped.astype('float32')).mean(axis=2)

            D = cv2.GaussianBlur(df,ksize=(5,5),sigmaX=-1)
            
            if self.params['model_sigma_w'] > 0:
                D = 255.0* (1.0 - np.exp(-(D/self.params['model_sigma_w'])**2))
                #pass

            l_warp[:,:,m] = D

            if use_homography == 2:
                cprint('[CG] Homography mean: {0}'.format(I_valid_homography.astype('float32').mean()), self.params)
                l_warp[I_valid_homography<1] = 0


        log_warp = (100 * self.params['model_gamma_warp'] * (l_warp - l_warp.min(axis=2)[:,:,np.newaxis])).astype('int32')
        return log_warp


    def _compute_unaries_location(self,
                                  kp0,
                                  n_models,
                                  homography_model,
                                  pcaflow_model,
                                  point_models,
                                  inliers
                                  ):
        """
        Compute unaries based on distance to feature points

        """

            
        l_feat_dist = np.zeros((self.pc_h,self.pc_w,n_models))
        
        tmp = np.ones((self.pc_h,self.pc_w),dtype='float32')
        for m in range(n_models):
            if m == homography_model:
                P0 = np.round(kp0[inliers==1,:]).astype('int32')
            elif m == pcaflow_model:
                P0 = np.round(kp0).astype('int32')
            else:
                P0 = np.round(kp0[point_models==m,:]).astype('int32')

            tmp[:] = 0
            tmp[P0[:,1],P0[:,0]] = 255.0

            tmp = cv2.GaussianBlur(tmp,ksize=(0,0),sigmaX=15)

            if m == pcaflow_model:
                tmp[:] = 0.5

            l_feat_dist[:,:,m] = tmp

        log_dist = -(100 * self.params['model_gamma_l'] * (l_feat_dist - l_feat_dist.max(axis=2)[:,:,np.newaxis])).astype('int32')

        return log_dist
        

    ##################################
    #
    # From here onwards, undocumented debugging code.
    #
    ##################################



    def get_flow_nearest(self,kp0,weights):
        
        from scipy.ndimage import interpolate
        
        # Determine flow fields
        n_models = weights.shape[0] #self.params['n_models']
        n_coeffs = self.flow_bases_u.shape[0]
        n_pixels = self.flow_bases_u.shape[1]

        flow_u_all = np.zeros((n_models,n_pixels))
        flow_v_all = np.zeros((n_models,n_pixels))
        for m in range(n_models):
            flow_u_all[m] = self.models[m,:n_coeffs].dot(self.flow_bases_u)
            flow_v_all[m] = self.models[m,n_coeffs:].dot(self.flow_bases_v)


        # Each keypoint has associated weights to each of the models.
        # Now, we interpolate these weights.
        x,y = np.meshgrid(range(self.pc_w),range(self.pc_h))
        weights_all = np.zeros((n_models,n_pixels))
        for m in range(n_models):
            weights_nn = interpolate.griddata(kp0,weights[m],(x,y),method='nearest').ravel()
            #weights_lin = interpolate.griddata(kp0,weights[m],(x,y),method='linear').ravel()
            weights_all[m] = weights_nn #* np.isnan(weights_lin) + weights_lin * (np.isnan(weights_lin)==0)

        flow_u = (flow_u_all * weights_all).sum(axis=0).reshape((self.pc_h,self.pc_w))
        flow_v = (flow_v_all * weights_all).sum(axis=0).reshape((self.pc_h,self.pc_w))

        return flow_u,flow_v


    def debug_GC(self,unaries_warp,unaries_colors,unaries_dist,tot,res,w_y,w_x,flow_u_all=None,flow_v_all=None):

        try:
            from mylib import viz
        except:
            return

        n_models = unaries_warp.shape[2]
        n_coeffs = self.flow_bases_u.shape[0]
        n_pixels = self.flow_bases_u.shape[1]

        I_warp = np.vstack([unaries_warp[:,:,m] for m in range(n_models)])
        I_col = np.vstack([unaries_colors[:,:,m] for m in range(n_models)])
        I_dist = np.vstack([unaries_dist[:,:,m] for m in range(n_models)])
        
        I_tot = np.vstack([tot[:,:,m] for m in range(n_models)])
        #I_tot = I_warp + I_col + I_dist

        if have_plt:
            plt.figure(figsize=(10,10))

            plt.subplot(1,4,1)
            plt.imshow(I_warp,cmap='hot')
            plt.title('Warp unaries')
            plt.subplot(1,4,2)
            plt.imshow(I_col,cmap='hot')
            plt.title('Color model unaries')
            plt.subplot(1,4,3)
            plt.imshow(I_dist,cmap='hot')
            plt.title('Distance unaries')
            plt.subplot(1,4,4)
            plt.imshow(I_tot,cmap='hot')
            plt.title('Combined unaries')
            plt.savefig('unaries.png',dpi=200,bbox_inches='tight')
            plt.close()

            if res is not None:
                plt.figure(figsize=(10,10))
                plt.imshow(res,cmap='hot')
                plt.title('Model results')
                plt.savefig('result_models.png',dpi=200,bbox_inches='tight')
                plt.close()

            plt.figure(figsize=(10,10))
            plt.subplot(2,1,1)
            plt.imshow(w_y,cmap='hot')
            plt.colorbar()
            plt.title('Vertical regularization weights')

            plt.subplot(2,1,2)
            plt.imshow(w_x,cmap='hot')
            plt.colorbar()
            plt.title('Horizontal reg weights')

            plt.savefig('reg_weights.png',dpi=200,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10,10))
            plt.subplot(411)
            plt.imshow(np.argmin(unaries_warp,axis=2),cmap='hot')
            plt.title('min of warp unaries')

            plt.subplot(412)
            plt.imshow(np.argmin(unaries_colors,axis=2),cmap='hot')
            plt.title('min of color unaries')

            plt.subplot(413)
            plt.imshow(np.argmin(unaries_dist,axis=2),cmap='hot')
            plt.title('min of dist unaries')

            plt.subplot(414)
            plt.imshow(np.argmin(unaries_warp + unaries_colors + unaries_dist,axis=2),cmap='hot')
            plt.title('min of combined unaries')
            plt.savefig('unaries_min.png',dpi=200,bbox_inches='tight')
            plt.close()

            if flow_u_all is not None:
                plt.figure(figsize=(10,10))
                nx = 3
                ny = int(np.ceil(float(n_models)/3.0))
                for n in range(n_models):
                    u = flow_u_all[n].reshape((256,512))
                    v = flow_v_all[n].reshape((256,512))
                    Iv = viz.viz_flow(u,v)
                    
                    plt.subplot(ny,nx,n+1)
                    plt.imshow(Iv)
                    plt.title('Model {}'.format(n))
                plt.savefig('./models_all.png',dpi=200,bbox_inches='tight')
                plt.close()



    def output_debug(self,kp0,ownership,iterstring,weights=None):
        try:
            from mylib import viz
        except:
            return

        n_models = self.models.shape[0] #self.params['n_models']
        n_coeffs = self.flow_bases_u.shape[0]

        sz_ownership = ownership.shape[0]
        if sz_ownership < n_models:
            z = np.zeros((n_models-sz_ownership,ownership.shape[1]))==1
            ownership = np.vstack((ownership,z)).copy()

        flow_u_all = np.zeros((n_models,self.flow_bases_u.shape[1]))
        flow_v_all = np.zeros((n_models,self.flow_bases_u.shape[1]))
        for m in range(n_models):
            flow_u_all[m] = self.models[m,:n_coeffs].dot(self.flow_bases_u)
            flow_v_all[m] = self.models[m,n_coeffs:].dot(self.flow_bases_v)

        if weights is None:
            u_combined,v_combined = self.get_flow_nearest(kp0,ownership)
        else:
            u_combined,v_combined = self.get_flow_nearest(kp0,weights)

        I_combined = viz.viz_flow(u_combined,v_combined)
        I_individuals = []
        for m in range(n_models):
            I_individuals.append(viz.viz_flow(
                flow_u_all[m].reshape((self.pc_h,self.pc_w)),
                flow_v_all[m].reshape((self.pc_h,self.pc_w))))

        n_x = int(np.ceil((n_models+2)/3.0))
        n_y = 3
        colors = np.argmax(ownership,axis=0)

        if have_plt:

            plt.figure(figsize=(n_x*8,n_y*4))
            plt.subplot(n_y,n_x,1)
            plt.scatter(kp0[:,0],kp0[:,1],linewidths=0,s=10,c=colors,vmin=-1,vmax=n_models,cmap='cubehelix')
            plt.xlim([0,self.pc_w])
            plt.ylim([self.pc_h,0])
            plt.title('Point ownerships')
            plt.subplot(n_y,n_x,2)
            plt.imshow(I_combined)
            plt.title('Combined flow')

            for m in range(n_models):
                plt.subplot(n_y,n_x,3+m)
                plt.imshow(I_individuals[m])
                inds = ownership[m]
                plt.scatter(kp0[inds,0],kp0[inds,1],linewidths=0,s=10,c='black')
                plt.xlim([0,self.pc_w])
                plt.ylim([self.pc_h,0])
                plt.title('Flow model {0}'.format(m))

            plt.savefig('./output_{0}.png'.format(iterstring),dpi=100,bbox_inches='tight')
            plt.close()



    def output_debug_scatter(self,A,b,ownership,iterstring):
        """
        Compute scatter plot of errors
        """
        n_models = self.params['n_models']
        n_coeffs = self.flow_bases_u.shape[0]
        n_points = A.shape[0]/2

        A_u = A[:n_points,:]
        A_v = A[n_points:,:]
        b_u = b[:n_points]
        b_v = b[n_points:]

        err_u = []
        err_v = []

        for m in range(n_models):
            inds = ownership[m]
            err_u_ = A_u[inds,:].dot(self.models[m])-b_u[inds]
            err_v_ = A_v[inds,:].dot(self.models[m])-b_v[inds]

            err_u.append(err_u_)
            err_v.append(err_v_)

        n_x = int(np.ceil(n_models/3.0))
        n_y = 3

        if have_plt:
            plt.figure(figsize=(n_x*5,n_y*5))

            for m in range(n_models):
                plt.subplot(n_y,n_x,m+1)
                colors = [m]*len(err_u[m])
                plt.scatter(err_u[m],err_v[m],c=colors,vmin=-1,vmax=n_models,linewidths=0,s=10,cmap='cubehelix')
                plt.xlim([-5,5])
                plt.ylim([-5,5])
                plt.title('Model {0}'.format(m))

            plt.savefig('./output_scatter_{0}.png'.format(iterstring),dpi=100,bbox_inches='tight')
            plt.close()




#    def debug_GC2(self,res):
#        #n_models = unaries_warp.shape[2]
#        #n_coeffs = self.flow_bases_u.shape[0]
#        #n_pixels = self.flow_bases_u.shape[1]
#
#        #I_warp = np.vstack([unaries_warp[:,:,m] for m in range(n_models)])
#        #I_col = np.vstack([unaries_colors[:,:,m] for m in range(n_models)])
#        #I_dist = np.vstack([unaries_dist[:,:,m] for m in range(n_models)])
#        
#        #I_tot = np.vstack([tot[:,:,m] for m in range(n_models)])
#        #I_tot = I_warp + I_col + I_dist
#
#        path = self.debug_path
#        if have_plt:
#            plt.figure(figsize=(10,10))
#            plt.imshow(res,cmap='cubehelix')
#            plt.title('Model results')
#            plt.xticks([],[])
#            plt.yticks([],[])
#            plt.savefig(os.path.join(path,'segments.png'),dpi=200,bbox_inches='tight',pad_inches=0)
#            plt.close()
#
#


    def output_debug2(self,kp0,point_models,res,flow_u_all,flow_v_all):
        try:
            from mylib import misc
        except:
            return

        n_models = flow_u_all.shape[0] #self.params['n_models']
        n_coeffs = self.flow_bases_u.shape[0]

#        sz_ownership = ownership.shape[0]
#        if sz_ownership < n_models:
#            z = np.zeros((n_models-sz_ownership,ownership.shape[1]))==1
#            ownership = np.vstack((ownership,z)).copy()
#
        I_individuals = []
        for m in range(n_models):
            I_individuals.append(viz.viz_flow(
                flow_u_all[m].reshape((self.pc_h,self.pc_w)),
                flow_v_all[m].reshape((self.pc_h,self.pc_w))))

        #colors = np.argmax(ownership,axis=0)
        colors = flow_u_all.shape[0]

        path = self.debug_path

        if have_plt:
            # Point ownershipts
            plt.figure(figsize=(10,10))
            plt.scatter(kp0[:,0],kp0[:,1],linewidths=0,s=40,c=point_models,vmin=0,vmax=n_models,cmap='cubehelix',alpha=0.5)

            #plt.axis('equal')
            plt.xlim([0,self.pc_w])
            plt.ylim([self.pc_h,0])
            plt.xticks([],[])
            plt.yticks([],[])
            ax = plt.gca()
            ax.set_aspect('equal')
            #plt.title('Point ownerships')
            plt.savefig(os.path.join(path,'ownerships.png'),dpi=200,bbox_inches='tight',pad_inches=0)
            plt.close() 

            for m in range(n_models):
                plt.figure(figsize=(10,10),frameon=False)
                plt.imshow(I_individuals[m])
                #inds = ownership[m]
                inds = point_models==m
                plt.scatter(kp0[inds,0],kp0[inds,1],c='black',alpha=0.5,linewidths=(1,),s=40)
                #print(kp0[inds,0])
                #print(kp0[inds,1])
                plt.xlim([0,self.pc_w])
                plt.ylim([self.pc_h,0])
                plt.xticks([],[])
                plt.yticks([],[])
                #plt.title('Flow model {0}'.format(m))
                if m == n_models-1:
                    outfname = 'model_homography.png'
                elif m == n_models-2:
                    outfname = 'model_pca.png'
                else:
                    outfname = 'model_{0:02}.png'.format(m)
                plt.savefig(os.path.join(path,outfname),dpi=200,bbox_inches='tight',pad_inches=0)
                plt.close()
           
            plt.figure(figsize=(10,10),frameon=False)
            plt.imshow(res,cmap='cubehelix',vmin=0,vmax=n_models-1)
            #plt.title('Segmentation')
            #plt.colorbar()
            plt.xticks([],[])
            plt.yticks([],[])
            plt.savefig(os.path.join(path,'segments.png'),dpi=200,bbox_inches='tight',pad_inches=0)
            plt.close()

        



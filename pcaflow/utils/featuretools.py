#! /usr/bin/env python2

import numpy as np
import cv2


def get_feature_errors(u,v,kp0,kp1):
    """
    Get x and y errors for each feature

    """

    uv_feat = kp1-kp0
    
    kp0_ = np.floor(kp0).astype('int')
    u_gt = u[kp0_[:,1],kp0_[:,0]]
    v_gt = v[kp0_[:,1],kp0_[:,0]]

    err = uv_feat - np.c_[u_gt,v_gt]
    return err


def get_feature_gt(u,v,kp0):
    """
    Get ground truth kp1 for given feature points

    """

    kp0_ = np.floor(kp0).astype('int')
    u_gt = u[kp0_[:,1],kp0_[:,0]]
    v_gt = v[kp0_[:,1],kp0_[:,0]]

    kp1 = kp0 + np.c_[u_gt,v_gt]
    return kp0,kp1




def get_random_features_gt(u,v,n_features):
    """
    Extract n_features random features from given ground truth optical flow.

    """
    x,y = np.meshgrid(np.arange(u.shape[1]),np.arange(u.shape[0]))
    np.random.seed(0)

    xn = x + u
    yn = y + v

    ind = np.random.choice(np.arange(x.size),n_features,replace=False)

    kp0 = np.c_[x.ravel()[ind],y.ravel()[ind]]
    kp1 = np.c_[xn.ravel()[ind],yn.ravel()[ind]]

    ind_valid = np.logical_and(np.logical_and(kp0[:,0] < u.shape[1],
                                              kp0[:,1] < u.shape[0]),
                               np.logical_and(kp1[:,0] < u.shape[1],
                                              kp1[:,1] < u.shape[0]))

    kp0 = kp0[ind_valid,:]
    kp1 = kp1[ind_valid,:]

    return kp0.astype('float32'),kp1.astype('float32')

                       

def viz_features(I1,feats_I1,I2,feats_I2,axis=None):
    """
    Show features overlayed over I1 and I2. Requires matplotlib.

    """

    try:
        from matplotlib import pyplot as plt
        have_plt=True
    except:
        plt=None
        have_plt=False
    
    I_comb_ = np.vstack((I1,I2))
    I_comb = np.hstack((I_comb_,I_comb_))

    lines_x = np.vstack((feats_I1[:,0],feats_I2[:,0]))
    lines_y = np.vstack((feats_I1[:,1],feats_I2[:,1]))

    print(lines_x.shape)
    print(lines_y.shape)

    lines_y[1,:] += I1.shape[0]

    # set up points for render display on the right
    pt1 = feats_I1.copy()
    pt1[:,0] += I1.shape[1]
    pt2 = feats_I2.copy()
    pt2[:,0] += I1.shape[1]
    pt2[:,1] += I1.shape[0]

    if have_plt:
        print('We have Matplotlib!')
    else:
        print('No Matplotlib.')
        plt=None
        return

    if not axis==None:
        axis.imshow(I_comb)
        plt.gray()
        axis.plot(lines_x,lines_y)
        axis.scatter(pt1[:,0],pt1[:,1])
        axis.scatter(pt2[:,0],pt2[:,1])
        axis.set_xlim(0,I_comb.shape[1])
        axis.set_ylim(I_comb.shape[0],0)
    else:
        plt.figure()
        plt.imshow(I_comb)
        plt.gray()
        plt.plot(lines_x,lines_y)
        plt.scatter(pt1[:,0],pt1[:,1])
        plt.scatter(pt2[:,0],pt2[:,1])
        plt.xlim(0,I_comb.shape[1])
        plt.ylim(I_comb.shape[0],0)
        plt.show()
    
    
def compute_homographies(kp1, kp2, imagesize, num_matches=8, err_thresh=1.0, niter_max=30):
    """
    Compute a set of homographies from feature matches.

    Parameters:
        kp1,kp2 :       Aligned features, as returned by get_matched_features()
        imagesize :     Tuple (imagewidth,imageheight)
        num_matches :   How many neighbors to consider for first match.
                        Default: 8
        err_thresh :    Maximal distance in pixel to be considered an inlier.
                        Default: 1.0
        niter_max :     Maximum number of iterations when growing the planes.
                        Default: 30

    Returns:
        Tuple (H,P,Unmatched), with
              H :           List of homographies
              P :           List of point IDs, so that kp1[P[n]] are the points
                            belonging to H[n]
              Unmatched :   List of point IDs that are not matched.
    """


    # For each feature, get the 6 spatially closest matches.
    bfm = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches = bfm.knnMatch(kp1.astype('float32'),kp1.astype('float32'),k=num_matches+1)

    match_ids = np.zeros((len(matches),num_matches)).astype('int32')

    for i,m in enumerate(matches):
        match_ids[i,:][:] = [T.trainIdx for T in m[1:]]


    H_ar = []
    P_ar = []
    
    points_unused = np.ones(len(matches))
    mask_inliers_prev = np.zeros_like(points_unused)
    
    err_thresh_sq = err_thresh**2

    # Convert points to [X,Y,Z]^T matrices, so that the dot product can
    #   be used.
    kp1_matrix = np.vstack((kp1.T,np.ones(kp1.shape[0])))
    kp2_matrix = np.vstack((kp2.T,np.ones(kp2.shape[0])))

    for i,m in enumerate(matches):
        # Check if the current point is already used up.
        if (points_unused[i] == 0) or np.any(points_unused[match_ids[i,:]]==0):
            continue
        
        # Find the first N neighbours
        p_f1 = kp1[match_ids[i,:]]
        p_f2 = kp2[match_ids[i,:]]
        H,m = cv2.findHomography(p_f1,p_f2,cv2.RANSAC,err_thresh)
        
        # Save the number of points that are currently used for fitting.
        mask_inliers_prev[match_ids[i,:]] = 1
        
        
        if m.sum() == num_matches:
            # The N closest features are inliers.
            #   Now, refine.
            for j in range(niter_max):
                # For each feature, compute the distance using the currently
                #   estimated homography.
                projected_points = H.dot(kp1_matrix)
                projected_points = projected_points / projected_points[2,:]
                dists = ((kp2_matrix - projected_points)**2).sum(axis=0)
                mask_inliers = (dists <= err_thresh_sq) & (points_unused>0)
                if np.all(mask_inliers==mask_inliers_prev):
                    # The fit didn't change so we break.
                    break
                else:
                    # Update set of points with the new inliers, and
                    #   recompute the homography.
                    p_f1 = kp1[np.nonzero(mask_inliers)[0],:]
                    p_f2 = kp2[np.nonzero(mask_inliers)[0],:]
                    H,m = cv2.findHomography(p_f1,p_f2,cv2.RANSAC,err_thresh)
                    mask_inliers_prev[:] = mask_inliers
                    
            
            mask_inliers_ids = np.nonzero(mask_inliers)[0]
            print('i = %s. number of iterations = %s'%(i,j))
            
            #print('i = %s. Number of inliers = %s'%(i,mask_inliers.sum()))
            points_unused[mask_inliers] = 0
            H_ar.append(H)
            P_ar.append(mask_inliers_ids)

    return (H_ar,P_ar,np.nonzero(points_unused)[0])

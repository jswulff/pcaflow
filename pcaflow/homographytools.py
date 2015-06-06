#! /usr/bin/env python2

import numpy as np
import cv2

skimage_available = True
try:
    from skimage import measure
except:
    skimage_available = False
    measure = None

#@profile
def get_largest_cc(u,v):
    """
    Return mask with largest connected component in u,v

    """

    if not skimage_available:
        print('*** skimage is not available. get_larget_cc() will not work. ***')
        return np.ones_like(u).astype('bool')
    
    fxx = np.array([[1,-2.0,1.0]])
    fxy = np.array([[-0.25,0,0.25],[0.0,0,0],[0.25,0,-0.25]])
    fyy = fxx.T

    u_ = u.astype('float32')
    v_ = v.astype('float32')
    uxx = cv2.filter2D(u_,-1,fxx)
    uxy = cv2.filter2D(u_,-1,fxy)
    uyy = cv2.filter2D(u_,-1,fyy)

    vxx = cv2.filter2D(v_,-1,fxx)
    vxy = cv2.filter2D(v_,-1,fxy)
    vyy = cv2.filter2D(v_,-1,fyy)

    THRESH=0.1
    ue = np.logical_or(np.logical_or(np.abs(uxx)>THRESH, np.abs(uxy)>THRESH),np.abs(uyy)>THRESH)
    ve = np.logical_or(np.logical_or(np.abs(vxx)>THRESH, np.abs(vxy)>THRESH),np.abs(vyy)>THRESH)
    edg = np.logical_or(ue,ve)
    
    L = measure.label(edg.astype('int32'),neighbors=4)
    
    sums = np.bincount(L.ravel())
    biggest_cc = L==np.argmax(sums)
    return biggest_cc

#@profile
def get_homography(u,v,method='ransac',use_cc=False):
    """
    Get homography that best describes the largest connected component in u,v.
    
    METHOD can be one of 'normal', 'ransac', 'lsmeds'

    """
    mask_subsampling = np.zeros(u.shape,dtype='bool')
    mask_subsampling[::5,::5] = 1

    if not use_cc:
        point_indices = mask_subsampling
    else:
        L = get_largest_cc(u,v)
        point_indices = (L * mask_subsampling) == 1
    
    h,w = u.shape
    x,y = np.meshgrid(np.arange(w),np.arange(h))
    x1 = x[point_indices]
    y1 = y[point_indices]
    x2 = (x1 + u[point_indices])
    y2 = (y1 + v[point_indices])
    
    x_in = np.c_[x1,y1].astype('float32')
    x_out = np.c_[x2,y2].astype('float32')

    if method.lower() == 'normal':
        method_ = 0
    elif method.lower() == 'ransac':
        method_ = cv2.RANSAC
    else:
        method_ = cv2.LMEDS

    
    #print(method_) 
    H,_ = cv2.findHomography(x_in,x_out,method=method_,ransacReprojThreshold=1)
    return H

  
#@profile
def remove_homography_from_flow(u,v,method='ransac',use_cc=False,H=None):
    """
    Remove homography from u,v, return corrected flow fields and resulting homography.

    METHOD can be one of 'normal', 'ransac', 'lsmeds'
    """
    if H is None:
        H = get_homography(u,v,method=method,use_cc=use_cc)

    h,w = u.shape
    x,y = np.meshgrid(np.arange(w),np.arange(h))

    xn = x + u
    yn = y + v
    Hinv = np.linalg.inv(H)

    x_orig = np.c_[x.ravel(),y.ravel()].astype('float32')
    x_in = np.c_[xn.ravel(),yn.ravel()].astype('float32')[np.newaxis,:,:]
    x_out = cv2.perspectiveTransform(x_in,Hinv)
    flow_new = x_out[0,:,:]-x_orig
    un = flow_new[:,0].reshape((h,w))
    vn = flow_new[:,1].reshape((h,w))
    
    return un,vn,H


def remove_homography_from_points(kp0,kp1,H=None,method=cv2.RANSAC,thresh=1): # Before: 1
    if H is None:
        H,inliers = cv2.findHomography(kp0,kp1,method=method,ransacReprojThreshold=thresh)
        if H is None:
            # Try again, with the first match removed
            H,inliers = cv2.findHomography(kp0[1:],kp1[1:],method=method,ransacReprojThreshold=thresh)
            inliers = np.r_[[[0]],inliers]
    else:
        inliers = np.ones(kp0.shape[0])
    Hinv = np.linalg.inv(H)

    kp1_ = cv2.perspectiveTransform(kp1[np.newaxis,:,:],Hinv)
    return kp0,kp1_[0,:,:],H,Hinv,inliers


def apply_homography_to_flow(u,v,H):
    h,w = u.shape
    x,y = np.meshgrid(np.arange(w),np.arange(h))

    xn = x+u
    yn = y+v
    #print(xn.dtype)
    #print(yn.dtype)
    #print(H.dtype)
    xout = cv2.perspectiveTransform(np.c_[xn.ravel(),yn.ravel()][np.newaxis,:,:],H)
    u_ = xout[0,:,0].reshape(u.shape) - x
    v_ = xout[0,:,1].reshape(v.shape) - y
    return u_,v_


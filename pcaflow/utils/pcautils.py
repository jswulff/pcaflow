#! /usr/bin/env python2

import numpy as np
import cv2

def scale_u_v(u,v,newsize,nearest=False):
    if nearest:
        u_ = cv2.resize(u,newsize,interpolation=cv2.INTER_NEAREST)
        v_ = cv2.resize(v,newsize,interpolation=cv2.INTER_NEAREST)
    else:
        u_ = cv2.resize(u,newsize)
        v_ = cv2.resize(v,newsize)
    u_ *= newsize[0] / float(u.shape[1])
    v_ *= newsize[1] / float(u.shape[0])
    return u_,v_

def scale_uv(uv,newsize,nearest=False):
    u,v = scale_u_v(uv[:,:,0],uv[:,:,1],newsize,nearest)
    return np.dstack((u,v))


def epe_u_v(u_gt,v_gt,u_est,v_est):
    return np.sqrt((u_gt-u_est)**2 + (v_gt-v_est)**2).mean()

def epe_uv(uv_gt,uv_est):
    return epe_u_v(uv_gt[:,:,0],uv_gt[:,:,1],uv_est[:,:,0],uv_est[:,:,1])

def errors_feats_u_v(kp0,kp1,u_gt,v_gt,relative=False):
    kp_v = kp1-kp0
    if True:
        indices = np.floor(kp0).astype('int32')
        u_gt_feats = u_gt[indices[:,1],indices[:,0]]
        v_gt_feats = v_gt[indices[:,1],indices[:,0]]
    else:
        u_gt_feats = cv2.remap(u_gt,kp0[:,0],kp0[:,1],cv2.INTER_LINEAR)[:,0]
        v_gt_feats = cv2.remap(v_gt,kp0[:,0],kp0[:,1],cv2.INTER_LINEAR)[:,0]
        
    uv_gt_feats = np.c_[u_gt_feats,v_gt_feats]
    #print(uv_gt_feats.shape)

    err = np.sqrt(np.sum((kp_v-uv_gt_feats)**2,axis=1))
    if relative:
        err /= np.linalg.norm(uv_gt_feats,axis=1)
    return err


def angerrors_feats_u_v(kp0,kp1,u_gt,v_gt):
    kp_v = kp1-kp0
    indices = np.floor(kp0).astype('int32')

    u_gt_feats = u_gt[indices[:,1],indices[:,0]]
    v_gt_feats = v_gt[indices[:,1],indices[:,0]]

    uv_gt_feats = np.c_[u_gt_feats,v_gt_feats]
    #print(uv_gt_feats.shape)

    ang_gt = np.arctan2(v_gt_feats,u_gt_feats)
    ang_kp = np.arctan2(kp_v[:,1],kp_v[:,0])

    err_max = np.maximum(ang_gt,ang_kp)
    err_min = np.minimum(ang_gt,ang_kp)

    err = np.minimum(err_max-err_min,-(err_max-err_min-2*np.pi))
    return err
    

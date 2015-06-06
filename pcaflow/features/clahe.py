#! /usr/bin/env python2

import numpy as np
import cv2

def convert_bw(I):
    cl = cv2.createCLAHE(clipLimit=5.0)
    if I.ndim > 2:
        I_g = cv2.cvtColor(I,7)
    else:
        I_g = I
    I_ = cl.apply(I_g)
    return I_

def convert_color(I):
    cl = cv2.createCLAHE(clipLimit=10.0)
    I_ycbcr = cv2.cvtColor(I,37)
    I_ycbcr[:,:,0] = cl.apply(I_ycbcr[:,:,0])
    I_ = cv2.cvtColor(I_ycbcr,39)
    return I_

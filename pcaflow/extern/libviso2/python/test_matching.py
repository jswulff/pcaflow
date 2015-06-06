#! /usr/bin/env python2

import numpy as np
from skimage import io

from libvisomatcher import LibvisoMatcher

from matplotlib import pyplot as plt


I0 = io.imread('../img/I1_000000.png').astype('uint8')
I1 = io.imread('../img/I1_000001.png').astype('uint8')
I2 = io.imread('../img/I1_000002.png').astype('uint8')

# Interesting part
M = LibvisoMatcher()
M.pushBack(I0)
M.pushBack(I1)
matches_p,matches_c = M.getMatches()

#print(matches)

x0 = matches_p[:,0]
y0 = matches_p[:,1]
x1 = matches_c[:,0]
y1 = matches_c[:,1]

x_ = np.c_[x0,x1].T
y_ = np.c_[y0,y1].T

x_ = x_[:,::2]
y_ = y_[:,::2]

plt.figure()
plt.imshow(I0)
plt.gray()
plt.plot(x_,y_)

M.pushBack(I2)
matches_p,matches_c = M.getMatches()
x0 = matches_p[:,0]
y0 = matches_p[:,1]
x1 = matches_c[:,0]
y1 = matches_c[:,1]
x_ = np.c_[x0,x1].T
y_ = np.c_[y0,y1].T

x_ = x_[:,::2]
y_ = y_[:,::2]

plt.figure()
plt.imshow(I1)
plt.gray()
plt.plot(x_,y_)

plt.show()

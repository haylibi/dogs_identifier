# Add a Gaussian Noise to Image
import cv2
import numpy as np
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt

# original image
f = cv2.imread('finalize_dogs_vs_cats/cats/cat.0.jpg' )
f = f/255 

# create gaussian noise
x, y, z = f.shape
mean = 0
var = 0.01
sigma = np.sqrt(var)
n = np.random.normal(loc=mean, 
                     scale=sigma, 
                     size=(x,y,z))

# add a gaussian noise
g = f + n

# display all
cv2.imshow('original image', f)
#cv2.imshow('Gaussian noise', n)
cv2.imshow('Corrupted Image', g)

cv2.waitKey(0)
cv2.destroyAllWindows()
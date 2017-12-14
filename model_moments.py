import numpy as np
from scipy.special import lpmv
from scipy.stats import threshold
import cv2

# number of channels
dim = 3

# moments
m1 = np.zeros(dim)
m2 = np.zeros((dim,dim))

# training function
def train(points):
	global m1
	global m2
	
	# generate constants
	for i in range(dim):
		m1[i] = np.mean(points[:,i])
	for i in range(dim):
		for j in range(dim):
			m2[i,j] = np.mean((points - m1)[:,i] * (points - m1)[:,j])
	
	print('m1',m1)
	print('m2',m2)
	
#	distance = np.zeros(len(points))
#	for i in range(dim):
#		for j in range(dim):
#			distance += (points[:,i]-m1[i]) * (points[:,j]-m1[j]) / m2[i,j]
#	print(distance)
	

def get_mask(img):
	print('get')
	distance = np.zeros(img.shape[:2])
	for i in range(dim):
		for j in range(dim):
			distance[:,:] += (img[:,:,i]-m1[i]) * (img[:,:,j]-m1[j]) / m2[i,j]
	
	return cv2.inRange(distance, 0, 10)
	return distance




























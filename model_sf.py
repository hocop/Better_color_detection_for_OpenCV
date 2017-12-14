import numpy as np
from scipy.special import lpmv
from scipy.stats import threshold
import cv2

# set accuracy level
N = 4

# trainable variables
A = np.zeros((N+1,2*N+1))
B = np.zeros((N+1,2*N+1))
center = np.zeros(3)

# set constants
n_arr = np.arange(0,N+1)
m_arr = np.arange(-N,N+1)
filt = np.ones((N+1,2*N+1)).astype(int)
for l in range(N+1):
	for m in range(2*N+1):
		if np.abs(m_arr[m]) > n_arr[l]:
			filt[l,m] = 0
n_bigarr = np.reshape(np.repeat(n_arr, 2*N+1), (N+1,2*N+1))
m_bigarr = np.transpose(np.reshape(np.repeat(m_arr, N+1), (2*N+1,N+1))) * filt

# our boundary surface function
def f(xyz):
	global A
	global B
	global center
	global n_arr
	global m_arr
	global filt
	global n_bigarr
	global m_bigarr
	p = cartesian2spherical(xyz-center)
	theta, phi = p[:,:,1], p[:,:,2]
	cosP = np.cos(np.outer(phi,m_arr))
	sinP = np.sin(np.outer(phi,m_arr))
	cosT = np.cos(np.outer(theta,n_arr) + m_arr[N:-N]*np.pi/2)
	#P = lpmv(m_bigarr, n_bigarr, cosT)
	shape1 = (2*N+1,)+xyz.shape[:2]+(N+1,)
	shape2 = (N+1,)+xyz.shape[:2]+(2*N+1,)
	P = np.reshape(np.transpose(np.reshape(np.repeat(cosT, 2*N+1), shape1)), shape2)
	return np.sum((A*cosP + B*sinP) * P)

def cartesian2spherical(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    ptsnew[:,:,0] = np.sqrt(xy + xyz[:,:,2]**2)
    ptsnew[:,:,1] = np.arctan2(np.sqrt(xy), xyz[:,:,2]) # elevation angle: from Z-axis down
    ptsnew[:,:,2] = np.arctan2(xyz[:,:,1], xyz[:,:,0])
    return ptsnew

def sigmoid_deriv(x):
	x = threshold(x, -10.0, 10.0, 10)
	ex = np.exp(x)
	exp1 = ex + 1.0
	return ex / (exp1 * exp1)

# training function
def train(points):
	global A
	global B
	global center
	global n_arr
	global m_arr
	global filt
	global n_bigarr
	global m_bigarr
	
	epochs = 3000
	training_rate = 0.1
	alpha = 1.0
	const = 1.0 / len(points)
	
	# generate constants
	center = np.mean(points, 0)
	centers = np.transpose(np.reshape(np.repeat(center, len(points)), (3,len(points))))
	R = np.sqrt(np.max(np.sum((centers-points)**2,1)))
	print('radius',R)
	
	# init weights
	A[0,N] = R
		
	for epoch in range(epochs):
		for p in points:
			ps = cartesian2spherical(np.array([[p-center]]))
			r, theta, phi = ps[0][0][0], ps[0][0][1], ps[0][0][2]
			cosP = np.cos(phi*m_arr)
			sinP = np.sin(phi*m_arr)
			#cosT = np.cos(theta*n_arr)
			cosT = np.cos(theta*n_arr + m_arr[N:-N]*np.pi/2)
			#P = lpmv(m_bigarr, n_bigarr, cosT)
			P = np.transpose(np.reshape(np.repeat(cosT, 2*N+1), (2*N+1,N+1)))
			rep = alpha*(r-f(np.array([[p]])))/R
			if rep > 10:
				landscape = -rep
			elif rep < -10:
				landscape = const
			else:
				landscape = const - np.log(1.0 + np.exp(rep))
			grad_A = -landscape * cosP * P * filt
			grad_B = -landscape * sinP * P * filt
			
			A += grad_A * training_rate
			B += grad_B * training_rate
		if epoch % 1000 == 0:
			print(A[0,N])

def get_mask(img):
#	mask = np.zeros(img.shape[:2])
#	for i in range(len(img)):
#		for j in range(len(img[0])):
#			if np.sum((img[i,j]-center)**2) < f(img[i,j])**2:
#				mask[i,j] = 0.5
	mask = f(img) / 255
	return mask




























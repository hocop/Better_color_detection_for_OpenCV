import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_moments import *
import cv2
from os import listdir
from os.path import isfile, join

# prepare arrays: positive and negative HSV points
hp, sp, vp = [], [], []
hn, sn, vn = [], [], []
positives = []

# read data
for line in open('color_dataset').readlines():
	h, s, v, result = map(float, line[:-1].split('\t'))
	if result > 0.5:
		hp.append(h)
		sp.append(s)
		vp.append(v)
		positives.append([h,s,v])
	else:
		hn.append(h)
		sn.append(s)
		vn.append(v)

# train model
positives = np.array(positives)
train(positives)

#exit()

# read pictures
path = 'input_data/'
names = [f for f in listdir(path) if isfile(join(path, f))]
for f in names:
	# read image from file
	img = cv2.imread(path+f)
	img = cv2.cvtColor(cv2.resize(img, (400,400)), cv2.COLOR_BGR2HSV)
	mask = get_mask(img)
	mask = cv2.dilate(mask, None, iterations=2)
	mask = cv2.erode(mask, None, iterations=2)
	print(mask.shape)
	cv2.imshow('mask',mask)
	print('ok')
	cv2.waitKey(4000)

# draw graphics
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(hn, sn, vn)
ax.scatter(hp, sp, vp)
leg = plt.legend(['negative','positive'])

# draw ellipsoid
# Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
coefs = (1/m2[0,0], 1/m2[1,1], 1/m2[2,2])
# Radii corresponding to the coefficients:
rx, ry, rz = 1/np.sqrt(coefs)

# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x = rx * np.outer(np.cos(u), np.sin(v)) + m1[0]
y = ry * np.outer(np.sin(u), np.sin(v)) + m1[1]
z = rz * np.outer(np.ones_like(u), np.cos(v)) + m1[2]
print(m1,m2)

# Plot:
ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.5)

plt.show()

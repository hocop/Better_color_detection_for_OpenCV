import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from model import *
import cv2
from os import listdir
from os.path import isfile, join

# prepare arrays: positive and negative HSV points
hp, sp, vp = [], [], []
hn, sn, vn = [], [], []
positives = []
negatives = []

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
		negatives.append([h,s,v])

# train model
positives = np.array(positives)
negatives = np.array(negatives)
train(positives)

# read pictures
path = 'input_data/'
names = [f for f in listdir(path) if isfile(join(path, f))]
for f in names:
	# read image from file
	img = cv2.imread(path+f)
	img = cv2.resize(img, (400,400))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = get_mask(img)
	cv2.imshow('mask',mask)
	cv2.waitKey(1000)

# draw graphics
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(hn, sn, vn)
ax.scatter(hp, sp, vp)
leg = plt.legend(['negative','positive'])

# draw cubes
for c, r, vec in zip(model_centers, model_radii, model_vectors):
	p1 = c-vec*r
	Z = [p1, p1+np.array([r,0,0])*vec*2,p1+np.array([r,r,0])*vec*2
			,p1+np.array([0,r,0])*vec*2,p1+np.array([0,0,r])*vec*2
			,p1+np.array([r,0,r])*vec*2,p1+np.array([r,r,r])*vec*2
			,p1+np.array([0,r,r])*vec*2]
	verts =[[Z[0],Z[1],Z[2],Z[3]],
			[Z[4],Z[5],Z[6],Z[7]],
			[Z[0],Z[1],Z[5],Z[4]],
			[Z[2],Z[3],Z[7],Z[6]],
			[Z[1],Z[2],Z[6],Z[5]],
			[Z[4],Z[7],Z[3],Z[0]],
			[Z[2],Z[3],Z[7],Z[6]]]
	ax.add_collection3d(Poly3DCollection(verts, 
		facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model_sf import *
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

# read pictures
path = 'input_data/'
names = [f for f in listdir(path) if isfile(join(path, f))]
for f in names:
	# read image from file
	img = cv2.imread(path+f)
	img = cv2.cvtColor(cv2.resize(img, (400,400)), cv2.COLOR_BGR2HSV)
	mask = get_mask(img)
	print(mask.shape)
	cv2.imshow('mask',mask)
	print('ok')
	cv2.waitKey(1)

exit()
# draw graphics
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(hn, sn, vn)
ax.scatter(hp, sp, vp)
leg = plt.legend(['negative','positive'])
plt.show()

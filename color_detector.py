import numpy as np
#from scipy.special import lpmv
#from scipy.stats import threshold
import cv2

class ColorDetector:
	def __init__(self, points=None):
		# number of channels
		self.dim = 3
		
		# spheres of interest
		self.centers = []
		self.radii = []
		self.vectors = []
		self.ranges = []
		
		if points is not None:
			if type(points)==str:
				# prepare arrays: positive and negative HSV points
				hp, sp, vp = [], [], []
				hn, sn, vn = [], [], []
				positives = []
				negatives = []

				# read data
				for line in open(points).readlines():
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
			else:
				positives = points
			self.train(positives)
	
	# training function
	def train(self, points):
		points = np.array(points)
		self.dim = points.shape[1]
		
		# find total volume
		def findMinMax(s):
			min_point = np.zeros(self.dim)
			max_point = np.zeros(self.dim)
			for i in range(self.dim):
				min_point[i] = np.min(s[:,i])
				max_point[i] = np.max(s[:,i])
			return min_point, max_point
		min_point, max_point = findMinMax(points)
		V = np.prod(max_point - min_point)
		
		# find 'radius of point' as mean dist. to the closest other point
		r_point = np.mean([[np.min([np.sqrt(np.sum(np.square(p-p_p))) for p in points if not (p==p_p).all()]) for p_p in points]])
		
		# sets of points
		sets = [np.array([points[i] for i in range(len(points))])]
		
		# define functions which divide set into two
		def props(s):
			min_point, max_point = findMinMax(s)
			center = (min_point + max_point) / 2.0
			radius = max_point - min_point
			radius = np.sqrt(np.sum(radius*radius)) / 2.0 + r_point
			vector = max_point - min_point
			norm = np.sqrt(np.sum(vector*vector))
			if norm > 0:
				vector /= norm
			if len(s) == 1:
				radius = r_point
				vector = np.array([0.,0.,0.])
			return center, radius, vector
		
		def divide(s, dir_i):
			if len(s) == 1:
				return [s]
			center, radius, vector = props(s)
			if dir_i in range(1,4):
				vector[dir_i-1] *= -1
			if dir_i == 4:
				vector = np.array([1.0,0.0,0.0])
			if dir_i == 5:
				vector = np.array([0.0,1.0,0.0])
			if dir_i == 6:
				vector = np.array([0.0,0.0,1.0])
			s1, s2 = [], []
			for i in range(len(s)):
				if np.inner(s[i]-center, vector) > 0:
					s1.append(s[i])
				else:
					s2.append(s[i])
			s1 = np.array(s1)
			s2 = np.array(s2)
			if len(s1) == 0 or len(s2) == 0:
				return [s]
			c1, r1, v1 = props(s1)
			c2, r2, v2 = props(s2)
			distance = c2 - c1
			distance = np.sqrt(np.sum(distance*distance))
			n1,n2,n = len(s1),len(s2),len(s)
			if distance < (r1 + r2) / 2.0:
				return [s]
			else:
				return [s1, s2]
		
		dir_i = 0
		while True:
			if dir_i == 7:
				break
			sets_new = []
			for s in sets:
				sets_new += divide(s, dir_i)
			if len(sets_new) == len(sets):
				dir_i += 1
			else:
				sets = np.array(sets_new)
				dir_i = 0
		
		for s in sets:
			c, r, v = props(s)
			if (v == [0.,0.,0.]).any():
				continue
			self.centers.append(c)
			self.radii.append(r)
			self.vectors.append(v)
		for c, r, v in zip(self.centers, self.radii, self.vectors):
			lower = c - r * v
			upper = c + r * v
			self.ranges.append( (lower,upper) )
	
	def get_mask(self, img):
		mask = np.zeros(img.shape[:2])
		for lower, upper in self.ranges:
			mask += cv2.inRange(img, lower, upper)
		mask = cv2.inRange(mask, 0.5, 256*len(self.centers))
		return mask




























import cv2
from os import listdir
from os.path import isfile, join

path = 'input_data/'
# read picture file names
names = [f for f in listdir(path) if isfile(join(path, f))]

# open file to write results
data = open('color_dataset','a')

# this function extracts color
def click(event, x, y, flags, param):
	res = 1.0 if event == cv2.EVENT_LBUTTONDOWN else 0.0
	if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
		color = img[y,x]
		s = ('%d\t%d\t%d\t%d' % (color[0],color[1],color[2],res))
		print(s)
		data.write(s+'\n')

# create window and attach this function to it
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

for f in names:
	# read image from file
	img = cv2.imread(path+f)
	
	# resize it to better fit screen
	img = cv2.cvtColor(cv2.resize(img, (700, 700)), cv2.COLOR_BGR2HSV)
	
	# show
	cv2.imshow('image',img)
	cv2.waitKey(3000)

data.close()

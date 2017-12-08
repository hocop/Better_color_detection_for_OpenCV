**Advanced color detecting tool for OpenCV**

When we detect color on the image in opencv we use `cv2.inRange` function.  

However, sometimes accuracy of this approach is not satisfactory. Also it is usually difficult to define the range.

In this code I use multiple `cv2.inRange` calls in different regions of color space:  
![Points in HSV color space](Figure_1.png)

To define these multiple ranges I have written code which optimizes this choice using pre-collected data.  
To collect color data from input images, launch `collect_data.py` and click left mouse key on colors which you need.

For images present here, dataset is already collected, just launch `test.py` to see how it works.

To use this code you will need only `model.py`.  
```Python
from model import *
```
Then load dataset from file:
```Python
positives = []
negatives = []
# read data
for line in open('color_dataset').readlines():
	h, s, v, result = map(float, line[:-1].split('\t'))
	if result > 0.5:
		positives.append([h,s,v])
	else:
		negatives.append([h,s,v])
```


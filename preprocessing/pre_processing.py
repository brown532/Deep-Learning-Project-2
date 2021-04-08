from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

import glob
import os
import math


def square_image(img):
	x,y = img.size

	left= 0
	upper = 0
	right = x
	lower = y

	if x>y:
		left=int((x/2)-y/2)
		right=left+y
	elif y>x:
		upper=int((y/2)-x/2)
		lower=upper+x
	return img.crop((left,upper,right,lower))	

def preprocess(img,basewidth):
	
	x,y = img.size

	if x<y:
		wpercent = (basewidth/float(img.size[0]))
		hsize = math.ceil((float(img.size[1])*float(wpercent)))
		img = img.resize((basewidth,hsize), Image.ANTIALIAS)

	else:
		wpercent = (basewidth/float(img.size[1]))
		vsize = math.ceil((float(img.size[0])*float(wpercent)))
		img = img.resize((vsize,basewidth), Image.ANTIALIAS)

	img=square_image(img)

	img = np.asarray(img)
	
	return img/255



dims = 128
pictures = []
index=0
for file in glob.glob("flickr30k/*.*"):#flickr30k_images
	_, file_extension = os.path.splitext(file)
	if (file_extension == '.png') or (file_extension == '.jpg'):
		x = Image.open(file)
		x = preprocess(x,dims)

		if x.shape == (128,128,3): #ensure image is not black and white		

			pictures.append(x)

			# print(index,pictures[-1].shape)

			
			index += 1

			if index%500 == 0:
				print("Image",index)
				break



pictures = np.array(pictures)

np.random.shuffle(pictures)


print(pictures.shape)

np.save("flckr30k",pictures)

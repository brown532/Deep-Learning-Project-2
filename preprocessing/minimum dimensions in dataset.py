import glob
from PIL import Image
import numpy as np
import os

dimensions = np.zeros(shape=(0,2),dtype=int)
index=0
for file in glob.glob("flickr30k_images/*.*"):
	_, file_extension = os.path.splitext(file)

	if (file_extension == '.png') or (file_extension == '.jpg'):
		x = Image.open(file)
		dimensions = np.append(dimensions,[x.size],axis=0)

		
		index += 1

		if index%500 == 0:
			print("Image",index)

print(dimensions.shape)
print('min x:',min(dimensions.T[0]))
print('min y:',min(dimensions.T[1]))
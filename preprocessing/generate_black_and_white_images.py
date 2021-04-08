import numpy as np

x = np.load('Flickr30k.npy')

from PIL import Image
from matplotlib import pyplot as plt
new_image=0
element=0

totall = len(x)
b_w_images=[]

for indx,x_ in enumerate(x):
    if (indx % 200) == 0:
        print(str(indx)+" of "+str(totall))

    new_image = Image.fromarray(np.uint8(x_*255),'RGB').convert('L')
    new_image = np.asarray(new_image)/255

    b_w_images.append(new_image)


b_w_images = np.asarray(b_w_images)
b_w_images = np.expand_dims(b_w_images,axis=-1)



np.save("Flickr30kblackandwhite1dim",b_w_images)
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import random

import tensorflow as tf
from tensorflow.keras import datasets, layers, models,Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,Concatenate
import gc

from keras.activations import  relu
lrelu = lambda x: relu(x, alpha=0.2)


def generate_batch(size=32):
	colored_batch = random.sample(range(train_y.shape[0]),size)

	b_w_batch = random.sample(range(train_y.shape[0]),size)
	return [colored_batch,b_w_batch]


def generate_batches(size=32):
	batches = []
	index = 0
	while index < train_X.shape[0]:
		batches.append(generate_batch(size))
		index += size

	batches= np.array(batches)

	# print(batches.shape)

	return batches


def create_generator_model():
	# U-Net generator model from paper

	# TODO: implement skip connections
	generator = models.Sequential()

	# Downsampling
	generator.add(layers.Conv2D(64, (4, 4), activation='relu', input_shape=(128, 128, 1),strides=2, padding="same"))
	# No batch for first layer in generator

	generator.add(layers.Conv2D(64, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(128, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(256, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2D(512, (4, 4), activation='relu',strides=2, padding="same"))

	# Upsampling
	generator.add(layers.Conv2DTranspose(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(512, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(256, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(128, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(64, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(64, (4, 4), activation='relu',strides=2, padding="same"))
	generator.add(BatchNormalization())

	generator.add(layers.Conv2DTranspose(3, (4, 4), activation='sigmoid',strides=1, padding="same"))
	# No batch for last layer in generator

	return generator



def create_discriminator_model():
	# Discriminator model

	#Compares the colored and B&W image to know if the colored image was gotten from that specific B&W image
	imA = Input(shape = (128,128,3))
	imB = Input(shape = (128,128,1))
	combined_imgs = Concatenate()([imA,imB])

	l1 = (layers.Conv2D(64, (4, 4), activation=lrelu,strides=2, padding="same"))(combined_imgs)

	l2 = (layers.Conv2D(64, (4, 4), activation=lrelu,strides=2, padding="same"))(l1)
	l2 = (BatchNormalization())(l2)

	l3 = (layers.Conv2D(128, (4, 4), activation=lrelu,strides=2, padding="same"))(l2)
	l3 = (BatchNormalization())(l3)

	l4 = (layers.Conv2D(256, (4, 4), activation=lrelu,strides=2, padding="same"))(l3)
	l4 = (BatchNormalization())(l4)

	l5 = (layers.Conv2D(512, (4, 4), activation=lrelu,strides=2, padding="same"))(l4)
	l5 = (BatchNormalization())(l5)


	l6 = (layers.Conv2D(512, (4, 4), activation=lrelu,strides=2, padding="same"))(l5)
	l6 = (BatchNormalization())(l6)

	l7 = (layers.Conv2D(512, (4, 4), activation=lrelu,strides=2, padding="same"))(l6)
	l7 = (BatchNormalization())(l7)

	l8 = (layers.Conv2D(512, (4, 4), activation=lrelu,strides=2, padding="same"))(l7)
	l8 = (BatchNormalization())(l8)

	l9 = (layers.Conv2D(1, (4, 4), activation='sigmoid',strides=2, padding="same"))(l8)
	l9 = (BatchNormalization())(l9)

	l10 = (layers.Flatten())(l9)

	discriminator = Model(inputs=[imA, imB], outputs=l10)

	return discriminator



def create_combined_model(generator,discriminator):
	bw_input = Input((128,128,1))

	generated_image = generator(bw_input)

	discriminator.trainable=False

	validity = discriminator([generated_image,bw_input])

	combined = models.Model(bw_input,validity)

	opt = Adam(lr=0.0002, beta_1=0.5)

	combined.compile(optimizer=opt,loss='binary_crossentropy')

	return combined




train_X = np.load('Flickr8kblackandwhite1dim.npy') #greyscale images
train_y = np.load('flickr8k.npy') #colored images

test_X = train_X[7000:]
test_y = train_y[7000:]

train_X = train_X[:7000]
train_y = train_y[:7000]


generator = create_generator_model()

discriminator = create_discriminator_model()

combined = create_combined_model(generator,discriminator)




#############################
#Start Training
##############################
epochs=500 
batch_size=32

#Used to track loss progression in the models
d_loss_f = []
d_loss_r = []
g_loss = []


for epoch in range(epochs):
	print("Epoch",epoch)
	# print(generate_batches(batch_size).shape)#.shape)

	batches = generate_batches(batch_size)


	aa = []
	bb = []
	cc = []

	for batch_i,batch in enumerate(batches):
		colored_images = np.copy(train_y[batch[0]])
		b_w_images = np.copy(train_X[batch[1]])
		b_w_colored = np.copy(train_y[batch[1]])

		fake_images = generator.predict(b_w_images)

		discriminator_loss_r,_,_ = discriminator.train_on_batch((b_w_colored,b_w_images),np.array([1]*batch_size))
		discriminator_loss_f,_,_ = discriminator.train_on_batch((fake_images,b_w_images),np.array([0]*batch_size))


		generator_loss = combined.train_on_batch(b_w_images,[np.array([1]*batch_size),b_w_colored])

		gc.collect()

		aa.append(discriminator_loss_f)
		bb.append(discriminator_loss_r)
		cc.append(generator_loss)


	d_loss_f.append( sum(aa)/len(aa) )
	d_loss_r.append(sum(bb)/len(bb))
	g_loss.append( sum(cc)/len(cc))





	if epoch%50 == 0: # PLot the loss and colorize images with the generator
		plt.plot(d_loss_f,label='Discriminator loss-Fake')
		plt.plot(d_loss_r,label='Discriminator loss-Real')
		plt.plot((np.array(d_loss_f)+np.array(d_loss_r))/2, label='Discriminaor avg loss' )
		plt.plot(g_loss,label='Generator loss')

		plt.xlabel('epoch')
		plt.ylabel('loss')

		plt.plot()
		plt.legend()
		plt.show()


		img = random.sample(range(train_X.shape[0]),4)
		fake_images=generator.predict(train_X[img])
		fig,axes = plt.subplots(nrows=3, ncols=4)
		indx=0

		for i in img:
			axes[0][indx].imshow(train_X[i,:,:,0],cmap='gray')
			axes[0][indx].axis('off')
			axes[1][indx].imshow(train_y[i])
			axes[1][indx].axis('off')
			axes[2][indx].imshow(fake_images[indx])
			axes[2][indx].axis('off')

			indx+=1


		fig.suptitle('Training Performance')
		plt.show()

		print("Test: \n=================\n")

		img = random.sample(range(test_X.shape[0]),4)
		fake_images = generator.predict(test_X[img])
		fig, axes = plt.subplots(nrows=3, ncols=4)
		indx=0

		for i in img:
			axes[0][indx].imshow(test_X[i,:,:,0],cmap='gray')
			axes[0][indx].axis('off')
			axes[1][indx].imshow(test_y[i])
			axes[1][indx].axis('off')
			axes[2][indx].imshow(fake_images[indx])
			axes[2][indx].axis('off')
			indx+=1

		fig.suptitle('Testing Performance')
		plt.show()
        





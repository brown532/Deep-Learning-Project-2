# Deep-Learning-Project-2
# Colorization of Greyscale Images Using Generational Adversarial Networks

## Abstract
Image restoration is a process that deals with recovering an original image from a degraded version. It has many applications, from denoising to recoloring black and white photos, which are especially relevant for improving historical photographs and videos. This latter application is implemented here using a Conditional Generative Adversarial Network (CGAN). Conditional Generative Adversarial Networks (CGANs) have produced promising results for restoring degraded images in recent years. This paper reports specifically on the steps taken to train a CGAN for color restoration in grayscale images. We further compare the results of different regularization and optimization techniques of training the CGAN by computing the disparity of the generated color image from the ground truth. It was observed that the CGAN performed reasonably well, with the disparity between the generated image and the ground truth being less than 7 percent for all data sets and architectural configurations. More specifically, a minor disparity was identified as a product of the Flickr30k dataset and the baseline architecture with an average disparity value of 5.8 percent. In contrast, the most significant disparity occurred when applying the all leaky ReLu configuration to the Flickr8k dataset. The leaky ReLu architecture also produced the most unsatisfactory results overall. For all architectures, the CGAN produced better results when operating on the Flickr30k dataset. The baseline architecture was essentially tied with the all (non-leaky) ReLu architecture for performance over both sets. The difference between them was only 0.01 percent in favor of all ReLu architecture.  Additionally, the images subjectively appear similar to human observers, indicating the CGAN has, at least in some measure, properly determined the manifold describing natural color images. 

## How to repeat the experiments
- Download the [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k) or the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) to the working directory
- Make sure you have Python 3.8.5 installed
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`
- Run the preprocessing scripts; `pre_processing.py` and then `generate_black_and_white_images.py`
- Run the updated `pix2pix.ipynb` script


## Folder Description
- **preprocessing**:<br/>
	`preprocessing.py` goes through a selected folder containing the image dataset, crops these images to `128x128x3`. It then shuffles and saves the dataset. `generate_black_and_white_images.py` creates and saves the greyscale versions of the images.<br/>

- **pix2pix**: Utilized model<br/>


- **Alternate considered model**: Contains an implementation of a model for the same task.


## Results
## Loss progression
<img src="results/original.png" width="700"><br/>
## Model trained on Flickr8k dataset and Flickr30k dataset
<img src="results/result1.png" width="700"><br/>
<img src="results/result2.png" width="700"><br/>
<img src="results/result3.png" width="700"><br/>
<img src="results/result4.png" width="700"><br/>

## Authors
- [Brown Ogum](https://github.com/brown532)
- [Cosmin Roger Harsulescu](https://github.com/cosminroger)
- [Lars Cordes](https://github.com/L-Cordes)
- [Valentin Alexandru Tanase](https://github.com/ValyT)

## References
- Add Flckr8k, Flickr30k and test set
- Add paper of implementation




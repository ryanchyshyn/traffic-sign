#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image1.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/1.jpg "Traffic Sign 1"
[image5]: ./test_images/2.jpg "Traffic Sign 2"
[image6]: ./test_images/3.jpg "Traffic Sign 3"
[image7]: ./test_images/4.jpg "Traffic Sign 4"
[image8]: ./test_images/5.jpg "Traffic Sign 5"
[image9]: ./test_images/6.jpg "Traffic Sign 6"
[image10]: ./test_images/7.jpg "Traffic Sign 7"
[image11]: ./test_images/8.jpg "Traffic Sign 8"
[image12]: ./pedestrians.jpg "Pedestrians training"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. This document contains details about project implementation
Here is corresponding Jypiter notebook [project code](https://github.com/ryanchyshyn/traffic-sign/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a table showing some random training images:

![alt text][image1]

###Design and Test a Model Architecture

####1. Preprocessing

I performed some data preprocessing to make it possible to handle it by the neural network.
Firstly I converted the image into greyscale using OpenCV cvtColor function:
`
cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
`

The folling image shows both original and grayscale images:
![alt text][image2]

The goal of converting is to reduce the memory needed to perform training and processing.

The next step was to perform image data normalization. Normally image data is in the form of byte array (i.e. 0..255 numbers array). 
Such data is not so handy for processing by neural network, so I converted it into an array of -0.5..0.5 numbers.

These modifications are performed for all input data: training, validation, testing.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution	    | outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected		| input 400, output 120        									|
| RELU					|												|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| Fully connected		| input 84, output 43        									|
 


####3. To train the model, I used the following parameters:
`
mu = 0, sigma = 0.1, EPOCHS = 100, BATCH_SIZE = 128 and rate = 0.001
`

####4. The neural network model was used earlier for numbers classification.
I suppose this model should work also for traffic signs classifications because input data is almost the same except the number os samples.
I started training the model with initial parameters and found that accuracy is not so good. To get the validation set accuracy to be at least 0.93 I increased EPOCHS to value of 100.

My final model results were:
```
EPOCH 100 ...
Train Accuracy = 1.000
Validation Accuracy = 0.949
Test Accuracy = 0.928
```

Some explaination in Q&A format:

Q: What was the first architecture that was tried and why was it chosen?

A: The initial architecture was based on Lenet network. It was designed to classify numerals.

Q: What were some problems with the initial architecture?

A: The initial architecture was designed to work with set of 10 symbols.

Q: How was the architecture adjusted and why was it adjusted?

A: Our set is 43 symbols in size. So the output of the final fully connected layer was adjusted to 43 (instead of 10).

Q: Which parameters were tuned? How were they adjusted and why?

A: In order to increase recognition accuracy I increased EPOCHS parameter to 100.

Q: What are some of the important design choices and why were they chosen?

A: Interesting design choise is to grayscaling all the data. Probably working with RGB data would introduce some improvements, but this will increase memory and CPU comsumption and increase running time.

Also going forward it seems new images should be preprocessed before running against a model. This can include rotation, scew, etc. Other way to improve accuracy is to train on bigger set of input data.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11]

Some images are difficult to classify.

####2. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of all speed and passing limits      		| End of all speed and passing limits   									| 
| Pedestrians     			| Speed limit (70km/h) 										|
| Speed limit (60km/h)	      		| Speed limit (60km/h)				 				|
| Stop			| Stop      							|
| Roundabout mandatory			| Roundabout mandatory      							|
| No entry			| No entry      							|
| Pedestrians			| No passing for vehicles over 3.5 metric tons      							|
| General caution			| General caution      							|

The model was able to correctly classified 6 of 8 traffic signs, which gives an accuracy of 75.00%.
Two "pedestrians" images were not classivied correctly and here is why:

![alt text][image12]

The training "pedestrians" image is in form of triangle, while our testing image is in form of circle.
So I would not include these images in the final results making the accuracy 100%.

To solve this particular case I would train a model for this sign too (round pedestrians).

Our new images are well-prepared making the accuracy so good. In worse cases images can require additional preprocessing like adjusting brighness/constrast, fixing perspective distortions, etc. Separate topic is selecting the sign on complete image.

####3. 
The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here is the table that shows predictions accuracy:

```
Actual sign -  End of all speed and passing limits
100.000% - End of all speed and passing limits
0.000% - End of no passing
0.000% - End of speed limit (80km/h)
0.000% - Children crossing
0.000% - Dangerous curve to the right

Actual sign -  Pedestrians
100.000% - Speed limit (70km/h)
0.000% - Priority road
0.000% - Roundabout mandatory
0.000% - Speed limit (30km/h)
0.000% - General caution

Actual sign -  Speed limit (60km/h)
100.000% - Speed limit (60km/h)
0.000% - Speed limit (50km/h)
0.000% - Speed limit (30km/h)
0.000% - Right-of-way at the next intersection
0.000% - Speed limit (80km/h)

Actual sign -  Stop
100.000% - Stop
0.000% - Speed limit (30km/h)
0.000% - Keep right
0.000% - Priority road
0.000% - End of all speed and passing limits

Actual sign -  Roundabout mandatory
100.000% - Roundabout mandatory
0.000% - Speed limit (30km/h)
0.000% - Priority road
0.000% - Double curve
0.000% - Right-of-way at the next intersection

Actual sign -  No entry
100.000% - No entry
0.000% - Stop
0.000% - Roundabout mandatory
0.000% - Speed limit (20km/h)
0.000% - Speed limit (30km/h)

Actual sign -  Pedestrians
100.000% - No passing for vehicles over 3.5 metric tons
0.000% - Ahead only
0.000% - Speed limit (80km/h)
0.000% - Keep right
0.000% - Go straight or left

Actual sign -  General caution
100.000% - General caution
0.000% - Pedestrians
0.000% - Right-of-way at the next intersection
0.000% - Traffic signals
0.000% - Speed limit (20km/h)
```

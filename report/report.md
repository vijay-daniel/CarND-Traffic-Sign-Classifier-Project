#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[orig-ghn]: ./orig-ghn.png "Original Image"
[ghn]: ./ghn.png "Gray/Histogram Equalized/Normalized"
[before-pre]: ./before-pre.png "Before Preprocessing"
[after-pre]: ./after-pre.png "After Preprocessing (Normalized)"
[classes-dist]: ./classes-dist.png "Classes Distribution"
[web-images]: ./web-images.png "Web Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the many samples are present for each output class in the training and validation sets.

![classes-dist]

This section also compares the grayscale, histogram equalized and normalized transformations of a random training image. The observation here is that the normalized image

![orig-ghn]
![ghn]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the features that describe a sign are independent of colour. So, we do not need the additional colour information and can instead let the network look at the other aspects of the image without using up its weights for recognizing colour features.

As a last step, I normalized the image data because ...


Here is an example of a traffic sign image before and after preprocessing.

![before-pre]
![after-pre]

I did a few training runs with the histogram equalized version of the image, but the normalized versions showed better results in general. There were certain cases where the histogram equalized version gave a better visual representation than the normalized image.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 99.9%
* validation set accuracy of 97.2%
* test set accuracy of 95.1%


The process of tweaking the architecture was equally fun and frustrating :)


My first architecture was LeNet-5 without any preprocessing of the image. This converged to an accuracy of around 85% on the validation set. After converting the image to grayscale and normalizing it, the same architecture converged to an accuracy of around 93% on the validation set.

Based on the premise that traffic signs contain far more features that need to be "remembered" and "recalled" as opposed to the MNIST data set, I decided to start tweaking the LeNet architecture. During this tweaking, I realized I was making frequent errors while calculating the size of the transformed images as it flowed through an architecture. So, I built a simple pipeline framework to simplify the specification of an architecture and also help me in experimenting quickly with various architectures. The calculation of sizes is done automatically with some sane defaults. The code for the pipeline framework can be seen in the 7th code cell in the IPython noteboo. An example architecture written using this pipeline looks like the following:

``` 
def pipeline_sample(x):
    return run_pipeline(x, image_shape, #Input: 32x32xD
    [
        convp(3, 3, 1, "CLayer1"), # 30x30x3
        convp(5, 6, 1, "CLayer2"), # 26x26x6
        max_poolp(2, 2), # 13x13x6
        convp(6, 18, 1, "CLayer3"), # 8x8x18
        max_poolp(2, 2), # 4x4x18
        local_flattenp(), # 288
        fcp(164),
        fcp(101),
        dropoutp(),
        fcp(n_classes, False) # 43, Don't apply RELU here
    ])   
```


Even though these architectures were not as deep as others, adding dropout did help in increasing the accuracy, typically, by around 3-4%. After experimenting with various dropout values, I settled on 0.5 as the keep probability. Values less than this (0.3, 0.4) resulted in underfitting and very slow training. Values above this (0.7, 0.8) resulted in overfitting and lower accuracies than a value of 0.5.

I tried giving high batch sizes like 1024 and 2048. However, this caused the accuracy to drop to the order of 5-10%. My suspicion is that the big batch size started causing overfitting. Interestingly, a batch size of 128 seemed to be just right. I don't know the reasons behind why this number is right, but empirically, this gave better results than the other bigger batch sizes I tried.

This classic case of image classification falls right into the stronghold of CNNs. I tested various filter sizes for the convolution. I observed that the smaller the filter size, the finer the nature of the features understood by that layer. I verified this using Tensorboard visualizations.


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![web-images]

and here are their classifications:
[27-Pedestrians], [18-General caution], [2-Speed limit (50km/h)], [13-Yield], [1-Speed limit (30km/h)]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
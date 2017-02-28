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
[aug-flip]: ./aug-flip.png "Augmented - Flip"
[aug-bright]: ./aug-brightness.png "Augmented - Brightness"
[aug-saturation]: ./aug-saturation.png "Augmented - Saturation"
[web-soft-1]: ./web-soft-1.png "Soft1"
[web-soft-2]: ./web-soft-2.png "Soft1"
[web-soft-3]: ./web-soft-3.png "Soft1"
[web-soft-4]: ./web-soft-4.png "Soft1"
[web-soft-5]: ./web-soft-5.png "Soft1"


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
* The size of validation set provided in the download is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the many samples are present for each output class in the training and validation sets.

![classes-dist]

This section also compares the grayscale, histogram equalized and normalized transformations of a random training image. I've discussed the comparsion between these images in more detail in the next section.

![orig-ghn]
![ghn]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 7th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the features that describe a sign are independent of color. So, we don't need the additional color information and can instead let the network look at other aspects of the image without using up its weights for recognizing color features.

As a last step, I normalized the image data to enable my network to better respond to images with poor contrast (https://en.wikipedia.org/wiki/Normalization_(image_processing)). My personal observation here while sampling various images and normalizing them was that normalization also helped in better visualizing images with poor lighting.

Here is an example of a traffic sign image before and after preprocessing. The code for this in cell 8.

![before-pre]
![after-pre]

I did a few training runs with the histogram equalized version of the image, but the normalized versions showed better results in general. There were certain cases where the histogram equalized version gave a better visual representation than the normalized image. There was a slight improvement in performance (3-4%) after preprocessing the images. Honestly, I'm still not sure whether this was because the network responded better or because it was a hyperparameter tuning problem.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Validation data was already present in the downloaded data set. So, I didn't have to split my data into training and validation data.

My training set had 34799 images. My validation set and test set had 4410 and 12630 images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the training data set may not be representative of multiple factors of noise. To add more data to the the data set, I used TensorFlow's random augmentation functions to randomly add either brightness, or contrast, or hue or saturation.

Here are a few examples of an original image, an augmented image and the difference between these two images:

![aug-flip]
![aug-bright]
![aug-saturation]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Normalized Image						| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 30x30x6 |
| RELU					|			-			|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x12 |
| RELU					|			-			|
| Avg pooling	      	| 2x2 stride,  outputs 14x14x12 |
| Convolution 5x5    	| 1x1 stride, VALID padding, outputs 10x10x32 |
| RELU					|			-			|
| Avg pooling	      	| 2x2 stride,  outputs 5x5x32 |
| Flatten						|	800		|
| Fully Connected						|		164 |
| RELU | - |
| Fully Connected | 101 |
| RELU | - |
| Dropout | Keep probability of 0.5 |
| Fully Connected | 43 |
| Softmax + L2 Regularization | Weights only (not biases) are L2 regularized |


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the cell 12 of the ipython notebook. 

To train the model, I used an Adam optimizer. I came to know that such an optimizer existed by looking at the code from the LeNet lab. Upon further investigation, I understood that this is better than naive Gradient Descent in most cases because it associates a decay with the learning rate over a period of time which is helpful in not jumping over minimas during the later stage of training.

I tried giving high batch sizes like 1024 and 2048. However, this caused the accuracy to drop to the order of 5-10%. My suspicion is that the big batch size started causing overfitting. I tried lower batch sizes like 64, 32 as well, but the results were no different from the ones I got for 128. So, I chose to stick around with 128. This gave better results than the other bigger batch sizes I tried (256, 1024, 2048).

I tried various learning rates from as low as 0.00001 to as high as 0.01. Very low learning rates required a lot of epochs to converge and very high rates typically converged to lower values of accuracy. I finally settled on a value of X because it seemed to work well for my network.

One thing that really stood out was the fact that how the science of training neural networks is still really young. Even looking around on the internet and incorporating suggestions such as L2 norms (also suggested by my mentor) didn't really help in drastic accuracy improvements. There was also no clear way to find out whether a particular accuracy was the maximum possible performance I could extract out from my architecture or a hyperparameter tuning problem for my learning. :)


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the cell 11, 13 and 14 of the Ipython notebook.

My final model results were:

* training set accuracy of 99.7%
* validation set accuracy of 92.3%
* test set accuracy of 95.7%

The low validation set accuracy is low because ~30% of the validation images are distorted ones. However, it looks like some features have actually been captured better because of the augmentation done. This is reflected in the test set having more accuracy than the validation set and the good performance on the web images as described below.

The process of tweaking the architecture was equally fun and frustrating :)

My first architecture was LeNet-5 without any preprocessing of the image. This converged to an accuracy of around 85% on the validation set. After converting the image to grayscale and normalizing it, the same architecture converged to an accuracy of around 93% on the validation set.

Based on the premise that traffic signs contain far more features that need to be "remembered" and "recalled" as opposed to the MNIST data set, I decided to start tweaking the LeNet architecture and roll out my own in order to understand things better. During this tweaking, I realized that I was making frequent errors while calculating the size of the transformed images as it flowed through an architecture. So, I built a simple pipeline framework to simplify the specification of an architecture and also help me in experimenting quickly with various architectures. The calculation of sizes is done automatically with some sane defaults. The code for this basic pipeline framework can be seen in the 9th code cell in the IPython notebook. An example architecture written using this pipeline looks like the following:

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

I tried keeping just a couple of convolutional layers. However, various tweakings of this shallow architecture still resulted in an accuracy of 85%. 

I tried out average pooling instead of max pooling as well, but there was no significant difference in performance between the two. However, I still stuck to average pooling.

Even though these architectures were not as deep as others, adding dropout did help in increasing the accuracy, typically, by around 3-4%. After experimenting with various dropout values, I settled on 0.5 as the probability for retaining weights. Values less than this (0.3, 0.4) resulted in underfitting and very slow training. Values above this (0.7, 0.8) resulted in overfitting and lower accuracies than a value of 0.5.

This classic case of image classification falls right into the stronghold of CNNs. I tested various filter sizes for the convolution. I observed that the smaller the filter size, the finer the nature of the features understood by that layer. I visualized this using image summaries in TensorBoard. However, I had to remove the code because I had a bug which forced me to restart the kernel for every run.

Cell 11 also contains code to calculate the precision and recall of each epoch. Also, match percentage is calculated as sum(actual_matches)/sum(expected_matches)). Here is the corresponding table for the test data set:

```
No  Sign                                               Match%    Prc    Rcl
0   Speed limit (20km/h)                                0.967    1.0  0.967
1   Speed limit (30km/h)                                1.061  0.933   0.99
2   Speed limit (50km/h)                                1.108  0.897  0.993
3   Speed limit (60km/h)                                0.996  0.944   0.94
4   Speed limit (70km/h)                                1.005  0.983  0.988
5   Speed limit (80km/h)                                1.003  0.953  0.956
6   End of speed limit (80km/h)                         0.807  0.975  0.787
7   Speed limit (100km/h)                               1.036  0.938  0.971
8   Speed limit (120km/h)                               0.976  0.954  0.931
9   No passing                                          1.033  0.968    1.0
10  No passing for vehicles over 3.5 metric tons        0.994  0.991  0.985
11  Right-of-way at the next intersection               0.943   0.99  0.933
12  Priority road                                         1.0  0.993  0.993
13  Yield                                               1.001  0.994  0.996
14  Stop                                                0.985  0.992  0.978
15  No vehicles                                         1.014  0.972  0.986
16  Vehicles over 3.5 metric tons prohibited            0.993    1.0  0.993
17  No entry                                            1.014  0.986    1.0
18  General caution                                     0.979  0.961  0.941
19  Dangerous curve to the left                         1.067  0.922  0.983
20  Dangerous curve to the right                        0.967  0.966  0.933
21  Double curve                                        0.856  0.857  0.733
22  Bumpy road                                          0.858   0.99   0.85
23  Slippery road                                         1.4  0.714    1.0
24  Road narrows on the right                           0.711    1.0  0.711
25  Road work                                            0.99  0.989  0.979
26  Traffic signals                                     0.872  0.949  0.828
27  Pedestrians                                         0.717  0.698    0.5
28  Children crossing                                   1.013  0.974  0.987
29  Bicycles crossing                                   0.956  0.988  0.944
30  Beware of ice/snow                                   0.86  0.868  0.747
31  Wild animals crossing                               1.111    0.9    1.0
32  End of all speed and passing limits                 1.233  0.797  0.983
33  Turn right ahead                                    0.995   0.99  0.986
34  Turn left ahead                                       1.0  0.992  0.992
35  Ahead only                                          0.951  0.995  0.946
36  Go straight or right                                1.008  0.967  0.975
37  Go straight or left                                 0.883  0.981  0.867
38  Keep right                                          0.994  0.977  0.971
39  Keep left                                           0.689  0.984  0.678
40  Roundabout mandatory                                0.944  0.882  0.833
41  End of no passing                                    0.75    1.0   0.75
42  End of no passing by vehicles over 3.5 metric tons  1.122  0.881  0.989
```


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![web-images]

The first image may be difficult to classify because of the noise in the middle of the image. The second image also has a bit of dust on the sign. The 3rd image has a slight skew in the board. The 4th and 5th images are straightforward and should be  breeze for the network.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Cell 15 contains code to load images from a local folder. The code for making predictions on my final model is located in cell 16 of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrians   		| Pedestrians 			| 
| Road work    			| Road work				|
| 50 km/h				| 50 km/h				|
| Stop     		| Stop				|
| 30 km/h			| 30 km/h				|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Honestly, this was a fairly surprising result.

Before I augmented my images, the test set accuracy was around 93%, but there was only a 60% match on the images I downloaded from the web. After I augmented the images and trained the network, both the test set and web image accuracy increased. (I do understand that I ran the test set twice but I couldn't resist comparing the effect of augmenting the input images.)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

---


The network is completely confident about this class. This is somewhat surprising given the low precision (0.698) and recall (0.5) scores for this class.

![web-soft-3]

---


The network is again extremely confident about this class. This is supported by the high precision (0.989) and recall (0.979) scores for the Road Work class.

![web-soft-4]

---

For the below image, the network is around 84% confident that it is a 50 km/h speed limit sign.  This correlates fairly well to the precision score of ~89% on the test set. Given that the next highest probability of 0.15 is assigned to the 30 km/h speed limit, it looks like the system is a bit confused because of the close similarity between these two images except for the digits, 3 and  5.

![web-soft-1]

---

The network is extremely confident that this is a stop sign. Given that the probability is 0.97 for this class, other probabilities are pretty much trounced. This class also has high precision (0.992) and recall (0.978) scores.

![web-soft-2]

---

The network is completely confident about this class as well. What is interesting though is that while the network had a bit of confusion for the first image where it assigned around 15% confidence to the 30km/h sign while identifying the 50km/h sign, it seems to have no such confusion here while classifying this 30km/h sign. This is again evident very strongly in the precision (0.933) and recall (0.99) scores for this class. There is no significant difference in training samples between these two classes, but the network seems to have quite nicely latched on to the characteristics of this sign.

![web-soft-5]


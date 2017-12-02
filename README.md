## Vehicle Detection Project 5
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car_noncar](https://github.com/tugrulzure/carndp5/blob/master/report/1.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_feat](https://github.com/tugrulzure/carndp5/blob/master/report/2.png)

The code for the HOG feature extraction is contained in the fourth code cell of the jupyter notebook. 

#### 2. Explain how you settled on your final choice of HOG parameters.

I settled with YCrCb, although it is not the fastest colorspace, because I got almost 0.99 accuracy in the SVC classifier, unlike RGB and YUV colorspaces.
Going with RGB and YUV results in shorter HOG feature extraction times, but this is done only once, so I went with YCrCb in this project.
Number of directions depends on the computing power and considering this project is not going to operate in realtime anyways, I went with popular choice among Udacity students which are between 8-12 orientations, and there rest of the parameters are included in item 3 of the project.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, as shown in code cell 6 in the notebook.

My final parameters are shown below:

```python
colorspace = 'YCrCb'
orient = 10
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True 
hist_feat = True
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to hardcode the scales and the Y start and Y stop pixel values. Researching through previous projects and udacity forums, I decided to stick with a popular 3 scale pipeline, that has hardcoded Y pixel start-stop and scale values. Hardcoded overlapping pixel values work well, but I suspect this approach would work in a crowded street. So it should be upgraded to a dynamic scaling approach.
Code can be seen in code block 9 in the notebook.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good result.  Here are some example images:

![pipeline1](https://github.com/tugrulzure/carndp5/blob/master/report/5.png)
![pipeline2](https://github.com/tugrulzure/carndp5/blob/master/report/6.png)
![pipeline3](https://github.com/tugrulzure/carndp5/blob/master/report/7.png)

To improve the performance of the classifier, adding the heat map thresholding will prevent false detections, however when the threshold is set higher than it has to be, real detections can be filtered out like the third picture. Optimizing the thresholding parameter can be useful to obtain a better result as seen below.

![pipeline4](https://github.com/tugrulzure/carndp5/blob/master/report/8.png)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 
Heatmap is used in code blocks 8,9 and 10.

An example of heatmap is shown in below image:

![heat](https://github.com/tugrulzure/carndp5/blob/master/report/4.png)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem is that the bounding box bounces around and loses the detected vehicle sometimes in the video. In order to overcome this problem, I need to implement a global variable or class to keep track of the bounding boxes average position for the last couple frames.

Second problem is choosing HOG parameters and colorspaces. It requires a lot of trial and error to get it to work, and optimizing these parameters would require hours. Mine worked OK most of the time, but it's because I read up on it a lot before implementing the parameters. 

To make it more robust, first I would train the SVM with a larger dataset, average the previous bounding box detections, and implement a filter to normalize the input images/video, to compensate for the overly bright conditions or darker colored cars. A dynamic scaling for detecting cars is a must for a robust system, with more scales than three.

*Special thanks to:
https://github.com/jeremy-shannon/
https://github.com/ILYAmLV/
https://github.com/mithi/
and Udacity forums for code snippets and awesome tips.

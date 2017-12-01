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
Going with RGB and YUV results in longer HOG feature extraction times, but this is done only once, so I went with YCrCb in this project.
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

To improve the performance of the classifier, adding the heat map thresholding will prevent false detections, however when the threshold is set higher than it has to be, real detections can be filtered out like the third picture. Also, lowering the threshold resulted in more false positives, so I settled with the values I used in this project. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


# **Advanced Lane Finding Project**
The project is to Write a software pipeline to identify the lane boundaries in a video from a front-facing camera in a car. 

## Overview
When we drive, we use our eyes to decide where to go. In this project, we use front-facing camera to  automatically detect lane lines using an algorithm. This project applied Computer Vision algorithms to help my camera to see the lane boundaries and used lane line brute search algorithms and sliding window algorithms to draw lines boundaries for the self-driving-car. It includes two different sliding window algorithms, that the first one using brute search plus search from prior works well for slightly curved road conditions; the second pipeline used sliding windows by applying convolution solves the problem for hard-curved road condition.

[input video here] 


## Steps for Advanced Lane Finding
The steps of this pipeline are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary, in the project, I experimented 3 different algorithms and pipelines to gradually improve the search efficiency and lane line detection accuracy in hard road condition, please refer the algorithms [here](....).
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

![original image and lane boundatary detected image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/data_lane_annoted_image_2.png)
change the pic here

### Camera Calibration

Due to different angles, distance to the object, the real world view have serveral major distortions such as radial distortion and tangential distortion.Due to radial distortion, straight lines will appear curved, object will apper in a wrong shape. Its effect is more as we move away from the center of image.  To stably detect lane in a self-driving-car, we need to calibrate the camera view so that we can get the correct image.

what I did here is to use some sample images of a well defined pattern (eg, chess board) to find inner corners in chess board. We know its coordinates in real world space and  its coordinates in image. With these data, some mathematical problem is solved in background to get the distortion coefficients to undistort lane images in the video.

I start by preparing copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/undistorted_image.png)

### Color and Gradient 

I experimented with RGB, HSV, HLS color spaces. RGB does not work well in white or yellow. HSV's hue measures relative lightness or darkness of a color, it works better for white/yellow lane lines. HLS's saturation is a measurement of colorness, it works very well on white and yellow color in every light considition as it is independent of lightness.

I firstly used cv2.cvtColor(im, cv2.COLOR_RGB2HLS) to convert to HLS color space, then I read the S (saturation) value [:,:,2]. After applied a color threshold, I can use S channel to robustly pick up the lines in changing conditions.
```
 hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
 l_channel = hls[:,:,1]
 s_channel = hls[:,:,2]
 s_binary = np.zeros_like(s_channel)
 s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
```
Then I applied Sobel operator to the image to create a binary output of edges. Taking the gradient in x direction emphasizes edges closer to vertical; while  taking the gradient in the y direction emphasizes edges closer to horizonal. So I applied the gradient in x direction to detect vertical lane lines.
```
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
```

After that, I stacked each channel, and expanded to 256 pixels, and combined the two thresholds to output combinaed threaded binary output.
```
 color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
 combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
 ```
![threaded image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/combined_s_h_gradient_binary.png)

### Perspective Transform

The code for my perspective transform includes a function called `warped()`, which takes as inputs the threaded image, as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[722, 470],[1110, 720],[220,720],[570, 470]])
dst = np.float32([[920,1],[920, 720],[320,720],[320,1]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/threaded_warped_image.png)

### Lane Line Detection Algorithms
 
#### Brute Search: Identify lane-line pixels and fit their positions with a polynomial
**Line Finding Method: Peaks in a Histogram**
After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line. Plotting a histogram of where the binary activations occur across the image is one potential solution for this. With this histogram we are adding up the pixel values along each column in the image. In our thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. We can use that as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

**Implement Sliding Windows and Fit a Polynomial**
As shown in the previous animation, we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go. It takes the sum of the histogram of the bottom half from the warped image. Then I used sliding windows to identify the nonzero pixels in x and y within the window, then append these indices to the lists of left and right lanes. Then I applied a second order polynomial to each using `np.polyfit`.
```
leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
 ```
![output image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/output%20image.png)

#### Brute search for the first sliding window, then search from prior
After using sliding windows to track the lane lines out into the distance, but using the full algorithm from before and starting fresh on every frame may seem inefficient, as the lines don't necessarily move a lot from frame to frame. The second method used here is instead I just searched in a margin around the previous line position from the previous image/ or previous frame in a video to help track the lanes through sharp curves and tricky conditions.When I lose track of the lines, the algorithem will go back to your sliding windows search to rediscover the lines.

The left_lane_inds and right_lane_indsto hold the pixel values contained within the boundaries of a given sliding window. This time, we'll take the polynomial functions we fit before (left_fit and right_fit), along with a hyperparameter margin, to determine which activated pixels fall into the area of search based on activated x-values within the +/- margin of our polynomial function.

```
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
	```
	
After located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve: f(y) = Ay^2 + By + Cf(y)=Ay +By+C. The output looks like the following:

![output image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/sliding-window-search-prior.png)

#### Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

I firstly defined conversion variables in x and y from pixels  to meter by considering the physical lane is about 30 meters long and 3.7 meters wide (U.S. regulations) and the image pixel is about (700,720).  Then fit to a second polynominal to pixel positions in each lane line. Then used below math model to calculation of R_curve (radius of curvature).
```
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
	```
	
```
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

#### Draw the detected lane boundaries onto the road image
```
left_line_window = np.array(np.transpose(np.vstack([left_fitx, y_points])))
right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, y_points]))))
line_points = np.vstack((left_line_window, right_line_window))
cv2.fillPoly(stacked_img, np.int_([line_points]), [0,255, 0])
nwarped = cv2.warpPerspective(out_img, Minv, img_size , flags=cv2.INTER_LINEAR)
result = cv2.addWeighted(img, 1, unwarped, 0.3, 0) 
```

![Final image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/data_lane_annoted_image.png)

### Result
The current pipeline works great for project and challenge videos. For the third harder curved video, it is a good start but needs to improve the accuracy of the lane detection.
Here's a [link to project video](./project_video_output.mp4)
Here's a [link to challenge project video](./challenge_video_output.mp4)
Here is a  [link to harder challenge project video](./harder_challenge_video_output.mp4)


### Discussion
This is the most challenging and fun project for term 1. The first challenging task to undistort the video frames for the camera, it suggests to use at 20 images to calibrate the camera to compute a stable camera matrix and distortion coefficients. It is import to do tests of the camera matrix and distrotion coefficients for more pictures, so ensure the computation of the calibration is correct. In my case, at the first first, I just tested on one image, then I came back to test on all the 20 images, I found out there were a mistake for the computation. After I made the correct adjustment, the output of lane detection get much improved. 

My first iteration of the project missed some curved lane line for the project video. There are two causes. The first one is due to the perspective transform matrix, it needs some experiments to make it right. In that iteration, I picked a random road picture to get the perspective transformation matrix. Which turns out not a good strategy. In my second iteration, I adjusted pick a picture to calculate the transformation matrix using straight line road picture, as straight lines will remain straight even after the transformation. To find this transformation matrix, I picked need 4 points on the input image and corresponding points on the output image (actually they are the same pints). Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the function cv2.getPerspectiveTransform. 

I tested the warp on a straight stretch of road as I'll be able to measure the warp success by seeing the lanes parallel to each other. 

Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?
Discussion includes some consideration of problems/issues faced, what could be improved about their algorithm/pipeline, and what hypothetical cases would cause their pipeline to fail.

I spent quite some time on experimenting the different sliding window approaches to find lane lines. In my first iteration, I used the bruth search method to find lane pixcels to fit to polynominal. Then I implemented search from prior to search and adjust the lane lines sliding windows more efficiently.




# **Advanced Lane Finding Project**
The project is to Write a software pipeline to identify the lane boundaries in a video from a front-facing camera in a car. 

### Overview
When we drive, we use our eyes to decide where to go. In this project, we use front-facing camera to detect the lane lines boundaries. The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project I will detect lane lines boundaries in images and videos using Python and OpenCV. 

### Lane Boundaries Detection Steps
The steps of this pipeline are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[![Watch the video](https://i.ytimg.com/vi/yFeQ0b3oCmY/hqdefault.jpg?sqp=-oaymwEYCMQBEG5IVfKriqkDCwgBFQAAiEIYAXAB&rs=AOn4CLA3RgTk0x5mh-BBUWb1Uo5FLelMsQ)](https://www.youtube.com/yFeQ0b3oCmY)



#### Camera Calibration

Due to different angles, distance to the object, the real world view have serveral major distortions such as radial distortion and tangential distortion.Due to radial distortion, straight lines will appear curved, object will apper in a wrong shape. Its effect is more as we move away from the center of image.  To stably detect lane in a self-driving-car, we need to calibrate the camera view so that we can get the correct image.

what I did here is to use some sample images of a well defined pattern (eg, chess board) to find inner corners in chess board. We know its coordinates in real world space and  its coordinates in image. With these data, some mathematical problem is solved in background to get the distortion coefficients to undistort lane images in the video.

I start by preparing copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![original image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/undistorted_image)

#### Color and Gradient 

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
![original image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/threaded_image)

#### Perspective Transform

The code for my perspective transform includes a function called `warped()`, which takes as inputs the threaded image, as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[722, 470],[1110, 720],[220,720],[570, 470]])
dst = np.float32([[920,1],[920, 720],[320,720],[320,1]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![original image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/warped_image)

####  Identify  lane-line pixels and fit their positions with a polynomial

It takes the sum of the histogram of the bottom half from the warped image. Then I used sliding windows to identify the nonzero pixels in x and y within the window, then append these indices to the lists of left and right lanes. Then I applied a second order polynomial to each using `np.polyfit`.
```
leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
 ```
![original image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/output image)

#### Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

I firstly defined conversions in x and y from pixels space to meterr by considering the physical lane is about 30 meters long and 3.7 meters wide and the pixel space of lane is about (700,720).  Then fit to a second polynominal to pixel positions in each lane line. Then used below math model to calculation of R_curve (radius of curvature).
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

![original image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/data_lane_annoted_image_2)

---

### Pipeline (video)

#### Once the pipeline works well for images, it is ready to perform on videos.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

I implemented the sliding windows method to find lane pixels by tracking every frame, which is not the most efficient mthod. I plan to implement the other two methods one is to search in a margin around the previous line position and another is to to apply a convolution, which will maximize the number of "hot" pixels in each window. 

In general, my current lane boundaries pipeline works well with the tested videos. More tests and alrogithm improvement with real road test will be exciting.

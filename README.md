# **Advanced Lane Finding Project**
The project is to Write a software pipeline to identify and track the traffic lane boundaries in a video from a front-facing camera in a car. 

When we drive, we use our eyes to decide where to go. In this project, I use a front-facing camera as my self-driving-car's eye to  automatically detect and track lane lines using this python pipeline. This pipeline algorithm can calibrate its vision to see clearly the lane boundaries, and search lane lines using sliding window approach. It finally draws the lines boundaries so that human beings are updated with what the self-driving-car's vision and decisions. 

## Steps for Advanced Lane Finding
The steps of this pipeline are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary, in the project, I used two different algorithms, find lanes pixels to fit to polynomial and find lanes pixels then search from prior.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Below shows the lane boundary detection drew back onto its original road image.

![original image and lane boundatary detected image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/data_lane_annoted_image.png)

### 1. Camera Calibration

Due to different angles, distance to the object and so on, the real world view have serveral major distortions such as radial distortion and tangential distortion. Due to radial distortion, straight lines will appear curved, object will apper in a wrong shape. Its effect is more as we move away from the center of image.  To stably detect lane in a self-driving-car, we need to calibrate the camera view so that we can get the correct image.

what I did here is to use some sample images of a chess board (as it has a well defined pattern) to find its inner corners. We know its coordinates in real world space and  its coordinates in image. OpenCV provides the function to calculate the distortion coefficients to undistort the images. This parameters will be used to undistort lane views from the same camera.

I start by preparing copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then use the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
```
ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
```
I then tested the undistortation results to visually evaluate the result, they lookds like the following: 

![undistorted image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/undistorted_image.png)

### 2. Color and Gradient 

I experimented with RGB, HSV, HLS color spaces. RGB does not work well in white or yellow. HSV's hue measures relative lightness or darkness of a color, it works better for white/yellow lane lines. HLS's saturation is a measurement of colorness, it works very well on white and yellow color in every light considition due to its independent of lightness.

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

### 3. Perspective Transform

The code for my perspective transform includes a function called `warped()`, which takes as inputs the threaded image, as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[722, 470],[1110, 720],[220,720],[570, 470]])
dst = np.float32([[920,1],[920, 720],[320,720],[320,1]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/threaded_warped_image.png)

I also tested the warped transformation matrix (M) on how well it works, here is the result.
![test warped image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/warp-test-straight-line-2.png)

### 4. Lane Line Detection Algorithms
 After the data preparation, now it is the time to find lane lines. There are many potential approaches to do it. I experimented two ways explained as follows.
 
#### 4.1 Implement Sliding Windows and Fit a Polynomial
**Peaks in a Histogram to Fine Lane Lines**

After applying calibration, thresholding, and a perspective transform to a road image, I have a binary image where the lane lines stand out clearly. I then use peaks in a histogram to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line. Since in my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram are good indicators of the x-position of the base of the lane lines, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go. 
```
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

**Implement Sliding Windows and Fit a Polynomial**

After set up the the sliding window parameter and have a starting point for both lanes, the next step I loop for each of the sliding window (nwindows), with the given window sliding left or right and finds the mean position of activated pixels within the window. Here's a few steps:
* Loop through each window in nwindows
* Find the boundaries of our current window
```
for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
```
* Use cv2.rectangle to draw these window boundaries onto our visualization image out_img
* Find out which activated pixels from nonzeroy and nonzerox above actually fall into the window
* Append these to our lists left_lane_inds and right_lane_inds.
* Re-center the sliding window if the number of pixels found are greater than minpix, based on the mean position of these pixels
* Fit a polynomial to the line

```
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
```
The sliding windows for left and right lane lines look like the following:

![output image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/output%20image.png)

#### 4.2 Brute search for the first sliding window, then search from prior line position to adjust the window only when necessary
After using sliding windows to track the lane lines out into the distance, but may seem inefficient, 
An improved apporach is that since the lines don't necessarily move a lot from frame to frame, rather than  using the full algorithm from before and starting fresh on every frame, I use another apporach here is instead I just search in a margin around the previous line position from the previous window. When I lose track of the lines, the algorithem will go back to brute search sliding windows to rediscover the lines.

The left_lane_inds and right_lane_inds to hold the pixel values contained within the boundaries of a given sliding window. This time, I take the polynomial functions we fit before (left_fit and right_fit), along with a hyperparameter margin, to determine which activated pixels fall into the area of search based on activated x-values within the +/- margin of our polynomial function.

```
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
```
	
After located the lane line pixels, I used their x and y pixel positions to fit a second order polynomial curve: f(y) = Ay^2 + By + C, f(y)=Ay +By+C. The output looks like the following:

![output image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/search-from-prior.png)

#### 5. Calculate the radius of lane curvature and the position of the vehicle with respect to center

I firstly defined conversion variables in x and y from pixels  to meter by considering the physical lane is about 30 meters long and 3.7 meters wide (U.S. regulations) and the image pixel is about (700,720).  Then fit to a second polynominal to pixel positions in each lane line. Then I use below math model to calculation of R_curve (radius of curvature).

```
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
...
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
The center to road is calculated as following:
```
car_position = binary_img.shape[1]/2
l_fit_x_int = l_fit[0]*h**2 + l_fit[1]*h + l_fit[2]
r_fit_x_int = r_fit[0]*h**2 + r_fit[1]*h + r_fit[2]
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
```
#### 6. Draw the detected lane boundaries onto the road image

Below is how I draw the lane boundaries onto the original road image:
```
left_line_window = np.array(np.transpose(np.vstack([left_fitx, y_points])))
right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, y_points]))))
line_points = np.vstack((left_line_window, right_line_window))
cv2.fillPoly(stacked_img, np.int_([line_points]), [0,255, 0])
nwarped = cv2.warpPerspective(out_img, Minv, img_size , flags=cv2.INTER_LINEAR)
result = cv2.addWeighted(img, 1, unwarped, 0.3, 0) 
```

![Final image](https://github.com/zmandyhe/advanced-lane-finding/blob/master/output_images/lane-boundaries.png)

### Result
The current pipeline performs very well for project video which represents light curves conditions in a highway. It has many spaces to improve the pipeline performance for hard curved road conditions.
* Here's a [link to project video](./video_input_and_output/project_video_output.mp4)
* Here's a [link to challenge project video](./video_input_and_output/challenge_video_output.mp4)
* Here is a  [link to harder challenge project video](./video_input_and_output/harder_challenge_video_output.mp4)


## Discussion
This is the most challenging and fun project that used many computer vision algorithms. As the project result has demonstrated, it has many opportunities to improve the lane detection accuracies for harder road conditions. I think in order to have a better accuracy for hard curved road, the pre-processing the image/frame data and fine tuning parameters are essential, particularly to fine tune the thresholding binary image for color and space gradients. I can see in my project, for test5.jpg where contains many noises (nearby trees), the result for thresholding binary image to highlight lane lines has a great space to improve, even although the same parameters for thresholding other test images work very well.

It is important to undistort the video frames for the camera, it suggests to use at least 20 images to calibrate the camera to compute a stable camera matrix and distortion coefficients. Always test the camera matrix and distrotion coefficients on more pictures, to ensure the computation of the calibration is correct. 

My first iteration of the project missed some curved lane line for the project video. There are two causes. The first one is due to the unreliable perspective transform matrix, it took me some experiments to make it right. In that iteration, I picked a random road picture to get the perspective transformation matrix. Which turned out not a good strategy. In my second iteration, I adjusted my strategy, and picked a straight line road picture to calculate the transformation matrix, as straight lines will remain straight even after the transformation. To find this transformation matrix, I picked 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the applying the function cv2.getPerspectiveTransform(). There are algorithms to calculate the matrix for perspective transform, I will search and give it a try.
My first iteration of the project missed some curved lane line for the project video. There are two causes. The first one is due to the unreliable perspective transform matrix, it took me some experiments to make it right. In that iteration, I picked a random road picture to get the perspective transformation matrix. Which turns out not a good strategy. In my second iteration, I adjusted my strategy, and pick a straight line road picture to calculate the transformation matrix, as straight lines will remain straight even after the transformation. To find this transformation matrix, I picked 4 points on the input image and corresponding points on the output image. Among these 4 points, 3 of them should not be collinear. Then transformation matrix can be found by the applying the function cv2.getPerspectiveTransform(). There are algorithms to calculate the matrix for perspective transform, I will search and give it a try later.

It is worth to mention that I tested the warp result on other straight stretch of road as I'll be able to measure the warp success by seeing the lanes parallel to each other. 

I also experimented sliding window by applying convolution, the result was not better than the two apporaches used in this project. I like this project and will continue to experiment to work on complicated road conditions.
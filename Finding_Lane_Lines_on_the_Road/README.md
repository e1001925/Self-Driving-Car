# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**


[//]: # (Image References)

[image1]: ./res/images/solidYellowLeft.jpg "a solid line"

---

### Reflection

### Something about the pipeline

First, I converted the images to grayscale, then I run Hough on edge detected image which is masked the image.

For processing the video, a queue was create to calculate the line with moving average, in order to prevent sudden change of environment like noise.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by calculate the interaction point to draw the line.

Since there are noises, there are some corrupted line like no interaction point should be ignored. This may be caused by noise like too brighter light.

And the line should only be considered with certain slop.


### 2. Identify potential shortcomings with current pipeline


One potential shortcoming would be slow reaction to sudden changes. And noise filter is rough and may treat sudden change of environment as noise.


### 3. possible improvements 

There are so many places can be improved, such as many parameters are hard coded include queue size, I measured it based on very few tests. 

I have so few test cases and I really don't like hard coded parameter like this, it should be more flexible. Maybe can use some kind of algorithm like PID control to refine. I don't know.. 

The noise filter is also rough.. So few criterion for it. 

The ability of reacting to sudden changes is very weak. I guess my car system will simply crash when a monkey jumping on it.. 

The whole project is rough..   

Maybe later I will refine it when I have more time..


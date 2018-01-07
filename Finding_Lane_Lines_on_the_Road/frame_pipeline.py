import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import os
import collections
from os.path import join, basename


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def compute_slope(x1,y1,x2,y2):
    return (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
    
def compute_bias(x1,y1,x2,y2):
    return y1 - compute_slope(x1,y1,x2,y2) * x1

def candidate_decision(img, candidate, right, shape, color=[255, 0, 0], thickness=2):
    candidate_slope = []
    candidate_bias = []
    for line in candidate:
        for x1,y1,x2,y2 in line:
            candidate_slope.append(compute_slope(x1,y1,x2,y2))
            candidate_bias.append(compute_bias(x1,y1,x2,y2))
    slope = np.median(candidate_slope)
    bias = np.median(candidate_bias).astype(int)
    if right < 0:
        x1, y1 = 0, bias
        x2, y2 = right*np.int32(np.round(bias / slope)), 0
    else:
        x1, y1 = 0, bias
        x2, y2 = np.int32(np.round((shape[0] - bias) / slope)), shape[0]
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def make_no_solid_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = compute_slope(x1,y1,x2,y2)
            if 0.5 <= np.abs(slope) <= 2:
                     cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img;
    
def make_lines(img, lines, color=[255, 0, 0], thickness=2):
    right_candidate_slope = []
    right_candidate_bias = []
    left_candidate_slope = []
    left_candidate_bias = []
    is_line_l = 0
    is_line_r = 0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = compute_slope(x1,y1,x2,y2)
            if 0.5 <= np.abs(slope) <= 2:
                if slope > 0:
                    is_line_r = 1
                    right_candidate_slope.append(compute_slope(x1,y1,x2,y2))
                    right_candidate_bias.append(compute_bias(x1,y1,x2,y2))
                else:
                    is_line_l = 1
                    left_candidate_slope.append(compute_slope(x1,y1,x2,y2))
                    left_candidate_bias.append(compute_bias(x1,y1,x2,y2))

    if is_line_r < 1 or is_line_l < 1:#SKIP NOISE
        return  [0, 0, 0, 0]
    slope_r = np.median(right_candidate_slope)
    bias_r = np.median(right_candidate_bias).astype(int)
    slope_l = np.median(left_candidate_slope)
    bias_l = np.median(left_candidate_bias).astype(int)

    return  [slope_r, bias_r, slope_l, bias_l]
    
    
def draw_lines(img, history_line, color=[255, 0, 0], thickness=2):
    line = np.array(history_line)
    arr = np.ma.empty((len(history_line),4))
    arr.mask = True
    for i in range(0, len(line)):
        arr[:line.shape[0]] = line[i]
    arr.mean(axis = 0)
    slope_r = arr[0][0]
    bias_r = arr[0][1]
    slope_l = arr[0][2]
    bias_l = arr[0][3]
    intersection_x = np.int32(np.round((bias_l-bias_r)/(slope_r-slope_l)))
    intersection_y = np.int32(np.round(intersection_x*slope_r+bias_r))
    img = cv2.line(img, (intersection_x, intersection_y), (np.int32(np.round((img.shape[0] - bias_r) / slope_r)), img.shape[0]), color, thickness)
    img = cv2.line(img, (0, np.int32(np.round(bias_l))), (intersection_x, intersection_y), color, thickness)
    return img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, is_solid):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def color_frame_pipeline(color_image, is_solid, history_line):
    kernel_size = 15
    low_threshold = 30
    high_threshold = 120
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    for i in range(0, len(color_image)):
        line_image = np.copy(color_image[i])*0 # creating a blank to draw lines on
        img_gray = grayscale(color_image[i])
        blur_gray = gaussian_blur(img_gray, kernel_size)
        edges = canny(blur_gray, low_threshold, high_threshold)
        imshape = color_image[i].shape
        vertices = np.array([[(50,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap, is_solid)

    line_img = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)
    if is_solid > 0:
        this_line = make_lines(line_img, lines)
        if len(color_image) > 0:
            if this_line[0] > 0:
                if len(history_line) >= 220:#I REALLY DON'Y HAVE TIME TO OPTIMIZE...MAYBE LATER
                    history_line.pop()
                history_line.insert(0, this_line)
            line_img = draw_lines(line_img, history_line)
            img_blend = weighted_img(color_image[0], line_img, α=0.8, β=1., λ=0.)
        else:
            history_line.insert(0, this_line)
            line_img = draw_lines(line_img, history_line)
            img_blend = weighted_img(color_image, line_img[-1], α=0.8, β=1., λ=0.)
    else:
        if len(color_image) > 0:
            line_img = make_no_solid_lines(color_image[0], lines)
            img_blend = weighted_img(color_image[0], line_img, α=0.8, β=1., λ=0.)
        else:
            print("image")
            line_img = make_no_solid_lines(color_image, lines)
            img_blend = weighted_img(color_image, line_img[-1], α=0.8, β=1., λ=0.)
    return img_blend
    
def on_image(is_solid):
    raw_images_dir = join('raw_data', 'test_images')
    raw_images = [join(raw_images_dir, name) for name in os.listdir(raw_images_dir)]
    in_image = []
    history_line = collections.deque()
    for this_raw_image in raw_images:
        res_path = join('res', 'images', basename(this_raw_image))
        in_image.append(cv2.cvtColor(cv2.imread(this_raw_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        print(len(in_image))
        out_image = color_frame_pipeline(in_image, is_solid, history_line)
        cv2.imwrite(res_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))
        plt.imshow(out_image)
        plt.show()

def on_video(is_solid):
    raw_videos_dir = join('raw_data', 'test_videos')
    raw_videos = [join(raw_videos_dir, name) for name in os.listdir(raw_videos_dir)]
    for this_raw_video in raw_videos:
        history_line = collections.deque()
        capture = cv2.VideoCapture(this_raw_video)
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        res = cv2.VideoWriter(join('res', 'videos', basename(this_raw_video)),
                              cv2.VideoWriter_fourcc(*'DIVX'),
                              20.0, size, True)
        fgbg= cv2.createBackgroundSubtractorMOG2()
        frame_buffer = collections.deque(maxlen=10)
        while capture.isOpened():
            ret, color_frame = capture.read()
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, size)
                frame_buffer.append(color_frame)
                print(len(frame_buffer))
                blend_frame = color_frame_pipeline(frame_buffer, is_solid, history_line)
                res.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
            else:
                break
        capture.release()
        res.release()
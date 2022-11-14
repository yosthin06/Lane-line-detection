"""
sdc_library.py

Description: Library with all the functions needed for the lane line detection

Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
First created: Monday 27 october, 2022
Usage:

"""


# Import required libraries
from turtle import left
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as plt
import cv2
from sklearn import linear_model
import math
import argparse
import os
import time


def input_image(img_name):
    """
    This function reads the input image and opens it in a new window
    
    Inputs:
    img_name: name of the image to be read

    Outputs: 
    img_colour: image read by opencv and reduced to one fourth of the original size

    """

    #Verify that image exists
    try:
        # Be advissed that cv2.IMREAD_REDUCED_COLOR_4 reduces the
        # image size by one-fourth
        img_colour = cv2.imread(img_name, cv2.IMREAD_REDUCED_COLOR_4)
    except:
        print('ERROR: image', img_name, 'could not be read')
        exit()
    cv2.imshow("Colour image", img_colour)
    return img_colour

def greyscale_img(img_colour):
    """
    This function takes the image read before and turn it into greyscale
    
    Inputs:
    img_colour: image with RGB color

    Outputs: 
    grey: image read turned into greyscale

    """

    grey = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Greyscale image", grey) 
    return grey

def smoothed_img(grey, kernel_size):
    """
    This function takes the greyscale image and smooth it with a function called Gaussian blur.
    
    Inputs:
    grey : image in greyscale
    kernel_size: size of the kernel which the image will be smoothed with

    Outputs: 
    blur_grey: image with blur

    """

    blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
    cv2.imshow("smoothed image", blur_grey)
    return blur_grey

def canny_img(blur_grey, low_threshold, high_threshold):
    """
    This function detect the edges of the image
    
    Inputs:
    blur_grey: image with blur
    low_threshold: low range of the threshold needed for the Canny function
    high_threshold: high range of the threshold needed for the Canny function

    Outputs: 
    edges: image in black and white with the detected edges in white

    """

    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
    cv2.imshow("Canny image", edges)
    return edges

#Get a region of interest
def region_of_interest(img, vertices):
    """
    This function gets the region of interest of the image where we want to detect the lanes
    
    Inputs:
    img: image with detected edges
    vertices: vertices of the polygon area where we are putting the region of interest

    Outputs: 
    masked_image: image with detected edges only in the region of interest

    """
    
    #mask with 0
    mask = np.copy(img)*0
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow("Canny image within Region of Interest", masked_image)
    return masked_image



def hough(img_colour, roi_image, rho, theta, threshold, min_line_len, max_line_gap):
    img_colour_with_lines = img_colour.copy()
    hough_lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]),
                                 minLineLength=min_line_len, maxLineGap=max_line_gap)
    for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (255,0,0), 5)
    cv2.imshow('Hough lines', img_colour_with_lines)
    return hough_lines

def left_and_right_lines(hough_lines, img_colour):
    img_colour_with_left_and_right_lines = img_colour.copy()
    
    #lines arrays
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    try:
        #get slope
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) #slope
                if math.fabs(slope) < 0.3: #Only consider extreme slope
                    continue  
                
                if slope <= 0: #Negative slope, left group.
                    if x1 <= 450 and x2 <= 450: #only consider the left lines in the left side of the image
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                        cv2.line(img_colour_with_left_and_right_lines, (x1, y1), (x2, y2), (0,255,0), 5) #draw left line
                        
                else: #Otherwise, right group.
                    if x1 > 450 and x2 > 450: #only consider the right lines in the right side of the image
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
                        cv2.line(img_colour_with_left_and_right_lines, (x1, y1), (x2, y2), (0,0,255), 5) #draw right line
            
    except:
        print("ERROR! No hough lines detected")
    cv2.imshow('Left and right lines', img_colour_with_left_and_right_lines)
    return left_line_x, left_line_y, right_line_x, right_line_y
    
def lane_lines(left_line_x, left_line_y, right_line_x, right_line_y, img_colour):
    img_lane_lines = img_colour.copy()
    if len(left_line_x)>0 and len(left_line_y)>0 and len(right_line_x)>0 and len(right_line_y)>0:
    
        #min and max of the line
        min_y = 240
        max_y = 500
        
        #Create a function that match with all the detected lines
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        #get the start and the end
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        
        #Create a function that match with all the detected lines
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        #get the start and the end
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        #save points
        define_lines=[[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]]   
        
        #Add both lines
        for line in define_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lane_lines, (x1, y1), (x2, y2), (255,0,0), 12)
        
        cv2.imshow('LINES', img_lane_lines)

        return define_lines
    else:
        print("ERROR! No lane lines detected") 

        
        
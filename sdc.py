"""
sdc.py

Description: Leer una m¿imagen y detecta lineas 

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

#Get a region of interest
def region_of_interest(img, vertices):
    #mask with 0
    mask = np.copy(img)*0
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    cv2.imshow("Canny image within Region of Interest", masked_image)
    return masked_image

def Gblur(grey, kernel_size):
    # 3.- Apply Gaussian smoothing
    # ---filtrado espacial de la imagen---
    
    blur_grey = cv2.GaussianBlur(grey, kernel_size, sigmaX=0, sigmaY=0)
    cv2.imshow("smoothed image", blur_grey)
    return blur_grey

def masked_edges(blur_grey, low_threshold, high_threshold):
    edges = cv2.Canny(blur_grey, low_threshold, high_threshold, apertureSize=3)
    cv2.imshow("Canny image", edges)
    return edges

def hough(roi_image, rho, theta, threshold, min_line_len, max_line_gap):
    hough_lines = cv2.HoughLinesP(roi_image, rho, theta, threshold, np.array([]),
                                 minLineLength=min_line_len, maxLineGap=max_line_gap)
    return hough_lines
    
def pipeline(img_name):
    
    # 1.- Read image
    # Be advissed that cv2.IMREAD_REDUCED_COLOR_4 reduces the
    # image size by one-fourth
    img_colour = cv2.imread(img_name, cv2.IMREAD_REDUCED_COLOR_4)
    
    #Verify that image exists
    if img_colour is None:
        print('ERROR: image', img_name, 'could not be read')
        exit()
    #plt.imshow(img_colour)
    #plt.show()
    cv2.imshow("Colour image", img_colour)
    
    #2.- Convert from BGR to RGB then from RGB to greyscale
    grey = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Greyscale image", grey) 

    kernel_size = (9, 9)
    blur_grey = Gblur(grey, kernel_size)
    
    

    #Apply Canny edge detector
    low_threshold = 70
    high_threshold = 100
    edges = masked_edges(blur_grey, low_threshold, high_threshold)
    

    # 5.- Get a region of interest using the just created polygon
    # Define a Region-of-Interest. Cahnge the below vertices according
    # to input image resolution
    p1, p2, p3, p4, p5, p6, p7, p8 = (3, 438), (3, 296), (325, 237), (610, 237), (910, 320), (910, 438),(590,290),(340,290)

    # create a vertices array that will be used for the roi
    vertices = np.array([[p1, p2, p3, p4, p5, p6, p7, p8]], dtype=np.int32)
    roi_image = region_of_interest(edges, vertices)
    
    # 6.- Apply Hoguh transform for lane lines detection
    rho = 2                             # distance resolution in pixels of the Hough grid
    theta = np.pi/180                   # angular resolution in radians of the Hough grid
    threshold = 100                      # minimum number of votes (intersections in Hough grid)
    min_line_len = 10                    # minimum number of pixels making up a line
    max_line_gap = 30                   # maximum gap in pixels between connectable line segments
    hough_lines = hough(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    #print(f"detected lines:\n {hough_lines}")
    #print(f"Number of lines:\n {hough_lines.shape}")

    # 7.- Initialise a new images to hold the original image with teh detected lines
    img_colour_with_lines = img_colour.copy()
    img_colour_with_left_and_right_lines = img_colour.copy()
    img_lane_lines = img_colour.copy()
    left_lines, left_slope, right_lines, right_slope = list(), list(), list(), list()
    ymin, ymax, xmin, xmax = 0.0, 0.0, 0.0, 0.0
    x_left, y_left, x_right, y_right = list(), list(), list(), list()

    # Slope and standard deviation for left and right lane lines
    # This metrics were previously obtained after analysing the left and right 
    # lane lines for a 50-metre road section
    left_slope_mean, left_slope_std = -20.09187457, 3.40155536
    right_slope_mean, right_slope_std = 21.71384095, 1.7311898404

    if hough_lines is not None:
        #lines arrays
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        #get slope
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) #slope
                if math.fabs(slope) < 0.3: #Only consider extreme slope
                    continue  
                if slope <= 0: #Negative slope, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                    cv2.line(img_colour_with_left_and_right_lines, (x1, y1), (x2, y2), (0,255,0), 5)
                else: #Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
                    cv2.line(img_colour_with_left_and_right_lines, (x1, y1), (x2, y2), (0,0,255), 5)
        print(f"left lines in x: {left_line_x} and left lines in y: {left_line_y}")
        print(f"right lines in x: {right_line_x} and right lines in y: {right_line_y}")

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
            #print(f"poly left: {poly_left}")
            #get the start and the end
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
            
            #Create a function that match with all the detected lines
            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))
            #print(f"poly right: {poly_right}")
            #get the start and the end
            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))

            #save points
            define_lines=[[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]]
            #print(f"define lines: {define_lines}")
            #Add hough lines in the image images
            #img_colour_with_lines = frame.copy()      

            
                

            #Add both lines
            #img_colour_with_Definelines = frame.copy()
            for line in define_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_lane_lines, (x1, y1), (x2, y2), (255,0,0), 12)
            
            
            #cv2.namedWindow('LINES', cv2.WINDOW_NORMAL)
            cv2.imshow('LINES', img_lane_lines)
            #cv2.resizeWindow('LINES', 1000,900)
            
            for line in hough_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img_colour_with_lines, (x1, y1), (x2, y2), (255,0,0), 5)

            cv2.imshow('Hough lines', img_colour_with_lines)

            #cv2.namedWindow('HOUGH', cv2.WINDOW_NORMAL)
            cv2.imshow('Left and right lines', img_colour_with_left_and_right_lines)
            #cv2.resizeWindow('HOUGH', 1000,900)

    cv2.waitKey(0)


# Ask the user to enter the path to input images
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_images", help="Path to input images")
args = parser.parse_args()

# Get the list of image files and sort it alphabetically
list_with_name_of_images = sorted(os.listdir(args.path_to_images))
# Loop through each input image
for im in list_with_name_of_images:
    # Build path and image name
    path_and_im= args.path_to_images+im
    # Get the start time
    start_time = time.process_time()
    # Run the workflow to each input image
    pipeline(path_and_im)
    # Print the name of image being processed and compute FPS
    print(f"Processing image:{path_and_im}",
          f"\tCPU execution time:{1/(time.process_time()-start_time):0.4f} FPS")
    # If the user presses the key 'q',
    # the program finishes
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        print("\nProgram interrupted by the user - bye mate!")
        break


# Test pipeline
#img_name = "dataset/G0073340.JPG"
#pipeline(img_name)



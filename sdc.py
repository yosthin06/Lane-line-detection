"""
sdc.py

Description: Leer una m¿imagen y detecta lineas 

Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
First created: Monday 27 october, 2022
Usage:

"""

# Import required libraries
import numpy as np
import cv2
import argparse
import os
import time

# Import user-defined libraries
import sdc_library 

def pipeline(img_name):
    
    # 1.- Read image
    img_colour = sdc_library.input_image(img_name)
   
    #2.- Convert from BGR to RGB then from RGB to greyscale
    grey = sdc_library.greyscale_img(img_colour)

    # 3.- Apply Gaussian smoothing
    kernel_size = (9, 9)
    blur_grey = sdc_library.smoothed_img(grey, kernel_size)
    
    # 4-. Apply Canny edge detector
    low_threshold = 70
    high_threshold = 100
    edges = sdc_library.canny_img(blur_grey, low_threshold, high_threshold)
    
    # 5.- Get a region of interest using the just created polygon
    # Define a Region-of-Interest. Change the below vertices according
    # to input image resolution
    p1, p2, p3, p4, p5, p6, p7, p8 = (3, 438), (3, 296), (325, 237), (610, 237), (910, 320), (910, 438),(590,290),(340,290)

    # create a vertices array that will be used for the roi
    vertices = np.array([[p1, p2, p3, p4, p5, p6, p7, p8]], dtype=np.int32)
    roi_image = sdc_library.region_of_interest(edges, vertices)
    
    # 6.- Apply Hoguh transform for lane lines detection
    rho = 2                             # distance resolution in pixels of the Hough grid
    theta = np.pi/180                   # angular resolution in radians of the Hough grid
    threshold = 100                      # minimum number of votes (intersections in Hough grid)
    min_line_len = 10                    # minimum number of pixels making up a line
    max_line_gap = 30                   # maximum gap in pixels between connectable line segments
    hough_lines = sdc_library.hough(img_colour, roi_image, rho, theta, threshold, min_line_len, max_line_gap)

    # 7-. Get the inlier left and right Hough lines
    left_line_x, left_line_y, right_line_x, right_line_y = sdc_library.left_and_right_lines(hough_lines, img_colour)
    
    # 8-. Draw a single line for the left and right lane lines
    defined_lane_lines = sdc_library.lane_lines(left_line_x, left_line_y, right_line_x, right_line_y, img_colour)      

if __name__ == "__main__":
    
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
        




# Steven Kundert, Jiaxing Liu, Prakriti Pandey
# CMPS 5443 - Data Mining - Griffin
# Assignment 5 - Occupied Parking Space Project
# Part 1 - Image Processing
# 10/2/17
# This program reads an image of a parking lot and a json file associated with
# that image (converted from an xml file using lot_xml_to_json.py). The program
# loops through each parking space in the json file, extracting the min and max
# x and y values of the four points that make up each space. Then the program
# uses those four values to create a cropped image for each space and a 
# histogram of that image, and saves both to files.

import numpy as np
import cv2 as cv        # Must pip install opencv-python before running 
from matplotlib import pyplot as plt
from json import loads

# Reads the json file and stores each space's data
# Function provided in Data Mining resources folder
def load_spaces(definition_file):
    f = open(definition_file,'r')
    spaces = loads(f.read())
    return spaces

# For a given space, find the points that make up the its quadrilateral
# Function provided in Data Mining resources folder
def extract_points(space):
    points = []
    for i in range(4):
        x = int(space['contour']['point'][i]['x'])
        y = int(space['contour']['point'][i]['y'])
        points.append((x,y))
    return points

# Creates a list of the file names to be written to. The base name is currently
# hardcoded, but should be adaptable later
def generate_filenames(space):
    basefilenames = []
    filename = '2012-12-16_12_05_07_'
    filename = filename + space['id']
    spacename = 'spaces/' + filename + '.jpg'
    histname = 'histograms/' + filename + '.csv'
    basefilenames.append(spacename)
    basefilenames.append(histname)
    return basefilenames

# Read in the image and json file associated with it. These are currently
# hardcoded, but should be adaptable later
img = cv.imread('parking1b.jpg')
spaces = load_spaces('2012-12-16_12_05_07.json')

for space in spaces:
    # Get the file names
    filenames = generate_filenames(space)
    # Get the points of the space
    points = extract_points(space)
    
    # Find the min and max of x and y, which appear in a definite order 
    # in the json file
    ymin = points[0][1]
    xmax = points[1][0]
    ymax = points[2][1]
    xmin = points[3][0]
    # Crop the space and save it to a jpg file
    crop_img = img[ymin:ymax,xmin:xmax]
    plt.imsave(filenames[0],crop_img)
    # Create a histogram for the space's image and save to a csv file
    hsv = cv.cvtColor(crop_img,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    np.savetxt(filenames[1], hist, delimiter=',')

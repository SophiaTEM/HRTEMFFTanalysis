# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:16:49 2021

@author: Sophia
"""


import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from matplotlib.widgets import Cursor
from skimage import io
import hyperspy.api as hs
import math

def generateMap(data):
    map_d_pxl = np.zeros([4, np.shape(data)[0], np.shape(data[0][0])[2]])
    map_angle = np.zeros([3, np.shape(data)[0], np.shape(data[0][0])[2]])
    for i in range (np.shape(data)[0]):
            for j in range (np.shape(data[0][0])[1]):
                map_d_pxl[0, i, j] = data[i][0][0][j][0]
                map_d_pxl[1, i, j] = data[i][0][1][j][0]
                map_d_pxl[2, i, j] = data[i][0][2][j][0]
                map_d_pxl[3, i, j] = data[i][0][3][j][0]
                map_angle[0, i, j] = data[i][1][0][j][0]
                map_angle[1, i, j] = data[i][1][1][j][0]
                map_angle[2, i, j] = data[i][1][2][j][0]
    return map_d_pxl, map_angle

def cleanUp(map_d_pxl, map_angle, peakPos, FFTwindowSize):
    ##### CLEAN-UP OF THE MAPS
    map_d_pxl_1A = np.zeros([3, np.shape(map_angle)[0], np.shape(map_angle)[1]])
    map_angle_1A = np.zeros([3, np.shape(map_angle)[0], np.shape(map_angle)[1]])
    d1 = np.sqrt(np.square(FFTwindowSize/2 - peakPos[0][0][1]) + np.square(FFTwindowSize/2 - peakPos[0][0][0]))
    d2 = np.sqrt(np.square(FFTwindowSize/2 - peakPos[1][0][1]) + np.square(FFTwindowSize/2 - peakPos[1][0][0]))
    d3 = np.sqrt(np.square(FFTwindowSize/2 - peakPos[2][0][1]) + np.square(FFTwindowSize/2 - peakPos[2][0][0]))
    a1 = abs(math.atan((FFTwindowSize/2 - peakPos[0][0][1])/(FFTwindowSize/2 - peakPos[0][0][0]))*180/math.pi)
    a2 = abs(math.atan((FFTwindowSize/2 - peakPos[0][0][1])/(FFTwindowSize/2 - peakPos[0][0][0]))*180/math.pi)
    a3 = abs(math.atan((FFTwindowSize/2 - peakPos[0][0][1])/(FFTwindowSize/2 - peakPos[0][0][0]))*180/math.pi)
    for i in range (np.shape(map_angle)[1]):
        for j in range (np.shape(map_angle)[1]):
            if map_d_pxl[0, i, j] < d1*0.75 or map_d_pxl[0, i, j] > d1*1.25 or map_angle[0, i, j] < a1-2.5 or map_angle[0, i, j] > a1+2.5:
                map_d_pxl_1A[0, i, j] = 0
                map_angle_1A[0, i, j] = 0
            else:
                map_d_pxl_1A[0, i, j] = map_d_pxl[0, i, j]
                map_angle_1A[0, i, j] = map_angle[0, i, j]
            if map_d_pxl[1, i, j] < d2*0.75 or map_d_pxl[1, i, j] > d2*1.25 or map_angle[1, i, j] < a2-2.5 or map_angle[1, i, j] > a2+2.5:
                map_d_pxl_1A[1, i, j] = 0
                map_angle_1A[1, i, j] = 0
            else:
                map_d_pxl_1A[1, i, j] = map_d_pxl[1, i, j]
                map_angle_1A[1, i, j] = map_angle[1, i, j]
            if map_d_pxl[2, i, j] < d3*0.75 or map_d_pxl[2, i, j] > d3*1.25 or map_angle[2, i, j] < a3-2.5 or map_angle[2, i, j] > a3+2.5:
                map_d_pxl_1A[2, i, j] = 0
                map_angle_1A[2, i, j] = 0
            else:
                map_d_pxl_1A[2, i, j] = map_d_pxl[2, i, j]
                map_angle_1A[2, i, j] = map_angle[2, i, j]
    return map_d_pxl_1A, map_angle_1A
    

def averageMaps(map_d_pxl_1A, map_angle_1A):
    map_d_av = np.zeros([4, 3, np.shape(map_d_pxl_1A)[0], np.shape(map_d_pxl_1A)[1]])
    map_angle_av = np.zeros([4, 2, np.shape(map_d_pxl_1A)[0], np.shape(map_d_pxl_1A)[1]])
    ##### AVERAGE 5x5 Pixels  
    for i in range (0, np.shape(map_d_pxl_1A)[0], 5):
        for j in range (0, np.shape(map_d_pxl_1A)[1], 5):
            map_d_av[0, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[0, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[0, i:i+5, j:j+5])
            map_d_av[1, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[1, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[1, i:i+5, j:j+5])
            map_d_av[2, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[2, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[2, i:i+5, j:j+5])    
            map_angle_av[0, i:i+5, j:j+5] = sum(sum(map_angle_1A[0, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[0, i:i+5, j:j+5])
            map_angle_av[1, i:i+5, j:j+5] = sum(sum(map_angle_1A[1, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[1, i:i+5, j:j+5])
            map_angle_av[2, i:i+5, j:j+5] = sum(sum(map_angle_1A[2, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[2, i:i+5, j:j+5])
    ##### AVERAGE 10x10 Pixels
    for i in range (0, np.shape(map_d_pxl_1A)[0], 10):
        for j in range (0, np.shape(map_d_pxl_1A)[1], 10):
            map_d_av[1, 0, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[0, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[0, i:i+10, j:j+10])
            map_d_av[1, 1, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[1, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[1, i:i+10, j:j+10])
            map_d_av[1, 2, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[2, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[2, i:i+10, j:j+10])
            map_angle_av[1, 0, i:i+10, j:j+10] = sum(sum(map_angle_1A[0, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[0, i:i+10, j:j+10])
            map_angle_av[1, 1, i:i+10, j:j+10] = sum(sum(map_angle_1A[1, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[1, i:i+10, j:j+10])
            map_angle_av[1, 2, i:i+10, j:j+10] = sum(sum(map_angle_1A[2, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[2, i:i+10, j:j+10])
            
    ##### AVERAGE 15x15 Pixels
    for i in range (0, np.shape(map_d_pxl_1A)[0], 15):
        for j in range (0, np.shape(map_d_pxl_1A)[1], 15):
            map_d_av[2, 0, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[0, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[0, i:i+15, j:j+15])
            map_d_av[2, 1, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[1, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[1, i:i+15, j:j+15])
            map_d_av[2, 2, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[2, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[2, i:i+15, j:j+15])
            map_angle_av[2, 0, i:i+15, j:j+15] = sum(sum(map_angle_1A[0, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[0, i:i+15, j:j+15])
            map_angle_av[2, 1, i:i+15, j:j+15] = sum(sum(map_angle_1A[1, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[1, i:i+15, j:j+15])
            map_angle_av[2, 2, i:i+15, j:j+15] = sum(sum(map_angle_1A[2, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[2, i:i+15, j:j+15])
    ##### AVERAGE 20x20 Pixels
    for i in range (0, np.shape(map_d_pxl_1A)[1], 20):
        for j in range (0, np.shape(map_d_pxl_1A)[1], 20):
            map_d_av[3, 0, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[0, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[0, i:i+20, j:j+20])
            map_d_av[3, 1, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[1, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[1, i:i+20, j:j+20])
            map_d_av[3, 2, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[2, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[2, i:i+20, j:j+20])
            map_angle_av[3, 0, i:i+20, j:j+20] = sum(sum(map_angle_1A[0, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[0, i:i+20, j:j+20])
            map_angle_av[3, 1, i:i+20, j:j+20] = sum(sum(map_angle_1A[1, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[1, i:i+20, j:j+20])
            map_angle_av[3, 2, i:i+20, j:j+20] = sum(sum(map_angle_1A[2, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[2, i:i+20, j:j+20])
    return map_d_av, map_angle_av

def map_scaled(map_d_av, pixelSize):    
    map_d_Angst = np.zeros([3, np.shape(map_d_av)[0], np.shape(map_d_av)[1]])
    for i in range (np.shape(map_d_av)[0]):
        for j in range (np.shape(map_d_av)[1]):
            map_d_Angst[0, i, j] = (1/(map_d_av[0, i, j]*pixelSize))*10
            map_d_Angst[1, i, j] = (1/(map_d_av[1, i, j]*pixelSize))*10
            map_d_Angst[2, i, j] = (1/(map_d_av[2, i, j]*pixelSize))*10
    return map_d_Angst

def makeMask(im, threshold):
    im_norm = im/np.mean(im)
    ret,mask = cv2.threshold(im_norm,threshold,1,cv2.THRESH_BINARY_INV)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, se1)
    return mask3

def final_maps(map_d_Angst, map_angle_av, mask3):
    


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

FFTwindowSize = 128
pixelSize = 0.033344114597
im = io.imread('Aligned 20201120 1138 620 kx Ceta_binned_aligned_slice1crop.tif')
data = np.load('data.npy')

def RefID(data, pixelSize, FFTwindowSize):
    def select_mvc(data):
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(data)
        plt.ylabel('Force (V)')
        plt.xlabel('Energy-loss (eV)')
        cursor = Cursor(ax, useblit=True, color='k', linewidth=1)
        zoom_ok = False
        print('\nZoom or pan to view, \npress spacebar when ready to click:\n')
        while not zoom_ok:
            zoom_ok = plt.waitforbuttonpress()
        print('Click once to select MVC force: ')
        val = plt.ginput(1)
        print('Selected values: ', val)
        plt.close()
        return val
    dict0 = {'size': int(np.shape(data)[0]), 'name':'Axis0', 'units':'nm', 'scale':pixelSize, 'offset':1}
    s1 = hs.signals.BaseSignal(data, axes=[dict0, dict0])
    i = int(np.shape(s1)[0]*0.4)
    j = int(np.shape(s1)[1]*0.4)
    dataset = s1
    dataset_crop = dataset.isig[i:(i+FFTwindowSize), j:(j+FFTwindowSize)]
    FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
    ax = plt.imshow(FF)
    peakPos = []
    A1 = select_mvc(FF)    
    peakPos.append(A1)
    A2 = select_mvc(FF)
    peakPos.append(A2)
    A3 = select_mvc(FF)
    peakPos.append(A3)
    return peakPos

peakPos = RefID(im, pixelSize, FFTwindowSize)

def generateMap(parts):
    if parts != 0:
        data = np.load('data.npy')
        map_d_pxl = np.zeros([4, np.shape(data)[0], np.shape(data[0][0])[2]])
        map_angle = np.zeros([3, np.shape(data)[0], np.shape(data[0][0])[2]])
        for i in range (np.shape(data)[0]):
                for j in range (np.shape(data[0][0])[1]):
                    map_d_pxl[0, i, j] = data[i][0][0][j][4]
                    map_d_pxl[1, i, j] = data[i][0][1][j][4]
                    map_d_pxl[2, i, j] = data[i][0][2][j][4]
                    map_d_pxl[3, i, j] = data[i][0][3][j][4]
                    map_angle[0, i, j] = data[i][1][0][j][4]
                    map_angle[1, i, j] = data[i][1][1][j][4]
                    map_angle[2, i, j] = data[i][1][2][j][4]
    else:
        data = np.load('data_0.npy')
        map_d_pxlA = np.zeros([parts, 4, np.shape(data)[0], np.shape(data[0][0])[2]])
        map_angleA = np.zeros([parts, 3, np.shape(data)[0], np.shape(data[0][0])[2]])
        for k in range(parts):
            data = np.load('data_' + str(k) + 'npy')
            for i in range (np.shape(data)[0]):
                for j in range (np.shape(data[0][0])[1]):
                    map_d_pxlA[k, 0, i, j] = data[i][0][0][j][4]
                    map_d_pxlA[k, 1, i, j] = data[i][0][1][j][4]
                    map_d_pxlA[k, 2, i, j] = data[i][0][2][j][4]
                    map_d_pxlA[k, 3, i, j] = data[i][0][3][j][4]
                    map_angleA[k, 0, i, j] = data[i][1][0][j][4]
                    map_angleA[k, 1, i, j] = data[i][1][1][j][4]
                    map_angleA[k, 2, i, j] = data[i][1][2][j][4] 
        maps_d_pxlB = map_d_pxlB[0, :, :, :]
        l = 1
        for i in range(0, np.square(parts)):     
            while k < parts-1:
                for j in range(np.shape(map_d_pxlA)[1]):
                    map_d_pxlB[l, j, :, :] = np.concatenate((map_d_pxlB[j, :, :], map_d_pxlA[k, j, :, :], axis=0)
                k = k + 1
        maps_d_pxlC = map_d_pxlB[0, :, :, :]
        for i in range(1, parts):
            for j in range(np.shape(map_d_pxlA)[1]):
                maps_d_pxlC[j, :, :] = np.concatenate((map_d_pxlC[j, :, :], map_d_pxlB[k, j, :, :]))
    
    
    return map_d_pxl, map_angle

def cleanUp(map_d_pxl, map_angle, peakPos, FFTwindowSize, pixelSize):
    ##### CLEAN-UP OF THE MAPS
    map_d_pxl_1A = np.zeros([3, np.shape(map_angle)[1], np.shape(map_angle)[1]])
    map_angle_1A = np.zeros([3, np.shape(map_angle)[1], np.shape(map_angle)[1]])
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
    
    f1 = open(path +'map_d_pxl_1A.pckl', 'wb')
    pickle.dump(map_d_pxl_1A[0], f1)
    f1.close()  
      
    map_d_av = np.zeros([4, 3, np.shape(map_angle)[1], np.shape(map_angle)[1]])
    map_angle_av = np.zeros([4, 2, np.shape(map_angle)[1], np.shape(map_angle)[1]])
    ##### AVERAGE 5x5 Pixels  
    for i in range (0, np.shape(map_angle)[1], 5):
        for j in range (0, np.shape(map_angle)[1], 5):
            map_d_av[0, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[0, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[0, i:i+5, j:j+5])
            map_d_av[1, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[1, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[1, i:i+5, j:j+5])
            map_d_av[2, i:i+5, j:j+5] = sum(sum(map_d_pxl_1A[2, i:i+5, j:j+5]))/np.count_nonzero(map_d_pxl_1A[2, i:i+5, j:j+5])    
            map_angle_av[0, i:i+5, j:j+5] = sum(sum(map_angle_1A[0, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[0, i:i+5, j:j+5])
            map_angle_av[1, i:i+5, j:j+5] = sum(sum(map_angle_1A[1, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[1, i:i+5, j:j+5])
            map_angle_av[2, i:i+5, j:j+5] = sum(sum(map_angle_1A[2, i:i+5, j:j+5]))/np.count_nonzero(map_angle_1A[2, i:i+5, j:j+5])
    ##### AVERAGE 10x10 Pixels
    for i in range (0, np.shape(map_angle)[1], 10):
        for j in range (0, np.shape(map_angle)[1], 10):
            map_d_av[1, 0, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[0, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[0, i:i+10, j:j+10])
            map_d_av[1, 1, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[1, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[1, i:i+10, j:j+10])
            map_d_av[1, 2, i:i+10, j:j+10] = sum(sum(map_d_pxl_1A[2, i:i+10, j:j+10]))/np.count_nonzero(map_d_pxl_1A[2, i:i+10, j:j+10])
            map_angle_av[1, 0, i:i+10, j:j+10] = sum(sum(map_angle_1A[0, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[0, i:i+10, j:j+10])
            map_angle_av[1, 1, i:i+10, j:j+10] = sum(sum(map_angle_1A[1, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[1, i:i+10, j:j+10])
            map_angle_av[1, 2, i:i+10, j:j+10] = sum(sum(map_angle_1A[2, i:i+10, j:j+10]))/np.count_nonzero(map_angle_1A[2, i:i+10, j:j+10])
            
    ##### AVERAGE 15x15 Pixels
    for i in range (0, np.shape(map_angle)[1], 15):
        for j in range (0, np.shape(map_angle)[1], 15):
            map_d_av[2, 0, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[0, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[0, i:i+15, j:j+15])
            map_d_av[2, 1, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[1, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[1, i:i+15, j:j+15])
            map_d_av[2, 2, i:i+15, j:j+15] = sum(sum(map_d_pxl_1A[2, i:i+15, j:j+15]))/np.count_nonzero(map_d_pxl_1A[2, i:i+15, j:j+15])
            map_angle_av[2, 0, i:i+15, j:j+15] = sum(sum(map_angle_1A[0, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[0, i:i+15, j:j+15])
            map_angle_av[2, 1, i:i+15, j:j+15] = sum(sum(map_angle_1A[1, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[1, i:i+15, j:j+15])
            map_angle_av[2, 2, i:i+15, j:j+15] = sum(sum(map_angle_1A[2, i:i+15, j:j+15]))/np.count_nonzero(map_angle_1A[2, i:i+15, j:j+15])
    ##### AVERAGE 20x20 Pixels
    for i in range (0, np.shape(map_angle)[1], 20):
        for j in range (0, np.shape(map_angle)[1], 20):
            map_d_av[3, 0, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[0, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[0, i:i+20, j:j+20])
            map_d_av[3, 1, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[1, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[1, i:i+20, j:j+20])
            map_d_av[3, 2, i:i+20, j:j+20] = sum(sum(map_d_pxl_1A[2, i:i+20, j:j+20]))/np.count_nonzero(map_d_pxl_1A[2, i:i+20, j:j+20])
            map_angle_av[3, 0, i:i+20, j:j+20] = sum(sum(map_angle_1A[0, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[0, i:i+20, j:j+20])
            map_angle_av[3, 1, i:i+20, j:j+20] = sum(sum(map_angle_1A[1, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[1, i:i+20, j:j+20])
            map_angle_av[3, 2, i:i+20, j:j+20] = sum(sum(map_angle_1A[2, i:i+20, j:j+20]))/np.count_nonzero(map_angle_1A[2, i:i+20, j:j+20])
    
    f1 = open(path +'map_d_pxl_av.pckl', 'wb')
    pickle.dump(map_d_av, f1)
    f1.close()
    f2 = open(path +'map_angle_av.pckl', 'wb')
    pickle.dump(map_angle_av, f1)
    f2.close()
    ##### SCALE in Angstrom
    map_d_Angst = np.zeros([3, np.shape(map_angle)[1], np.shape(map_angle)[1]])
    for i in range (np.shape(map_angle)[1]):
        for j in range (np.shape(map_angle)[1]):
            map_d_Angst[0, i, j] = (1/(map_d_av3[0, i, j]*pixelSize))*10
            map_d_Angst[1, i, j] = (1/(map_d_av3[1, i, j]*pixelSize))*10
            map_d_Angst[2, i, j] = (1/(map_d_av3[2, i, j]*pixelSize))*10
    
    f1 = open(path +'map_d_Angst.pckl', 'wb')
    pickle.dump(map_d_Angst, f1)
    f1.close()
    return map_d_av, map_d_Angst, map_angle_av

def makeMask(im, threshold):
    im_norm = im/np.mean(im)
    ret,mask = cv2.threshold(im_norm,threshold,1,cv2.THRESH_BINARY_INV)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
    mask3 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, se1)
    mask4 = np.zeros([np.shape(map_d_av)[1], np.shape(map_d_av)[2]])
    return mask4

im_woBG = im*mask4

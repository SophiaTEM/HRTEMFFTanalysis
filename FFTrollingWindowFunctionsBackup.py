# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:59:44 2021

@author: Sophia
"""
from scipy import optimize
import scipy.io
import multiprocessing as mp
import numpy as np
import h5py
import time
import hyperspy.api as hs
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from matplotlib.widgets import Cursor

def RefID(data, RefNumber, pixelSize, FFTwindowSize):
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
    size = np.shape(data)[0]
    dict0 = {'size': int(np.shape(data)[0]), 'name':'Axis0', 'units':'nm', 'scale':pixelSize, 'offset':1}
    s1 = hs.signals.BaseSignal(data, axes=[dict0, dict0])
    i = int(np.shape(s1)[0]*0.4)
    j = int(np.shape(s1)[1]*0.4)
    dataset = s1
    dataset_crop = dataset.isig[i:(i+FFTwindowSize), j:(j+FFTwindowSize)]
    FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
    ax = plt.imshow(FF)
    peakPos = []
    if RefNumber == 1:
        A1 = select_mvc(FF)    
        peakPos.append(A1)
    elif RefNumber == 2:
        A1 = select_mvc(FF)    
        peakPos.append(A1)
        A2 = select_mvc(FF)
        peakPos.append(A2)
    elif RefNumber == 3:
        A1 = select_mvc(FF)    
        peakPos.append(A1)
        A2 = select_mvc(FF)
        peakPos.append(A2)
        A3 = select_mvc(FF)
        peakPos.append(A3)
    elif RefNumber == 4:
        A1 = select_mvc(FF)    
        peakPos.append(A1)
        A2 = select_mvc(FF)
        peakPos.append(A2)
        A3 = select_mvc(FF)
        peakPos.append(A3)
        A4 = select_mvc(FF)
        peakPos.append(A4)
    else:
        'too many reflections selected, maximal number is 4'
    return peakPos



def FFTrollingWindow(i, data, peakPos, pixelSize, FFTwindowSize):
    
    def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        (x, y) = xdata_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
        return g.ravel()
    
    dict0 = {'size': np.shape(data)[0], 'name':'Axis0', 'units':'nm', 'scale':pixelSize, 'offset':1}
    dataset = hs.signals.BaseSignal(data, axes=[dict0, dict0])
    #start_j = 0
    #end_j = np.shape(dataset)[0]-FFTwindowSize
    start_j=10
    end_j=15
    #end_j = np.shape(dataset)[0]
    map_d_pxl =  np.zeros([np.shape(peakPos)[0]+1, np.shape(dataset)[0], np.shape(dataset)[1]])
    map_angle =  np.zeros([np.shape(peakPos)[0], np.shape(dataset)[0], np.shape(dataset)[1]])
    vector_original = np.zeros([3])
    vector_original[0] = 0
    vector_original[1] = FFTwindowSize/2
    vector_original[2] = np.sqrt(np.square(vector_original[0])+np.square(vector_original[1]))
    error3 = 0
    error1 = 0
    error2 = 0
    error4 = 0
    z = np.zeros(end_j)
    
    for j in range(start_j, end_j):
        map_d_pxl[np.shape(peakPos)[0]+1, i, j] = dataset.data[i, j]
        dataset_crop = dataset.isig[i:(i+FFTwindowSize), j:(j+FFTwindowSize)]
        FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
        if np.shape(peakPos)[0] == 1: 
        # AREA 1
            area1_x1 = int(peakPos[0][0][1]-FFTwindowSize/16)
            area1_x2 = int(peakPos[0][0][1]+FFTwindowSize/16)
            area1_y1 = int(peakPos[0][0][0]-FFTwindowSize/16)
            area1_y2 = int(peakPos[0][0][0]+FFTwindowSize/16)
            area1 = FF.data[area1_x1:area1_x2, area1_y1:area1_y2]
            #p = np.asarray(area1).astype('float')
            w, h = np.shape(area1)
            x, y = np.mgrid[0:h, 0:w]
            #xy = (x, y)
            initial_guess = (1, 6, 6, 1, 1, 0, 1)
            if np.max(area1) < 2:
                map_d_pxl[0, j] = 0
                map_angle[0, j] = 0
            else:
                try:
                    #area1a = FF_filtered[80:90, 13:23]   
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess)
                    data1_fitted = twoD_Gaussian((x, y), *popt1)
                    #plt.figure(1)
                    #plt.imshow(data1_fitted.reshape(15, 15))
                    reflection1 = np.zeros([2])
                    reflection1[0] = area1_x1 + popt1[1]
                    reflection1[1] = area1_y1 + popt1[2]
                    vector1 = np.zeros([2])
                    vector1[0] = FFTwindowSize/2 - reflection1[0]
                    vector1[1] = FFTwindowSize/2 - reflection1[1]
                    d1_pxl = np.sqrt(np.square(vector1[0]) + np.square(vector1[1]))
                    angle1 = math.acos((vector_original[0]*vector1[0] + vector_original[1]*vector1[1])/(d1_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[0, j] = d1_pxl
                    map_angle[0, j] = angle1
                    del popt1, vector1, reflection1
                except RuntimeError:
                    map_d_pxl[0, j] = 0
                    map_angle[0, j] = 0
                    error1 = error1 + 1
        elif np.shape(peakPos)[0] == 2: 
        # AREA 1
            area1_x1 = int(peakPos[0][0][1]-FFTwindowSize/16)
            area1_x2 = int(peakPos[0][0][1]+FFTwindowSize/16)
            area1_y1 = int(peakPos[0][0][0]-FFTwindowSize/16)
            area1_y2 = int(peakPos[0][0][0]+FFTwindowSize/16)
            area1 = FF.data[area1_x1:area1_x2, area1_y1:area1_y2]
            #p = np.asarray(area1).astype('float')
            w, h = np.shape(area1)
            x, y = np.mgrid[0:h, 0:w]
            #xy = (x, y)
            initial_guess = (1, 6, 6, 1, 1, 0, 1)
            if np.max(area1) < 2:
                map_d_pxl[0, j] = 0
                map_angle[0, j] = 0
            else:
                try:
                    #area1a = FF_filtered[80:90, 13:23]   
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess)
                    data1_fitted = twoD_Gaussian((x, y), *popt1)
                    #plt.figure(1)
                    #plt.imshow(data1_fitted.reshape(15, 15))
                    reflection1 = np.zeros([2])
                    reflection1[0] = area1_x1 + popt1[1]
                    reflection1[1] = area1_y1 + popt1[2]
                    vector1 = np.zeros([2])
                    vector1[0] = FFTwindowSize/2 - reflection1[0]
                    vector1[1] = FFTwindowSize/2 - reflection1[1]
                    d1_pxl = np.sqrt(np.square(vector1[0]) + np.square(vector1[1]))
                    angle1 = math.acos((vector_original[0]*vector1[0] + vector_original[1]*vector1[1])/(d1_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[0, j] = d1_pxl
                    map_angle[0, j] = angle1
                    del popt1, vector1, reflection1
                except RuntimeError:
                    map_d_pxl[0, j] = 0
                    map_angle[0, j] = 0
                    error1 = error1 + 1
            # AREA 2
            area2_x1=int(peakPos[1][0][1]-FFTwindowSize/16)
            area2_x2=int(peakPos[1][0][1]+FFTwindowSize/16)
            area2_y1=int(peakPos[1][0][0]-FFTwindowSize/16)
            area2_y2=int(peakPos[1][0][0]+FFTwindowSize/16)
            area2 = FF.data[area2_x1:area2_x2, area2_y1:area2_y2]
            if np.max(area2) < 0.0:
                map_d_pxl[1, j] = 0
                map_angle[1, j] = 0
            else:
                try:
                    popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess)
                    data2_fitted = twoD_Gaussian((x, y), *popt2)
                    #plt.figure(2)
                    #plt.imshow(data2_fitted.reshape(15, 15))
                    reflection2 = np.zeros([2])
                    reflection2[0] = area2_x1 + popt2[1]
                    reflection2[1] = area2_y1 + popt2[2]
                    vector2 = np.zeros([2])
                    vector2[0] = FFTwindowSize/2 - reflection2[0]
                    vector2[1] = FFTwindowSize/2 - reflection2[1]
                    d2_pxl = np.sqrt(np.square(vector2[0]) + np.square(vector2[1]))
                    angle2 = math.acos((vector_original[0]*vector2[0] + vector_original[1]*vector2[1])/(d2_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[1, j] = d2_pxl
                    map_angle[1, j] = angle2
                    del popt2, vector2, reflection2
                except RuntimeError:
                    map_d_pxl[1, j] = 0
                    map_angle[1, j] = 0
                    error2 = error2 + 1
        elif np.shape(peakPos)[0] == 3: 
        # AREA 1
            area1_x1 = int(peakPos[0][0][1]-FFTwindowSize/16)
            area1_x2 = int(peakPos[0][0][1]+FFTwindowSize/16)
            area1_y1 = int(peakPos[0][0][0]-FFTwindowSize/16)
            area1_y2 = int(peakPos[0][0][0]+FFTwindowSize/16)
            area1 = FF.data[area1_x1:area1_x2, area1_y1:area1_y2]
            #p = np.asarray(area1).astype('float')
            w, h = np.shape(area1)
            x, y = np.mgrid[0:h, 0:w]
            #xy = (x, y)
            initial_guess = (1, 6, 6, 1, 1, 0, 1)
            if np.max(area1) < 2:
                map_d_pxl[0, j] = 0
                map_angle[0, j] = 0
            else:
                try:
                    #area1a = FF_filtered[80:90, 13:23]   
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess)
                    data1_fitted = twoD_Gaussian((x, y), *popt1)
                    #plt.figure(1)
                    #plt.imshow(data1_fitted.reshape(15, 15))
                    reflection1 = np.zeros([2])
                    reflection1[0] = area1_x1 + popt1[1]
                    reflection1[1] = area1_y1 + popt1[2]
                    vector1 = np.zeros([2])
                    vector1[0] = FFTwindowSize/2 - reflection1[0]
                    vector1[1] = FFTwindowSize/2 - reflection1[1]
                    d1_pxl = np.sqrt(np.square(vector1[0]) + np.square(vector1[1]))
                    angle1 = math.acos((vector_original[0]*vector1[0] + vector_original[1]*vector1[1])/(d1_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[0, j] = d1_pxl
                    map_angle[0, j] = angle1
                    del popt1, vector1, reflection1
                except RuntimeError:
                    map_d_pxl[0, j] = 0
                    map_angle[0, j] = 0
                    error1 = error1 + 1
            # AREA 2
            area2_x1=int(peakPos[1][0][1]-FFTwindowSize/16)
            area2_x2=int(peakPos[1][0][1]+FFTwindowSize/16)
            area2_y1=int(peakPos[1][0][0]-FFTwindowSize/16)
            area2_y2=int(peakPos[1][0][0]+FFTwindowSize/16)
            area2 = FF.data[area2_x1:area2_x2, area2_y1:area2_y2]
            if np.max(area2) < 0.0:
                map_d_pxl[1, j] = 0
                map_angle[1, j] = 0
            else:
                try:
                    popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess)
                    data2_fitted = twoD_Gaussian((x, y), *popt2)
                    #plt.figure(2)
                    #plt.imshow(data2_fitted.reshape(15, 15))
                    reflection2 = np.zeros([2])
                    reflection2[0] = area2_x1 + popt2[1]
                    reflection2[1] = area2_y1 + popt2[2]
                    vector2 = np.zeros([2])
                    vector2[0] = FFTwindowSize/2 - reflection2[0]
                    vector2[1] = FFTwindowSize/2 - reflection2[1]
                    d2_pxl = np.sqrt(np.square(vector2[0]) + np.square(vector2[1]))
                    angle2 = math.acos((vector_original[0]*vector2[0] + vector_original[1]*vector2[1])/(d2_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[1, j] = d2_pxl
                    map_angle[1, j] = angle2
                    del popt2, vector2, reflection2
                except RuntimeError:
                    map_d_pxl[1, j] = 0
                    map_angle[1, j] = 0
                    error2 = error2 + 1
            # AREA 3
            area3_x1 = int(peakPos[2][0][1]-FFTwindowSize/16)
            area3_x2 = int(peakPos[2][0][1]+FFTwindowSize/16)
            area3_y1 = int(peakPos[2][0][0]-FFTwindowSize/16)
            area3_y2 = int(peakPos[2][0][0]+FFTwindowSize/16)
            area3 = FF.data[area3_x1:area3_x2, area3_y1:area3_y2]
            if np.max(area3) < 0.0:
                popt3 = np.zeros([7])
                map_d_pxl[2, j] = 0
                map_angle[2, j] = 0
            else:
                try:
                    popt3, pcov3 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area3), p0=initial_guess)
                    data3_fitted = twoD_Gaussian((x, y), *popt3)
                    #plt.figure(3)
                    #plt.imshow(data3_fitted.reshape(15, 15))
                    reflection3 = np.zeros([2])
                    reflection3[0] = area3_x1 + popt3[1]
                    reflection3[1] = area3_y1 + popt3[2]
                    vector3 = np.zeros([2])
                    vector3[0] = FFTwindowSize/2 - reflection3[0]
                    vector3[1] = FFTwindowSize/2 - reflection3[1]
                    d3_pxl = np.sqrt(np.square(vector3[0]) + np.square(vector3[1]))
                    angle3 = math.acos((vector_original[0]*vector3[0] + vector_original[1]*vector3[1])/(d3_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[2, j] = d3_pxl
                    map_angle[2, j] = angle3
                    del popt3, vector3, reflection3
                except RuntimeError:
                    map_d_pxl[2, j] = 0
                    map_angle[2, j] = 0
                    error3 = error3 + 1
        elif np.shape(peakPos)[0] == 4: 
        # AREA 1
            area1_x1 = int(peakPos[0][0][1]-FFTwindowSize/16)
            area1_x2 = int(peakPos[0][0][1]+FFTwindowSize/16)
            area1_y1 = int(peakPos[0][0][0]-FFTwindowSize/16)
            area1_y2 = int(peakPos[0][0][0]+FFTwindowSize/16)
            area1 = FF.data[area1_x1:area1_x2, area1_y1:area1_y2]
            #p = np.asarray(area1).astype('float')
            w, h = np.shape(area1)
            x, y = np.mgrid[0:h, 0:w]
            #xy = (x, y)
            initial_guess = (1, 6, 6, 1, 1, 0, 1)
            if np.max(area1) < 2:
                map_d_pxl[0, j] = 0
                map_angle[0, j] = 0
            else:
                try:
                    #area1a = FF_filtered[80:90, 13:23]
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess)
                    data1_fitted = twoD_Gaussian((x, y), *popt1)
                    #plt.figure(1)
                    #plt.imshow(data1_fitted.reshape(15, 15))
                    reflection1 = np.zeros([2])
                    reflection1[0] = area1_x1 + popt1[1]
                    reflection1[1] = area1_y1 + popt1[2]
                    vector1 = np.zeros([2])
                    vector1[0] = FFTwindowSize/2 - reflection1[0]
                    vector1[1] = FFTwindowSize/2 - reflection1[1]
                    d1_pxl = np.sqrt(np.square(vector1[0]) + np.square(vector1[1]))
                    angle1 = math.acos((vector_original[0]*vector1[0] + vector_original[1]*vector1[1])/(d1_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[0, j] = d1_pxl
                    map_angle[0, j] = angle1
                    del popt1, vector1, reflection1
                except RuntimeError:
                    map_d_pxl[0, j] = 0
                    map_angle[0, j] = 0
                    error1 = error1 + 1
            # AREA 2
            area2_x1=int(peakPos[1][0][1]-FFTwindowSize/16)
            area2_x2=int(peakPos[1][0][1]+FFTwindowSize/16)
            area2_y1=int(peakPos[1][0][0]-FFTwindowSize/16)
            area2_y2=int(peakPos[1][0][0]+FFTwindowSize/16)
            area2 = FF.data[area2_x1:area2_x2, area2_y1:area2_y2]
            if np.max(area2) < 0.0:
                map_d_pxl[1, j] = 0
                map_angle[1, j] = 0
            else:
                try:
                    popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess)
                    data2_fitted = twoD_Gaussian((x, y), *popt2)
                    #plt.figure(2)
                    #plt.imshow(data2_fitted.reshape(15, 15))
                    reflection2 = np.zeros([2])
                    reflection2[0] = area2_x1 + popt2[1]
                    reflection2[1] = area2_y1 + popt2[2]
                    vector2 = np.zeros([2])
                    vector2[0] = FFTwindowSize/2 - reflection2[0]
                    vector2[1] = FFTwindowSize/2 - reflection2[1]
                    d2_pxl = np.sqrt(np.square(vector2[0]) + np.square(vector2[1]))
                    angle2 = math.acos((vector_original[0]*vector2[0] + vector_original[1]*vector2[1])/(d2_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[1, j] = d2_pxl
                    map_angle[1, j] = angle2
                    del popt2, vector2, reflection2
                except RuntimeError:
                    map_d_pxl[1, j] = 0
                    map_angle[1, j] = 0
                    error2 = error2 + 1
            # AREA 3
            area3_x1 = int(peakPos[2][0][1]-FFTwindowSize/16)
            area3_x2 = int(peakPos[2][0][1]+FFTwindowSize/16)
            area3_y1 = int(peakPos[2][0][0]-FFTwindowSize/16)
            area3_y2 = int(peakPos[2][0][0]+FFTwindowSize/16)
            area3 = FF.data[area3_x1:area3_x2, area3_y1:area3_y2]
            if np.max(area3) < 0.0:
                popt3 = np.zeros([7])
                map_d_pxl[2, j] = 0
                map_angle[2, j] = 0
            else:
                try:
                    popt3, pcov3 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area3), p0=initial_guess)                
                    data3_fitted = twoD_Gaussian((x, y), *popt3)
                    #plt.figure(3)
                    #plt.imshow(data3_fitted.reshape(15, 15))
                    reflection3 = np.zeros([2])
                    reflection3[0] = area3_x1 + popt3[1]
                    reflection3[1] = area3_y1 + popt3[2]
                    vector3 = np.zeros([2])
                    vector3[0] = FFTwindowSize/2 - reflection3[0]
                    vector3[1] = FFTwindowSize/2 - reflection3[1]
                    d3_pxl = np.sqrt(np.square(vector3[0]) + np.square(vector3[1]))
                    angle3 = math.acos((vector_original[0]*vector3[0] + vector_original[1]*vector3[1])/(d3_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[2, j] = d3_pxl
                    map_angle[2, j] = angle3
                    del popt3, vector3, reflection3
                except RuntimeError:
                    map_d_pxl[2, j] = 0
                    map_angle[2, j] = 0
                    error3 = error3 + 1
            # AREA 4
            area4_x1 = int(peakPos[3][0][1]-FFTwindowSize/16)
            area4_x2 = int(peakPos[3][0][1]+FFTwindowSize/16)
            area4_y1 = int(peakPos[3][0][0]-FFTwindowSize/16)
            area4_y2 = int(peakPos[3][0][0]+FFTwindowSize/16)
            area4 = FF.data[area4_x1:area4_x2, area4_y1:area4_y2]
            if np.max(area4) < 0.0:
                popt4 = np.zeros([7])
                map_d_pxl[3, j] = 0
                map_angle[3, j] = 0
            else:
                try:
                    popt4, pcov4 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area4), p0=initial_guess)                
                    data4_fitted = twoD_Gaussian((x, y), *popt4)
                    #plt.figure(3)
                    #plt.imshow(data3_fitted.reshape(15, 15))
                    reflection4 = np.zeros([2])
                    reflection4[0] = area4_x1 + popt4[1]
                    reflection4[1] = area4_y1 + popt4[2]
                    vector4 = np.zeros([2])
                    vector4[0] = FFTwindowSize/2 - reflection4[0]
                    vector4[1] = FFTwindowSize/2 - reflection4[1]
                    d4_pxl = np.sqrt(np.square(vector4[0]) + np.square(vector4[1]))
                    angle4 = math.acos((vector_original[0]*vector4[0] + vector_original[1]*vector4[1])/(d4_pxl*vector_original[2]))*180/math.pi
                    map_d_pxl[3, j] = d4_pxl
                    map_angle[3, j] = angle4
                    del popt4, vector4, reflection4
                except RuntimeError:
                    map_d_pxl[3, j] = 0
                    map_angle[3, j] = 0
                    error4 = error4 + 1
    return map_d_pxl, map_angle

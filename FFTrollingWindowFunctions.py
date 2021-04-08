# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:26:09 2020

@author: Sophia Betzler
"""

#%%
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
import warnings
from skimage import io
from scipy.optimize import OptimizeWarning
from matplotlib.widgets import Cursor
warnings.simplefilter("error", OptimizeWarning)

##### Data
im = io.imread('Aligned 20201120 1138 620 kx Ceta_binned_aligned_slice1crop.tif')

##### set variables
FFTwindowSize=128
pixelSize = 0.033344114597
coreNumber = 15

# Determine position of the reflections in the image
start = time.time()

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
    dict1 = {'size': int(np.shape(data)[1]), 'name':'Axis0', 'units':'nm', 'scale':pixelSize, 'offset':1}
    s1 = hs.signals.BaseSignal(data, axes=[dict0, dict1])
    j = int(np.shape(s1)[0]*0.4)
    i = int(np.shape(s1)[1]*0.4)
    dataset = s1
    dataset_crop = dataset.isig[i:(i+FFTwindowSize), j:(j+FFTwindowSize)]
    FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
    ax = plt.imshow(FF)
    peakPos = []
    A1 = select_mvc(FF)    
    peakPos.append(np.flip(A1))
    A2 = select_mvc(FF)
    peakPos.append(np.flip(A2))
    A3 = select_mvc(FF)
    peakPos.append(np.flip(A3))
    return peakPos

##############################################################################
############################ Copy this part into your script
##############################################################################
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

def fftanalysis2(i):
    start_j = 0
    end_j = np.shape(dataset)[0]-FFTwindowSize
    #area1 
    A1x1 = int(peakPos[0][0][1]) - 7
    A1x2 = int(peakPos[0][0][1]) + 8
    A1y1 = int(peakPos[0][0][0]) - 7
    A1y2 = int(peakPos[0][0][0]) + 8
    #area2
    A2x1 = int(peakPos[1][0][1]) - 7
    A2x2 = int(peakPos[1][0][1]) + 8
    A2y1 = int(peakPos[1][0][0]) - 7
    A2y2 = int(peakPos[1][0][0]) + 7
    #area3
    A3x1 = int(peakPos[2][0][1]) - 7
    A3x2 = int(peakPos[2][0][1]) + 8
    A3y1 = int(peakPos[2][0][0]) - 7
    A3y2 = int(peakPos[2][0][0]) + 8
    #end_j = np.shape(dataset)[0]
    map_d_pxl =  np.zeros([4, np.shape(dataset)[0]-FFTwindowSize, 1])
    map_angle =  np.zeros([3, np.shape(dataset)[0]-FFTwindowSize, 1])
    error3 = 0
    error1 = 0
    error2 = 0
    for j in range(start_j, end_j):
        map_d_pxl[3, j, 0] = dataset.data[j, i]
        dataset_crop = dataset.isig[i:(i+FFTwindowSize), j:(j+FFTwindowSize)]
        FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
    # AREA 1
        area1 = FF.data[A1x1:A1x2, A1y1:A1y2]
        area1 = np.clip(area1, 0.85*np.max(area1), np.max(area1))
        w, h = np.shape(area1)
        x, y = np.mgrid[0:h, 0:w]
        initial_guess = (1, 6, 6, 3, 3, 0, 8)
        bounds = ([-np.inf, 1, 1, -np.inf, -np.inf, -np.inf, -np.inf],[np.inf, 14, 14, np.inf, np.inf, np.inf, np.inf])
        if np.max(area1) < 5:
            map_d_pxl[0, j, 0] = 0
            map_angle[0, j, 0] = 0 
        else:
            try:
                try:
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess, bounds=bounds, method='dogbox')
                except OptimizeWarning:
                    area1 = FF.data[A1x1:A1x2, A1y1:A1y2]
                    popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess, bounds=bounds, method='dogbox')
                #area1a = FF_filtered[80:90, 13:23]   
                #popt1, pcov1 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area1), p0=initial_guess)
                #data1_fitted = twoD_Gaussian((x, y), *popt1)
                #plt.figure(1)
                #plt.imshow(data1_fitted.reshape(15, 15))
            #def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
                d1_pxl = np.sqrt(np.square(FFTwindowSize/2 - (A1x1 + popt1[1])) + np.square(FFTwindowSize/2 - (A1y1 + popt1[2])))
                angle1 = abs(math.atan((FFTwindowSize/2 - (A1y1 + popt1[2]))/(FFTwindowSize/2 - (A1x1 + popt1[1])))*180/math.pi)
                map_d_pxl[0, j, 0] = d1_pxl
                map_angle[0, j, 0] = angle1
                del popt1, pcov1
            except RuntimeError:
                map_d_pxl[0, j, 0] = 0
                map_angle[0, j, 0] = 0
                error1 = error1 + 1
    # AREA 2
        area2 = FF.data[A2x1:A2x2, A2y1:A2y2]
        area2 = np.clip(area2, 0.85*np.max(area2), np.max(area2))
        w, h = np.shape(area2)
        x, y = np.mgrid[0:h, 0:w]
        initial_guess = (1, 6, 6, 3, 3, 0, 8)
        bounds = ([-np.inf, 1, 1, -np.inf, -np.inf, -np.inf, -np.inf],[np.inf, 14, 14, np.inf, np.inf, np.inf, np.inf])
        if np.max(area2) < 5:
            map_d_pxl[1, j, 0] = 0
            map_angle[1, j, 0] = 0
        else:
            try:
                try:
                    popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess, bounds=bounds, method='dogbox')
                except OptimizeWarning:
                    area2 = FF.data[A2x1:A2x2, A2y1:A2y2]
                    popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess, bounds=bounds, method='dogbox')
                #popt2, pcov2 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area2), p0=initial_guess)
                #data2_fitted = twoD_Gaussian((x, y), *popt2)
                #plt.figure(2)
                #plt.imshow(data2_fitted.reshape(15, 15))
                d2_pxl = np.sqrt(np.square(FFTwindowSize/2 - (A2x1 + popt2[1])) + np.square(FFTwindowSize/2 - (A2y1 + popt2[2])))
                angle2 = abs(math.atan((FFTwindowSize/2 - (A2y1 + popt2[2]))/(FFTwindowSize/2 - (A2x1 + popt2[1])))*180/math.pi)
                map_d_pxl[1, j, 0] = d2_pxl
                map_angle[1, j, 0] = angle2
                del popt2, pcov2
            except RuntimeError:
                map_d_pxl[1, j, 0] = 0
                map_angle[1, j, 0] = 0
                error2 = error2 + 1
    # AREA 3
        area3 = FF.data[A3x1:A3x2, A3y1:A3y2]
        area3 = np.clip(area3, 0.85*np.max(area3), np.max(area3))
        w, h = np.shape(area3)
        x, y = np.mgrid[0:h, 0:w]
        initial_guess = (1, 6, 6, 3, 3, 0, 8)
        bounds = ([-np.inf, 1, 1, -np.inf, -np.inf, -np.inf, -np.inf],[np.inf, 14, 14, np.inf, np.inf, np.inf, np.inf])
        if np.max(area3) < 5:
            popt3 = np.zeros([7])
            map_d_pxl[2, j, 0] = 0
            map_angle[2, j, 0] = 0
        else:
            try:
                try:
                    popt3, pcov3 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area3), p0=initial_guess, bounds=bounds, method='dogbox')
                except OptimizeWarning:
                    area3 = FF.data[A3x1:A3x2, A3y1:A3y2]
                    popt3, pcov3 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area3), p0=initial_guess, bounds=bounds, method='dogbox')
                #popt3, pcov3 = optimize.curve_fit(twoD_Gaussian, (x, y), np.ravel(area3), p0=initial_guess)                
                #data3_fitted = twoD_Gaussian((x, y), *popt3)
                #plt.figure(3)
                #plt.imshow(data3_fitted.reshape(15, 15))
                d3_pxl = np.sqrt(np.square(FFTwindowSize/2 - (A3x1 + popt3[1])) + np.square(FFTwindowSize/2 - (A3y1 + popt3[2])))
                angle3 = abs(math.atan((FFTwindowSize/2 - (A3y1 + popt3[2]))/(FFTwindowSize/2 - (A3x1 + popt3[1])))*180/math.pi)
                map_d_pxl[2, j, 0] = d3_pxl
                map_angle[2, j, 0] = angle3        
                del popt3, pcov3
            except RuntimeError:
                map_d_pxl[2, j, 0] = 0
                map_angle[2, j, 0] = 0
                error3 = error3 + 1
    return map_d_pxl, map_angle

def testfunc(start, end):
    array = list(range(start, end))
    return array

peakPos = RefID(im, pixelSize, FFTwindowSize)
data2 = np.pad(im, ((int(FFTwindowSize/2), int(FFTwindowSize/2)),(int(FFTwindowSize/2), int(FFTwindowSize/2))))

dict0 = {'name':'Axis0', 'size': np.shape(data2)[0],  'units':'nm', 'scale': pixelSize, 'offset':1}
dict1 = {'name':'Axis1', 'size': np.shape(data2)[1],  'units':'nm', 'scale': pixelSize, 'offset':1}

dataset = hs.signals.BaseSignal(data2, axes=[dict0, dict1])

if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[1]-FFTwindowSize
    array = list(range(start_i, end_i))
    p = mp.Pool(coreNumber)
    data = p.map(fftanalysis2, array)
    #print(data)
    np.save('data' + '.npy', data)
        
end = time.time()
time_passed = (end-start)*60
print(time_passed)


# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 00:30:24 2020

@author: Sophia
"""


# -*- coding: utf-8 -*-
"""
Created on Sun May 17 13:26:09 2020

@author: HaimeiGroup
"""
from scipy import optimize
import multiprocessing as mp
import numpy as np
import h5py
import time
import hyperspy.api as hs
import matplotlib.pyplot as plt
from matplotlib import cm
import math
#%%
##### Gaussian function to fit to the peaks
#define model function and pass independant variables x and y as a list
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
#%%
##### Import of the matlab file
f1 = h5py.File('D:\\Sophia\\H2inPd\\StrainAnalysis\\data02_movie_1701_denoised_32_32_6.mat')
# list(f.keys())
data1 = f1['sigRecon13']
dict0 = {'size': 1536, 'name':'Axis0', 'units':'nm', 'scale':0.05440, 'offset':1}
s1 = hs.signals.BaseSignal(data1[94], axes=[dict0, dict0])
[X, Y] = np.shape(data1[219])
  #%% 
##### 
dataset = s1
start = time.time()
def fftanalysis2(i):
    start_j = 0
    end_j = np.shape(dataset)[0]-128
    #end_j = np.shape(dataset)[0]
    map_d_pxl =  np.zeros([4, np.shape(dataset)[1], np.shape(dataset)[1]])
    map_angle =  np.zeros([3, np.shape(dataset)[1], np.shape(dataset)[1]])
    vector_original = np.zeros([3])
    vector_original[0] = 0
    vector_original[1] = 64
    vector_original[2] = np.sqrt(np.square(vector_original[0])+np.square(vector_original[1]))
    error3 = 0
    error1 = 0
    error2 = 0
    z = np.zeros(end_j)
    for j in range(start_j, end_j):
        map_d_pxl[3, i, j] = dataset.data[i, j]
        dataset_crop = dataset.isig[i:(i+128), j:(j+128)]
        FF = np.log(dataset_crop.fft(shift=True, apodization=True).amplitude)
    # AREA 1
        area1 = FF.data[77:92, 11:26]
        p = np.asarray(area1).astype('float')
        w, h = np.shape(area1)
        x, y = np.mgrid[0:h, 0:w]
        xy = (x,y)
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
            #def twoD_Gaussian((x,y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
                reflection1 = np.zeros([2])
                reflection1[0] = 77 + popt1[1]
                reflection1[1] = 11 + popt1[2]
                vector1 = np.zeros([2])
                vector1[0] = 64 - reflection1[0]
                vector1[1] = 64 - reflection1[1]
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
        area2 = FF.data[89:104, 45:60]
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
                reflection2[0] = 89 + popt2[1]
                reflection2[1] = 45 + popt2[2]
                vector2 = np.zeros([2])
                vector2[0] = 64 - reflection2[0]
                vector2[1] = 64 - reflection2[1]
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
        area3 = FF.data[43:58, 24:39]
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
                reflection3[0] = 43 + popt3[1]
                reflection3[1] = 24 + popt3[2]
                vector3 = np.zeros([2])
                vector3[0] = 64 - reflection3[0]
                vector3[1] = 64 - reflection3[1]
                d3_pxl = np.sqrt(np.square(vector3[0]) + np.square(vector3[1]))
                angle3 = math.acos((vector_original[0]*vector3[0] + vector_original[1]*vector3[1])/(d3_pxl*vector_original[2]))*180/math.pi
                map_d_pxl[2, j] = d3_pxl
                map_angle[2, j] = angle3        
                del popt3, vector3, reflection3
            except RuntimeError:
                map_d_pxl[2, j] = 0
                map_angle[2, j] = 0
                error3 = error3 + 1
    return map_d_pxl, map_angle

def testfunc(start, end):
    array = list(range(start, end))
    return array

if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[0]-128
    array = list(range(start_i, end_i))
    p = mp.Pool(12)
    data = p.map(fftanalysis2, array)
    #print(data)

end = time.time()
time_passed = end-start
print(time_passed)
# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. 

Make sure to copy the functions into the main file, otherwise the multiprocessing (parallelization of the code) won't work and make sure to use the same variable names as described here. First a high resolution image must be imported. It might be useful to bin the image to reduce the required memory. Make sure provide the correct pixel size, determine the window size used to calcualte the FFT pattern, currently the method uses three reflections for the analysis. Alter the values of these variables in the preamble of the script.
```
dataset = import(filename)
pixelsize = 0.3 #nm
FFTwindowSize = 128
coreNumber = 15
```
First the reflections used for the analysis must be selected in the image using the following function: 
```
peakPos = RefID(dataset, FFTwindowSize)
```
Subsequently the rolling window algorithm determines spatial variations of the selected reflections within the crystal. Parallelization is used to reduce the time it takes for the process to finish (the number of selected CPU cores is given be coreNumber). Copy the function <b>fftanalysis2(i)</b> into your Python script and add the following lines to run the parallelized code:
```
dict0 = {'name':'Axis0', 'size': np.shape(data2)[0],  'units':'nm', 'scale': pixelSize, 'offset':1}
dict1 = {'name':'Axis1', 'size': np.shape(data2)[1],  'units':'nm', 'scale': pixelSize, 'offset':1}
dataset = hs.signals.BaseSignal(data2, axes=[dict0, dict1])
if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[1]-FFTwindowSize
    #start_i = 0
    #end_i = 200
    array = list(range(start_i, end_i))
    p = mp.Pool(coreNumber)
    data = p.map(fftanalysis2, array)
    #print(data)
np.save('data' + '.npy', data)
```

The maps are created from the data file obtained from the parallel processed data analysis:
```
data = np.load('data_0_128.npy', allow_pickle=True)
pixelSizeImage = 0.0333442
FFTwindowSize = 128
pixelSize = (1/pixelSizeImage/FFTwindowSize)
if np.shape(data)[0] > np.shape(data[0][0])[1]:
    map_d_pxl = np.zeros([4, np.shape(data)[0], np.shape(data)[1]])
    map_angle = np.zeros([4, np.shape(data)[0], np.shape(data)[1]])
    for i in range (np.shape(data)[0]):
        for j in range (np.shape(data[0][0])[1]):
            map_d_pxl[0, i, j] = data[i][0][0][j][0]
            map_d_pxl[1, i, j] = data[i][0][1][j][0]
            map_d_pxl[2, i, j] = data[i][0][2][j][0]
            map_d_pxl[3, i, j] = data[i][0][3][j][0]
            map_angle[0, i, j] = data[i][1][0][j][0]
            map_angle[1, i, j] = data[i][1][1][j][0]
            map_angle[2, i, j] = data[i][1][2][j][0]
else:
    map_d_pxl = np.zeros([4, np.shape(data[0][0])[1], np.shape(data[0][0])[1]])
    map_angle = np.zeros([4, np.shape(data[0][0])[1], np.shape(data[0][0])[1]])
    for i in range (np.shape(data)[0]):
        for j in range (np.shape(data[0][0])[1]):
            map_d_pxl[0, i, j] = data[i][0][0][j][0]
            map_d_pxl[1, i, j] = data[i][0][1][j][0]
            map_d_pxl[2, i, j] = data[i][0][2][j][0]
            map_d_pxl[3, i, j] = data[i][0][3][j][0]
            map_angle[0, i, j] = data[i][1][0][j][0]
            map_angle[1, i, j] = data[i][1][1][j][0]
            map_angle[2, i, j] = data[i][1][2][j][0]
```
Falsely identified reflections are removed based on the angle (a variation of 5 degree is considered acceptable). In addition, segmentation is used to remove the amorphous background using a threshold value which must be determined for each dataset individually. Finally, several pixel values (5x5, 10x10, 15x15 and 20x20) are averaged to obtain the spatial variation of the d-spacing for the different reflection using the following function and a created mask separating the nanoparticle from the amorphous background. 

```
mapgeneration(map_d_pxl, map_angle, mask):

```



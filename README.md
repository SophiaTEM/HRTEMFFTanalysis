# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. It is easiest to copy the script <b> FFTrollingwindowFunctions.py</b> and run it with your own image. Make sure to copy the functions into the main file, otherwise the multiprocessing (parallelization of the code) won't work and make sure to use the same variable names as described here. The steps in the script are described in detail in the following.

<b> step 1: </b>
First a high resolution image must be imported. It might be useful to bin the image to reduce the required memory and process time. Make sure provide the correct pixel size <b>pixelSize</b>, determine the window size used to calcualte the FFT pattern (<b>FFTwindowSize</b>). Since the script is parallelized the number of CPU cores used must be defined (<b>coreNumber</b>).
```
##### Data
im = io.imread('Aligned 20201120 1138 620 kx Ceta_binned_aligned_slice1crop.tif')
FFTwindowSize = 128
pixelSize = 0.033344114597
coreNumber = 15
```
<b> step 2: </b>
The following function can be used to select the reflections used for the analysis: 
```
peakPos = RefID(dataset, FFTwindowSize)
```
<b> step 3: </b>
Subsequently the rolling window algorithm determines spatial variations of the selected reflections within the crystal (<b>fftanalysis2</b>) using a two-dimensional Gaussian fit to determine the peak position (<b>twoD_Gaussian</b>). The parallelization of the code is done using the multiprocessing package using the following code:
```
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
```
<b> step 4: </b>
The final maps are created from the data file obtained from the parallel processed data analysis:
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
<b> step 5: </b>
Falsely identified reflections are removed based on the angle (a variation of 5 degree is considered acceptable). In addition, segmentation is used to remove the amorphous background using a threshold value which must be determined for each dataset individually. Finally, several pixel values (5x5, 10x10, 15x15 and 20x20) are averaged to obtain the spatial variation of the d-spacing for the different reflection using the following function and a created mask separating the nanoparticle from the amorphous background. 

```
mapgeneration(map_d_pxl, map_angle, mask)

```



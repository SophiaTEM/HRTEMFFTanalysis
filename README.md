# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. It is easiest to copy the script <b> FFTrollingwindowFunctions.py</b> and run it with your own image. Make sure to copy the functions into the main file, otherwise the multiprocessing (parallelization of the code) won't work and make sure to use the same variable names as described here. The steps in the script are described in detail in the following.

<b> step 1: </b>
First a high resolution image must be imported. It might be useful to bin the image to reduce the required memory and process time. Make sure provide the correct pixel size <b>pixelSize</b>, determine the window size used to calcualte the FFT pattern (<b>FFTwindowSize</b>). Since the script is parallelized the number of CPU cores used must be defined (<b>coreNumber</b>).
```
##### Data
im = io.imread('filename.tif')
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
The final maps are created from the data file obtained from the parallel processed data analysis using the functions stored in <b>CreateMaps.py</b>:
```
data = np.load('data_0_128.npy', allow_pickle=True)

generateMap(data)
```
<b> step 5: </b>
Falsely identified reflections are removed based on the angle (a variation of 5 degree is considered acceptable). In addition, segmentation is used to remove the amorphous background using a threshold value which must be determined for each dataset individually. Several pixel values (5x5, 10x10, 15x15 and 20x20) are averaged to obtain the spatial variation of the d-spacing for the different reflection and the correct pixel size is applied for the maps.

```
[map_d_pxl_1A, map_angle_1A] = cleanUp(map_d_pxl, map_angle, peakPos, FFTwindowSize)
[map_d_av, map_angle_1A] = averageMaps(map_d_pxl_1A, map_angle_1A)
map_d_Angst = map_scaled(map_d_av, pixelSize) 
```
<b> step 6: </b>
A mask is created and used to separate the nanoparticle from the amorphous background.
```
```
  

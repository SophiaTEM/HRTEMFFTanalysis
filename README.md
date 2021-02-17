# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. 

Make sure to switch the working directory to the directory which contains the function file and the analysis file. This is essential to use the multiprocessing package and parallelize the code. 
First a high resolution image must be imported. It might be useful to bin the image to reduce the required memory. Make sure provide the correct pixel size, determine the window size used to calcualte the FFT pattern, currently the method uses three reflections for the analysi. 
```
dataset = import(filename)
pixelsize = 0.3 #nm
FFTwindowSize = 128
```
First the reflections used for the analysis must be selected in the image using the following function: 
```
import HRTEMFFTanalysis as FFw
peakPos = FFw.RefID(dataset, RefNumber, FFTwindowSize)
```
Subsequently the rolling window algorithm determines spatial variations of the selected reflections within the crystal. Parallelization is used to reduce the time it takes for the process to finish (the number of selected CPU cores is given be coreNumber):
```
def fftanalysis(i):
    'see file'
  
if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[0]
    array = list(range(start_i, end_i))
    func = partial(fftanalysis, dataset)
    p = mp.Pool(coreNumber)
    data = p.map(func, array)
```
The maps are created from the data file obtained from the previous lines:
```
map_d_pxl = np.zeros([4, np.shape(data)[0], np.shape(data[0][0])[2]])
map_angle = np.zeros([4, np.shape(data)[0], np.shape(data[0][0])[2]])
for i in range (np.shape(data)[0]):
    for j in range (np.shape(data[0][0])[1]):
        map_d_pxl[0, i, j] = data[i][0][0][j][4]
        map_d_pxl[1, i, j] = data[i][0][1][j][4]
        map_d_pxl[2, i, j] = data[i][0][2][j][4]
        map_d_pxl[3, i, j] = data[i][0][3][j][4]
        map_angle[0, i, j] = data[i][1][0][j][4]
        map_angle[1, i, j] = data[i][1][1][j][4]
        map_angle[2, i, j] = data[i][1][2][j][4]
```
Segmentation is used to remove the amorphous background. Falsely identified reflections are removed based on the angle (a variation of 5 degree is considered acceptable). In addition several pixel (5x5, 10x10, 15x15 and 20x20) are averaged to obtain the spatial variation of the d-spacing for the different reflection. 


To reduce the memory requirement of large datasets they can be split into smaller parts:
```

```

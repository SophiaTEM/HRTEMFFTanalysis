# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. 

Make sure to switch the working directory to the directory which contains the function file and the analysis file. This is essential to use the multiprocessing package and parallelize the code. 
First a high resolution image must be imported. It might be useful to bin the image to reduce the required memory. Make sure provide the correct pixel size, determine the window size used to calcualte the FFT pattern, the number of reflections used for the analysis (maximum 4). 
```
dataset = import(filename)
pixelsize = 0.3 #nm
FFTwindowSize
RefNumber = 3
```
First the reflections used for the analysis must be selected in the image using the following function: 
```
import HRTEMFFTanalysis as FFw
peakPos = FFw.RefID(dataset, RefNumber, FFTwindowSize)
```
Subsequently the rolling window algorithm determines spatial variations of the selected reflections within the crystal. Parallelization is used to reduce the time it takes for the process to finish (the number of selected CPU cores is given be coreNumber). 
```
if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[0]
    array = list(range(start_i, end_i))
    func = partial(FFw.FFTrollingWindow, dataset, peakPos, FFTwindowSize)
    p = mp.Pool(coreNumber)
    Data = p.map(func, array)
```
Segmentation is used to remove the amorphous background. Falsely identified reflections are removed based on the angle (a variation of xyz degree is considered acceptable). In addition 10x10 pixel are averaged to obtain the local d-spacing for the reflection. 
```

```

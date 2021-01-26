# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. A high resolution image must be imported first. It might be useful to bin the image to reduce the required memory. Make sure provide the correct pixel size. To run the code you will have to select three reflections in the FFT which are used for the analysis. 
```
dataset = import(filename)
pixelsize = 0.3 #nm
coreNumber = 12
FFTwindowSize
RefNumber = 3
peakPos = fftroll.RefID(dataset, RefNumber, pixelSize, FFTwindowSize)
```
Subsequently the rolling window algorithm determines spatial variations of the selected reflections within the crystal. Parallelization is used to reduce the time it takes for the process to finish. 
```
if __name__ == '__main__':
    start_i = 0
    end_i = np.shape(dataset)[0]
    array = list(range(start_i, end_i))
    func = partial(fftroll.FFTrollingWindow, dataset, peakPos, pixelSize, FFTwindowSize)
    p = mp.Pool(coreNumber)
    Data = p.map(func, array)
```


# HRTEMFFTanalysis
Tool to analyze the lattice spacing within HRTEM images using a rolling window algorithm to gain localized structural informations, e.g. lattice deformations. A high resolution image must be imported first. It might be useful to bin the image to reduce the required memory. Make sure provide the correct pixel size.
```
data = import(filename)
pixelsize = 0.3 #nm
```
To run the code you will have to select three reflections in the FFT which are used for the analysis. File 1 is used to process the data and generate the results. The code is parallized, make sure to select the appropriate number of cores. 
```

```
Finally, the results are plotted using a second analysis file. 
```
```

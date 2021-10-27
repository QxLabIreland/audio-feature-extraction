# Audio Feature Extraction
Audio feature extraction computes typical features used in audio research. The script automatically integrates features from frame-level to clip-level and stores the values in a csv file. 

## Feature List
The script computes the following features plus some audio quality metrics. For each feature, the actual values and the first order difference (FOD) are computed except where indicated.
Features are computed per frame and integrated with the MeanVar model i.e., for each feature, it computes the mean and variance of feature frames. Finally the values are stored in a csv file.
### Spectral Features
```
Spectral Centroid
Spectral Bandwidth
Spectral Roll Off
Mel Frequency Cepstral Coefficients
Logarithmic Mel Filter Bank Energies (no FOD)
Spectral Flux
Spectral Subband Centroid
Linear Predictive Coding Coefficients
```
### Loudness
```
Root Mean Square
Loudness ITU-R BS.1770-4 (no FOD)
Peak Envelope
```

### Tonalness
```
Tonal Power Ratio
Spectral Crest Factor
Spectral Flatness
Zero-Crossing-Rate
```
### Non-Intrusive Quality Metric (no FOD)
```
Speech-To-Reverberation Modulation Energy Ratio (SRMR)
ITU-T P.563
MOSNet
```
## Requirements
```
pip install librosa
pip install pyloudnorm
pip install pandas
pip install click
pip install python_speech_features
pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu]
pip install numpy
You need to install p.563 https://github.com/qin/p.563
```
## Usage
1) Upload WAV files in the folder `data/waveform`. 
2) Run `processing.py` specifying 3 parameters:
```
nfft - Number of DFT points for all the features
  Default value 1024
hop  - Number of overlapping points when computing the STFT
  Default value 256
 sr  - Sampling rate of the audio files
  Default value 48000 Hz
```
The output will be stored in `features.csv` where each cell represents a pair (filename, feature).
If you need to use a different parameter only for a subset of features you have to modify `method_args.csv`. The file is structured as follows. Each cell in the csv file identifies a pair [parameter, feature]. Just modify/insert a value in the specific cell. This value will be used for that particular feature instead of the value that you pass as argument when calling `processing.py`. Each feature might have specific parameters that are not shared with other features (e.g., for computing MFCC you need to specify the number of coefficients). If you want to modify these parameters you need to follow the same procedure. 

## Contributing
Please follow the instructions below if you want to add new features:
1) Add a function in `feature_computation.py` at the end of the file e.g., `new_feature(self, ...)`
2) Include the following parameters with default values `feat_integration=True` and `feature_name=None`
3) Add a new column in `method_args.csv` with the same function_name you assigned above (point 1) e.g.,  `new_feature`
4) Add a feature name at the cell [feature_name, function_name] specifying the name that will be assigned in `features.csv`. 

*This example explains the difference between feature name and function name. If you add `def zero_crossing_rate(...)`  then `zero_crossing_rate` is the function name. This will generate two features `mean_zcr` and `var_zcr` which represent features name. In `features.csv` you will find feature_name and not the function_name.

*If your feature is based on the STFT and you want to allow its usage at different NFFT points you should add these parameters: `nfft=None` and `hop_length=None`. Then you should add this instruction at the beginning of your function `n_fft, hop_length = self.__update_parameters(n_fft, hop_length)`. 

*If you want to integrate features from frame-level to clip-level you need to call the function `feature_integration(frames)` where `frames` is a 1D or 2D numpy array depending on your feature (e.g., MFCC would be a 2D numpy array since you have N coefficients for each frame)
## Feedback
Feel free to improve the code and provide feedback. 

import librosa
import pandas as pd
import numpy as np
import pyloudnorm as pyln
import python_speech_features as psf
from scipy.signal import find_peaks
import math
import speechmetrics
import subprocess
import configparser
from shutil import which

config = configparser.ConfigParser()
config.read('config/config.ini')

class FeatureExtraction():
    '''
    The FeatureExtraction class is used to compute audio features. 
    It requires a csv file indicating feature parameters (index) and feature names (header).
    Feature names in the csv file must be t he same as method names of this class i.e., each letter of both strings must match

    Attributes:
        csv_file (str): Path to the csv file that includes feature information

        n_fft (int): The number of frequency points of the DFT that is used in the computation of hand-crafted features.
    
        sr (int): The sampling rate of the input waveform
    
        win_length (int): The number of samples used for windowing the signal in time domain. If not specified n_fft value will be used
    
        hop_length (int): The amount of overlapping in samples. If not specified it will be equal to n_fft/2
    '''
    def __init__(self, func_args, n_fft, sr, win_length=None, hop_length=None, feature_values=None):
        self.__filename = None
        self.__clipname = None
        self.__func_args = func_args
        self.__feature_values = feature_values
        self.__n_fft = n_fft
        self.__sr = sr
        self.__win_length = n_fft if win_length is None else win_length
        self.__hop_length = n_fft//2 if hop_length is None else hop_length
        feature_list = pd.read_csv(func_args, index_col='args').columns.to_list()
        self.__methods = dict(zip(feature_list, feature_list))
        self.waveform = None
        self.srmr_func = speechmetrics.load(metrics='srmr', window=None)
        self.mosnet_func = speechmetrics.load(metrics='mosnet', window=None)

    def set_waveform(self, waveform):
        self.waveform = waveform
    
    def __update_parameters(self, n_fft, hop_length):
        '''
        Each function has local parameters so that you are allowed to change parameters in each function individually
        '''
        # update parameters
        if n_fft == None:
            n_fft = self.__n_fft
        if hop_length == None:
            hop_length = int(n_fft)//2
        return n_fft, hop_length
        
    def set_filename(self, filename):
        '''
        Path to the audio file. Some features require the file path instead of the waveform

        Args:
            filename (str): Path of the audio file
        '''
        self.__filename = filename
    
    def set_clipname(self, clipname):
        '''
        Name of the audio clip. Needed to store feature values into the dataframe

        Args:
            clipname (str): Unique name of the audio clip
        '''
        self.__clipname = clipname

    def set_transcript(self, transcript):
        '''
        Set the transcript of the corresponding audio clip that you want to test

        Args:
            transcript (str): Original transcript of the audio clip
        '''
        self.__transcript = transcript
    
    def feature_integration(self, feature_frame_level):
        '''
        Integrate frame-level features to clip-level using the MeanVar model.
        The MeanVar model assumes that each frame-level feature is a Gaussian distribution and that the mean and the variance represent each feature at clip-level
        '''
        return np.mean(feature_frame_level, axis=1), np.var(feature_frame_level, axis=1)

    def df_store(self, feature_name, feature):
        if type(feature) is tuple:
            feature = [el for arr in feature for el in arr]
            for name, value in zip(feature_name.split(';'), feature):
                self.__feature_values.loc[self.__clipname, name] = value
        elif type(feature) is list:
            for name, value in zip(feature_name.split(';'), feature):
                self.__feature_values.loc[self.__clipname, name] = value        
        else:
            self.__feature_values.loc[self.__clipname, feature_name] = feature 
    

    def call_function(self, func_name, **args):
        '''
        Call the method you want to use to compute a certain feature

        Args:
            func_name (str): Method name
        
            **args: A list of arguments that you want to pass to a specific method

        Returns:
            The output of the corresponding method that is called
        '''
        return getattr(self, self.__methods[func_name])(**args)

    
    def zero_crossing_rate(self, win_length=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        win_length, hop_length = self.__update_parameters(win_length, hop_length)
        
        # compute zcr
        out = librosa.feature.zero_crossing_rate(self.waveform, frame_length=int(win_length), hop_length=int(hop_length))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out                
    
    def zero_crossing_rate_fo(self, win_length=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # compute first order difference
        out = np.diff(self.zero_crossing_rate(win_length, hop_length, feat_integration=False))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out  

    def spectral_flatness(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)
        
        # compute spectral flatness
        out = librosa.feature.spectral_flatness(self.waveform, n_fft=int(n_fft), hop_length=int(hop_length))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out

    def spectral_flatness_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # compute first order difference
        out = np.diff(self.spectral_flatness(n_fft, hop_length, feat_integration=False))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def spectral_bandwidth(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)
        
        # compute spectral bandwidth
        out = librosa.feature.spectral_bandwidth(self.waveform, sr=self.__sr, n_fft=int(n_fft), hop_length=int(hop_length))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out

    def spectral_bandwidth_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # compute first order difference
        out = np.diff(self.spectral_bandwidth(n_fft, hop_length, feat_integration=False))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out

    def spectral_rolloff(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)
        
        # compute spectral rolloff
        out = librosa.feature.spectral_rolloff(self.waveform, sr=self.__sr, n_fft=int(n_fft), hop_length=int(hop_length))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
     
    def spectral_rolloff_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # compute first order difference
        out = np.diff(self.spectral_rolloff(n_fft, hop_length, feat_integration=False))

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)
        
        return out
    
    def spectral_centroid(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute spectral centroid
        out = librosa.feature.spectral_centroid(self.waveform, sr=self.__sr, n_fft=int(n_fft), hop_length=int(hop_length))
        
         # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def spectral_centroid_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # compute first order difference
        out = np.diff(self.spectral_centroid(n_fft, hop_length, feat_integration=False))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out

    def mfcc(self, n_mfcc=26, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute mfcc
        out = librosa.feature.mfcc(self.waveform, sr=self.__sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)
        
        return out
    
    def mfcc_fo(self, n_mfcc=26, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
    
        # compute mfcc fo
        out = np.diff(self.mfcc(n_mfcc, n_fft, hop_length, feat_integration=False))
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def rms(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute mfcc
        out = librosa.feature.rms(self.waveform, frame_length=n_fft, hop_length=hop_length)

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def rms_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        out = self.rms(n_fft, hop_length, feat_integration=False)
        
        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out    
    
    def loudness(self, feature_name=None):

        # compute loudness according to ITU-R BS.1770-4
        meter = pyln.Meter(self.__sr)
        out = meter.integrated_loudness(self.waveform)

        return out

    def logfbank(self, n_filt=26, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):

        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)
        win_len = librosa.core.samples_to_time(n_fft, self.__sr)
        hop_length = librosa.core.samples_to_time(hop_length, self.__sr)

        # compute log filter banks
        out = psf.logfbank(self.waveform, self.__sr, win_len, hop_length, n_filt, n_fft)
        out = out.T

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def ssc(self, n_filt=10, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)
        win_len = librosa.core.samples_to_time(n_fft, self.__sr)
        hop_length = librosa.core.samples_to_time(hop_length, self.__sr)

        # compute log filter banks
        out = psf.ssc(self.waveform, self.__sr, win_len, hop_length, n_filt, n_fft)
        out = out.T

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out        
    
    def ssc_fo(self, n_filt=10, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
            
        # compute log filter banks
        out = np.diff(self.ssc(n_filt=n_filt, feat_integration=False))

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)

        return out
    
    def lpc(self, order=4, feat_integration=False, feature_name=None):
        out = librosa.core.lpc(self.waveform, order)

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)   
        out = list(out)

        return out 
    
    def spectral_flux(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute power spectrogram
        X = np.square(np.abs(librosa.stft(self.waveform, n_fft=n_fft, hop_length=hop_length)))
        
        # difference spectrum (set first diff to zero)
        X = np.c_[X[:, 0], X]
        
        # X = np.concatenate(X[:,0],X, axis=1)
        afDeltaX = np.diff(X, 1, axis=1)

        # flux
        out = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
        out = out.reshape(1, -1)

        # integrate features from frame-level to clip-level
        if feat_integration:
            out = self.feature_integration(out)   
        
        return out  

    def tonal_power_ratio(self, G_T=5e-4, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute power spectrogram
        X = np.square(np.abs(librosa.stft(self.waveform, n_fft=n_fft, hop_length=hop_length)))

        fSum = X.sum(axis=0)
        out = np.zeros(fSum.shape)

        for n in range(0, X.shape[1]):
            if fSum[n] < G_T:
                continue

            # find local maxima above the threshold
            afPeaks = find_peaks(X[:, n], height=G_T)

            if not afPeaks[0].size:
                continue

            # calculate ratio
            out[n] = X[afPeaks[0], n].sum() / fSum[n]
        
        out = out.reshape(1, -1)
        if feat_integration:
            out = self.feature_integration(out)      
        
        return out
    
    def tonal_power_ratio_fo(self, G_T=5e-4, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        out = np.diff(self.tonal_power_ratio(G_T, n_fft, hop_length, feat_integration=False))
        
        if feat_integration:
            out = self.feature_integration(out)
        
        return out
 
    def spectral_crest(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        # update parameters
        n_fft, hop_length = self.__update_parameters(n_fft, hop_length)

        # compute power spectrogram
        X = np.square(np.abs(librosa.stft(self.waveform, n_fft=n_fft, hop_length=hop_length)))

        norm = X.sum(axis=0)
        if X.shape[1] ==1:
            if norm == 0:
                norm = 1
        else:
            norm[norm == 0] = 1

        out = X.max(axis=0) / norm
        out = out.reshape(1, -1)
        if feat_integration:
            out = self.feature_integration(out)
         
        return out
    
    def spectral_crest_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        out = np.diff(self.spectral_crest(n_fft, hop_length, feat_integration=False))
        
        if feat_integration:
            out = self.feature_integration(out)

        return out 

    
    def peak_envelope(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        
        def ppm(x, filterbuf, alpha):
            # initialization
            ppmout = np.zeros(x.shape[0])

            alpha_AT = alpha[0]
            alpha_RT = alpha[1]

            for i in range(0, x.shape[0]):
                if filterbuf > x[i]:
                    # release state
                    ppmout[i] = (1 - alpha_RT) * filterbuf
                else:
                    # attack state
                    ppmout[i] = alpha_AT * x[i] + (1 - alpha_AT) * filterbuf

                filterbuf = ppmout[i]

            return (ppmout)
        
        x = self.waveform
        f_s = self.__sr

        iBlockLength, iHopLength = self.__update_parameters(n_fft, hop_length)
        
        # number of results
        iNumOfBlocks = math.ceil(x.size / iHopLength)

        # compute time stamps
        t = (np.arange(0, iNumOfBlocks) * iHopLength + (iBlockLength / 2)) / f_s

        alpha = 1 - np.array([np.exp(-2.2 / (f_s * 0.01)), np.exp(-2.2 / (f_s * 1.5))])

        # allocate memory
        vppm = np.zeros([2, iNumOfBlocks])
        v_tmp = np.zeros(iBlockLength)

        for n in range(0, iNumOfBlocks):

            i_start = n * iHopLength
            i_stop = np.min([x.size - 1, i_start + iBlockLength - 1])

            x_block = np.abs(x[np.arange(i_start, i_stop + 1)])

            # detect the maximum per block
            vppm[0, n] = np.max(x_block)

            # calculate the PPM value - take into account block overlaps
            # and discard concerns wrt efficiency
            v_tmp = ppm(x_block, v_tmp[iHopLength - 1], alpha)
            vppm[1, n] = np.max(v_tmp)

        # convert to dB
        epsilon = 1e-5  # -100dB

        vppm[vppm < epsilon] = epsilon
        vppm = 20 * np.log10(vppm)

        out = vppm[0].reshape(1, -1)

        if feat_integration:
            out = self.feature_integration(out)
        
        return out

    def peak_envelope_fo(self, n_fft=None, hop_length=None, feat_integration=True, feature_name=None):
        out = np.diff(self.peak_envelope(n_fft, hop_length, feat_integration=False))
        if feat_integration:
            out = self.feature_integration(out)

        return out
       
    def mosnet(self, feat_integration=False, feature_name=None):
        out = self.mosnet_func(self.__filename)['mosnet'][0][0]
        
        if feat_integration:
            out = self.feature_integration(out)
         
        return out
    
    def srmr(self, feat_integration=False, feature_name=None):
        out = self.srmr_func(self.__filename)['srmr']
        
        if feat_integration:
            out = self.feature_integration(out)
        
        return out      
    
    def cmd_exist(self, name):
        return which(name) is not None
    
    def p563(self, feat_integration=False, feature_name=None):
        
        cmd_name = 'p563'
        if self.cmd_exist(cmd_name):
            cmd = [cmd_name, self.__filename]
            result = subprocess.run(cmd, stdout=subprocess.PIPE) 
            stdout = result.stdout.decode('utf-8')
            if len(stdout) == 0:
                print("fdfd")
            out = [line for line in stdout.split('\t')][2]
            
            if feat_integration:
                out = self.feature_integration(out)
            
            return out
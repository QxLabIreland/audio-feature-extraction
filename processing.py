import configparser
import os
import pandas as pd
from feature_computation import FeatureExtraction
import librosa
import numpy as np
import click

config = configparser.ConfigParser()
config.read('config.ini')
config_name = 'AUDIO_FEATURE_EXTRACTION'

@click.command()
@click.option('--nfft', default=1024)
@click.option('--hop', default=256)
@click.option('--sr', default=48000)
def process_features(nfft, hop, sr):
    # extract path
    feature_path = config[config_name]['features']
    method_args_path = config[config_name]['method_args']
    waveform_path = config[config_name]['waveform_input']

    # extract csv files
    features = pd.read_csv(feature_path, index_col='filename')
    method_args = pd.read_csv(method_args_path, index_col='args').to_dict()
    method_args = {k1: {k:v for k, v in v1.items() if pd.notnull(v)} for k1, v1 in method_args.items()}

    # check if filenames already exist in the csv file where you save the feature values
    list_filename = [os.path.splitext(filename)[0] for filename in os.listdir(waveform_path)]
    current_index = features.index
    new_files_to_add = [filename for filename in list_filename if filename not in current_index.to_list()]
    if new_files_to_add is not None:
        new_index = pd.Index(new_files_to_add)
        features = features.reindex(current_index.append(new_index))
        features.index.name = 'filename'
    
    fe = FeatureExtraction(method_args_path, n_fft=nfft, sr=sr, hop_length=hop, feature_values=features)
    
    # loop over each audio file
    for clipname, _ in features.iterrows():
        # set filename
        print("Audio file: {}".format(clipname))
        fe.set_clipname(clipname)
        filename = os.path.join(waveform_path, clipname + '.wav')
        fe.set_filename(filename)

        # load waveform
        waveform, _ = librosa.load(filename, sr, mono=True)
        waveform = waveform - np.mean(waveform)
        fe.set_waveform(waveform)

        # loop over each feature
        for feature, args in method_args.items():

            # compute feature
            print("Computing: {}".format(feature))
            feature_value = fe.call_function(func_name=feature, **args)

            # if you passed a name for the feature (e.g., mean_zcr) store the feature value in the dataframe
            feature_name = args['feature_name']
            if args['feature_name'] is not None:
                fe.df_store(feature_name, feature_value)

    # store features
    features.to_csv(feature_path)
    
if __name__ == "__main__":
    process_features()
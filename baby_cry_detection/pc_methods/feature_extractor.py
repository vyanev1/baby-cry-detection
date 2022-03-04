# -*- coding: utf-8 -*-

import numpy as np
import logging
import timeit
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth,\
    rms #rms in librpsa 0.7.0, rmse in previous version

__all__ = [
    'FeatureExtractor'
]


class FeatureExtractor:

    ZERO_CROSSING_RATE = 'zero_crossing_rate'
    RMSE = 'rmse'
    MFCC = 'mfcc'
    SPECTRAL_CENTROID = 'spectral_centroid'
    SPECTRAL_ROLLOFF = 'spectral_rolloff'
    SPECTRAL_BANDWIDTH = 'spectral_bandwidth'

    RATE = 44100   # All recordings in ESC are 44.1 kHz
    FRAME = 512    # Frame size in samples

    def __init__(self, label=None):
        # Necessary to re-use code for training and prediction
        if label is None:
            self.label = ''
        else:
            self.label = label

    def extract_features(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged [mean].
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        zcr_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='zero_crossing_rate')
        rmse_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='rmse')
        mfcc_feature = self.compute_librosa_features(audio_data=audio_data, feature_name= 'mfcc')
        spectral_centroid_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='spectral_centroid')
        spectral_rolloff_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='spectral_rolloff')
        spectral_bandwidth_feature = self.compute_librosa_features(audio_data=audio_data, feature_name='spectral_bandwidth')

        concatenated_features = np.concatenate((zcr_feature,
                                      rmse_feature,
                                      mfcc_feature,
                                      spectral_centroid_feature,
                                      spectral_rolloff_feature,
                                      spectral_bandwidth_feature
                                      ), axis=0)

        logging.info('Averaging...')
        start = timeit.default_timer()

        mean_feature = np.mean(concatenated_features, axis=1, keepdims=True).transpose()

        stop = timeit.default_timer()
        logging.info('Time taken: {0}'.format(stop - start))

        return mean_feature, self.label

    def compute_librosa_features(self, audio_data, feature_name):
        """
        Compute feature using librosa methods

        :param audio_data: signal
        :param feature_name: feature to compute 
            possible values: 'zero_crossing_rate', 'rmse', 'mfcc', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth'
        :return: np array
        """
        # # http://stackoverflow.com/questions/41896123/librosa-feature-tonnetz-ends-up-in-typeerror
        # chroma_cens_feat = chroma_cens(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

        logging.info('Computing {}...'.format(feature_name))

        if feature_name == self.ZERO_CROSSING_RATE:
            return zero_crossing_rate(y=audio_data, hop_length=self.FRAME)
        elif feature_name == self.RMSE:
            return rms(y=audio_data, hop_length=self.FRAME)
        elif feature_name == self.MFCC:
            return mfcc(y=audio_data, sr=self.RATE, n_mfcc=13)
        elif feature_name == self.SPECTRAL_CENTROID:
            return spectral_centroid(y=audio_data, sr=self.RATE, hop_length=self.FRAME)
        elif feature_name == self.SPECTRAL_ROLLOFF:
            return spectral_rolloff(y=audio_data, sr=self.RATE, hop_length=self.FRAME, roll_percent=0.90)
        elif feature_name == self.SPECTRAL_BANDWIDTH:
            return spectral_bandwidth(y=audio_data, sr=self.RATE, hop_length=self.FRAME)

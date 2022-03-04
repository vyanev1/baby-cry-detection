# -*- coding: utf-8 -*-

import numpy as np
from librosa.feature import zero_crossing_rate, mfcc, spectral_centroid, spectral_rolloff, spectral_bandwidth, rms


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

    def __init__(self):
        pass

    def extract_features(self, audio_data):
        """
        Extract features using librosa.feature.

        Each signal is cut into frames, features are computed for each frame and averaged [median].
        The numpy array is transformed into a data frame with named columns.

        :param audio_data: the input signal samples with frequency 44.1 kHz
        :return: a numpy array (numOfFeatures x numOfShortTermWindows)
        """

        zcr_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.ZERO_CROSSING_RATE)
        rmse_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.RMSE)
        mfcc_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.MFCC)
        spectral_centroid_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.SPECTRAL_CENTROID)
        spectral_rolloff_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.SPECTRAL_ROLLOFF)
        spectral_bandwidth_feature = self.compute_librosa_features(audio_data=audio_data, feature_name=self.SPECTRAL_BANDWIDTH)

        concat_feature = np.concatenate((zcr_feature,
                                      rmse_feature,
                                      mfcc_feature,
                                      spectral_centroid_feature,
                                      spectral_rolloff_feature,
                                      spectral_bandwidth_feature
                                      ), axis=0)

        return np.mean(concat_feature, axis=1, keepdims=True).transpose()

    def compute_librosa_features(self, audio_data, feature_name):
        """
        Compute feature using librosa methods

        :param audio_data: signal
        :param feat_name: feature to compute
        :return: np array
        """

        ## /!\ rmse in librosa 0.4.3 and 0.5.0
        ## /!\ rms in librosa 0.7.0

        ## /!\ librosa 0.5.0
        # if rmse_feat.shape == (1, 427):
        #     rmse_feat = np.concatenate((rmse_feat, np.zeros((1, 4))), axis=1)

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


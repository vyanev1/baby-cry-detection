from __future__ import division

from tests import TestBabyCry
from baby_cry_detection.pc_methods import AudioReader
from baby_cry_detection.pc_methods.feature_extractor import FeatureExtractor


class PcMethodsTest(TestBabyCry):
    """
    Test pc_methods
    """

    def test_read_audio_file(self):

        reader = AudioReader(file_name=self.file_name)

        track, sr = reader.read_audio_file()

        self.assertEqual(sr, 44100)
        self.assertEqual(track.size, 5*sr)

    def test_feature_engineer(self):

        feature_engineer = FeatureExtractor(self.label)

        features, label = feature_engineer.extract_features(self.pc_sample)

        self.assertEqual(features.shape, (1, 18))
        self.assertEqual(label, self.label)

    def test_compute_librosa_features(self):

        feature_engineer = FeatureExtractor()

        expected_computed_points = int(round(feature_engineer.RATE*5/feature_engineer.FRAME, 0))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='zero_crossing_rate').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='rmse').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='mfcc').shape,
                         (20, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='spectral_centroid').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='spectral_rolloff').shape,
                         (1, expected_computed_points))

        self.assertEqual(feature_engineer.compute_librosa_features(audio_data=self.pc_sample, feature_name='spectral_bandwidth').shape,
                         (1, expected_computed_points))




# -*- coding: utf-8 -*-

import argparse
import os
import pickle
import sys
import warnings

egg_path = '{}/../lib/baby_cry_detection-1.1-py2.7.egg'.format(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(egg_path)

from baby_cry_detection.rpi_methods import Reader
from baby_cry_detection.rpi_methods.baby_cry_predictor import CryingBabyPredictor
from baby_cry_detection.rpi_methods.feature_extractor import FeatureExtractor
from baby_cry_detection.rpi_methods.majority_voter import MajorityVoter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path_data',
                        default='{}/../recording/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--load_path_model',
                        default='{}/../model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../prediction/'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path_data = os.path.normpath(args.load_path_data)
    load_path_model = os.path.normpath(args.load_path_model)
    save_path = os.path.normpath(args.save_path)

    # Read signal
    file_name = 'signal_9s.wav'       # only one file in the folder
    file_reader = Reader(os.path.join(load_path_data, file_name))
    audio_data = file_reader.read_audio_file()

    # Feature extraction
    feature_extractor = FeatureExtractor()
    audio_data_features = list()
    for audio_signal_features in audio_data:
        tmp = feature_extractor.extract_features(audio_signal_features)
        audio_data_features.append(tmp)

    # Open Model
    # https://stackoverflow.com/questions/41146759/check-sklearn-version-before-loading-model-using-joblib
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)

      with open((os.path.join(load_path_model, 'model.pkl')), 'rb') as fp:
          model = pickle.load(fp)

    # Make Predictions
    predictor = CryingBabyPredictor(model)
    predictions = list()
    for audio_signal_features in audio_data_features:
        tmp = predictor.classify(audio_signal_features)
        predictions.append(tmp)

    # Majority Vote
    majority_voter = MajorityVoter(predictions)
    majority_vote = majority_voter.vote()

    # Save prediction result
    with open(os.path.join(save_path, 'prediction.txt'), 'wb') as text_file:
        text_file.write("{0}".format(majority_vote))

if __name__ == '__main__':
    main()

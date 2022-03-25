# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import pickle

import numpy as np

from baby_cry_detection.pc_methods.audio_classifier import AudioClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        default='{}/../../output/dataset/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--save_path',
                        default='{}/../../output/model/'.format(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument('--log_path',
                        default='{}/../../'.format(os.path.dirname(os.path.abspath(__file__))))

    # Arguments
    args = parser.parse_args()
    load_path = os.path.normpath(args.load_path)
    save_path = os.path.normpath(args.save_path)
    log_path = os.path.normpath(args.log_path)

    # Set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        filename=os.path.join(log_path, 'logs_pc_methods_model.log'),
                        filemode='w',
                        level=logging.DEBUG)

    # Train model
    logging.info('Calling TrainClassifier')

    X = np.load(os.path.join(load_path, 'dataset.npy'))
    y = np.load(os.path.join(load_path, 'labels.npy'))

    audio_classifier = AudioClassifier(X, y)
    performance, parameters, best_estimator = audio_classifier.train()

    # Save model
    logging.info('Saving model...')
    with open(os.path.join(save_path, 'model.pkl'), 'wb') as fp:
        pickle.dump(best_estimator, fp)

    # Save parameters
    with open(os.path.join(save_path, 'parameters.json'), 'w') as fp:
        json.dump(parameters, fp)

    # Save model performance
    with open(os.path.join(save_path, 'performance.json'), 'w') as fp:
        json.dump(performance, fp)

if __name__ == '__main__':
    main()

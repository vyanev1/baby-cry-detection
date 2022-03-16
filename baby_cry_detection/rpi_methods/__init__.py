# -*- coding: utf-8 -*-

import librosa

__all__ = [
    'Reader'
]


class Reader:
    """
    Read input audio file
    file_name: 'path/to/file/filename.mp3'
    """

    def __init__(self, file_name):
        self.file_name = file_name
        pass

    def read_audio_file(self):
        """
        Read audio file using librosa package. librosa allows resampling to desired sample rate and convertion to mono.

        :return:
        * audio_data: a list of audio_data as numpy.ndarray. There are 5 overlapping signals, each one is 5-second long.
        """

        audio_data = list()

        for offset in range(5):
            audio_data_offset, _ = librosa.load(self.file_name, sr=44100, mono=True, offset=offset, duration=5.0)
            audio_data.append(audio_data_offset)

        return audio_data

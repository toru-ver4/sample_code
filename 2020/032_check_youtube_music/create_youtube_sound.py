# -*- coding: utf-8 -*-
"""
create test sound for YouTube Music
===================================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from scipy.io import wavfile
import wavio

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def create_sine_wave(frequency=440, seconds=1.5, sampling_rate=44100):
    """
    create the sine wave

    Parameters
    ----------
    frequency : int
        frequency
    seconds : float
        length of the time
    sampling_rate : int
        sampling rate

    Returns
    -------
    ndarray (np.float)
        sine wave data

    Notes
    -----
    the gain is normalized to 1.0
    """
    tt = np.linspace(0, seconds, int(seconds * sampling_rate), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * tt)

    return sine_wave


def mono_to_streo(data):
    """
    Parameters
    ----------
    data : ndarray (np.float)
        monoral data
    """
    out_data = np.dstack((data, data)).reshape((data.shape[0], 2))
    return out_data


def mono_to_stereo_if_data_is_mono(data):
    if len(data.shape) != 2:
        out_data = mono_to_streo(data)
    else:
        out_data = data
    
    return out_data


def save_sine_wave_16bit(sine_wave, sampling_rate, filename):
    """
    save the sine wave in 16 bit depth.

    Parameters
    ----------
    sine_wave : ndarray (np.float)
        sine wave data
    sampling_rate : int
        sampling rate
    filename : str
        filename
    """
    sine_wave = mono_to_stereo_if_data_is_mono(sine_wave)
    sine_wave = np.clip(sine_wave, -1.0, 1.0)
    sine_wave_16bit = np.int16(np.round(sine_wave * np.iinfo(np.int16).max))
    wavfile.write(filename=filename, rate=sampling_rate, data=sine_wave_16bit)


def save_sine_wave_24bit(sine_wave, sampling_rate, filename):
    """
    save the sine wave in 24 bit depth.

    Parameters
    ----------
    sine_wave : ndarray (np.float)
        sine wave data
    sampling_rate : int
        sampling rate
    filename : str
        filename
    """
    max_value_int24 = (2 ** 23) - 1
    sine_wave = mono_to_stereo_if_data_is_mono(sine_wave)
    sine_wave = np.clip(sine_wave, -1.0, 1.0)
    sine_wave_24bit = np.int32(np.round(sine_wave * max_value_int24))
    wavio.write(
        filename, sine_wave_24bit, sampling_rate, scale="none", sampwidth=3)


def fade_in_out_effect(data, sampling_rate=48000, seconds=0.0065):
    """
    add fade in-out effect to the audio data

    Parameters
    ----------
    data : ndarray (np.float)
        sound data.
    sampling_rate : int
        sampling rate
    seconds : float
        length of the time.
    """
    sample = int(sampling_rate * seconds + 0.5)
    x = np.linspace(0, 0.5 * np.pi, sample)
    y = np.sin(x)
    data[:sample] = data[:sample] * y
    data[-sample:] = data[-sample:] * y[::-1]


def db_to_linear_gain(db_value):
    """
    Parameters
    ----------
    db_value : float
        Decibel value

    Examples
    --------
    >>> db_to_linear_gain(db_value=-3)
    0.5011872336272722
    >>> db_to_linear_gain(db_value=10)
    10.0
    """
    return 10 ** (db_value / 10)


def apply_gain(data, db_value):
    """
    apply gain

    Parameters
    ----------
    data : ndarray (np.float)
        data
    db_value : float
        Decibel value

    Examples
    --------
    >>> data = np.arange(5)
    >>> apply_gain(data, db_value=-3)
    [0.         0.50118723 1.00237447 1.5035617  2.00474893]
    """
    return data * db_to_linear_gain(db_value=db_value)


def create_and_save_sine_wave(
        frequency=440, sampling_rate=44100, seconds=0.5,
        fade_in_out=True, db_value=-3.0):
    """
    create and save sine wave.

    Parameters
    ----------
    frequency : int
        frequency
    samplineg_rate : int
        sampling rate
    seconds : float
        length of the time
    fade_in_out : bool
        whether to fade in / fade out
    db_value : float
        gain in Decibel unit.
    """
    fname = f"./audio/freq_{int(frequency):05d}Hz_db{db_value}_24bit.wav"
    data = create_sine_wave(
        frequency=frequency, seconds=seconds, sampling_rate=sampling_rate)
    if fade_in_out:
        fade_in_out_effect(data, sampling_rate=sampling_rate)
    data = apply_gain(data, db_value=db_value)
    save_sine_wave_24bit(
        sine_wave=data, sampling_rate=sampling_rate, filename=fname)


def make_test_data():
    freq = 440
    sr = 44100
    sec = 0.5
    fade_sec = 0.007
    db_value = -3
    fname_16bit = f"./audio/freq_{int(freq):05d}Hz_db{db_value}_16bit.wav"
    fname_24bit = f"./audio/freq_{int(freq):05d}Hz_db{db_value}_24bit.wav"
    data = create_sine_wave(
        frequency=freq, seconds=sec, sampling_rate=sr)
    fade_in_out_effect(data, sampling_rate=sr, seconds=fade_sec)
    data = apply_gain(data, db_value=db_value)
    save_sine_wave_16bit(
        sine_wave=data, sampling_rate=sr, filename=fname_16bit)
    save_sine_wave_24bit(
        sine_wave=data, sampling_rate=sr, filename=fname_24bit)


def create_sine_wave_files():
    sampling_rate = 44100
    seconds = 0.25
    fade_in_out = True
    db_value_list = [0]
    freq_factor_list = [0.5, 1.0, 2.0, 4.0, 8.0, 16]
    freq_idx_list = [
        261.626, 293.665, 329.628, 349.228, 391.995, 440.000, 493.883]
    for freq_factor in freq_factor_list:
        for freq_idx in freq_idx_list:
            freq = (freq_idx + 1) * freq_factor
            for db_value in db_value_list:
                create_and_save_sine_wave(
                    frequency=freq, sampling_rate=sampling_rate,
                    seconds=seconds, fade_in_out=fade_in_out,
                    db_value=db_value)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # make_test_data()
    create_sine_wave_files()

# -*- coding: utf-8 -*-
"""
音声ファイルを作る
===================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from scipy.io import wavfile

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_cycle_param(freq=440, sec=5, sampling_rate=48000):
    """
    基本的なパラメータを計算する。

    sample_per_one_cycle: 1周期に必要なサンプル数
    cycle_num: 生成秒数に必要な周期の数
    total_sample_num: sample_per_one_cycle * cycle_num

    # memo
    time_per_one_cycle[s] = sample_per_one_cycle / sampling_rate
    freq_per_one_cycle[Hz] = sampling_rate / sample_per_one_cycle
    sample_per_one_cycle = sampling_rate / freq_per_one_cycle[Hz]
    """
    temp_total_sample_num = sec * sampling_rate
    sample_per_one_cycle = int(round(sampling_rate / freq))
    cycle_num = int(temp_total_sample_num / sample_per_one_cycle + 0.5)
    total_sample_num = cycle_num * sample_per_one_cycle

    return total_sample_num, sample_per_one_cycle, cycle_num


def write_wav_file(fname, sampling_rate, data):
    """
    wav形式で保存する。

    data: ndarray
        [-1.0:1.0]の浮動小数点データであること。
        量子化は本関数内で行う。
    """
    int_data = np.int16(np.round(data * np.iinfo(np.int16).max))
    wavfile.write(fname, sampling_rate, int_data)


def get_sine_wave(freq=440, sec=5, sampling_rate=48000, gain=0.9):
    """
    """
    total_sample_num, sample_per_one_cycle, cycle_num\
        = calc_cycle_param(freq=freq, sec=sec, sampling_rate=sampling_rate)

    x = np.linspace(0, 2*np.pi, sample_per_one_cycle, endpoint=False)
    one_sine = np.sin(x)
    all_sine = np.tile(one_sine, cycle_num) * gain

    return np.int16(np.round(all_sine * np.iinfo(np.int16).max))


def add_fade_in_out(data, sec=0.005, sampling_rate=48000):
    """
    データにプチノイズが乗らないように
    始めと終わりをまろやかにする。
    """
    sample = int(sampling_rate * sec + 0.5)
    x = np.linspace(0, 0.5 * np.pi, sample)
    y = np.sin(x)
    data[:sample] = data[:sample] * y
    data[-sample:] = data[-sample:] * y[::-1]


def make_countdown_sound(sampling_rate=48000):
    count_down_sec = 4
    left_st_sec = 1
    right_st_sec = 2
    center_st_sec = 3
    low_freq = 1000
    high_freq = 2000
    beep_sec = 0.06
    fade_in_out_sec = 0.0065
    total_sample = count_down_sec * sampling_rate
    # offset_sample = int(fade_in_out_sec * sampling_rate * 0.5)
    offset_sample = int(0.0)

    # 無音ファイル
    np.zeros((total_sample), dtype=np.int16)

    # 100ms だけ鳴らすファイル
    sine_low = get_sine_wave(
        freq=low_freq, sec=beep_sec, sampling_rate=sampling_rate, gain=0.9)
    sine_high = get_sine_wave(
        freq=high_freq, sec=beep_sec, sampling_rate=sampling_rate, gain=0.7)
    add_fade_in_out(sine_low, fade_in_out_sec, sampling_rate)
    add_fade_in_out(sine_high, fade_in_out_sec, sampling_rate)

    # left
    st_sample = sampling_rate * left_st_sec - offset_sample
    left_sound = np.zeros((total_sample), dtype=np.int16)
    left_sound[st_sample:st_sample+sine_low.shape[0]] = sine_low

    # right
    st_sample = sampling_rate * right_st_sec - offset_sample
    right_sound = np.zeros((total_sample), dtype=np.int16)
    right_sound[st_sample:st_sample+sine_low.shape[0]] = sine_low

    # center
    st_sample = sampling_rate * center_st_sec - offset_sample
    left_sound[st_sample:st_sample+sine_high.shape[0]] = sine_high
    right_sound[st_sample:st_sample+sine_high.shape[0]] = sine_high

    stereo = np.dstack((left_sound, right_sound)).reshape((total_sample, 2))
    wavfile.write("./wav/countdown.wav", sampling_rate, stereo)


def make_countdown_sound_99s(sampling_rate=48000):
    count_down_sec = 100
    left_st_sec_list = [x for x in range(1, count_down_sec, 2)]
    right_st_sec_list = [x for x in range(2, count_down_sec-1, 2)]
    low_freq = 1000
    beep_sec = 0.06
    fade_in_out_sec = 0.0065
    total_sample = count_down_sec * sampling_rate
    # offset_sample = int(fade_in_out_sec * sampling_rate * 0.5)
    offset_sample = 0.0

    # 無音ファイル
    np.zeros((total_sample), dtype=np.int16)

    # 100ms だけ鳴らすファイル
    sine_low = get_sine_wave(
        freq=low_freq, sec=beep_sec, sampling_rate=sampling_rate, gain=0.9)
    add_fade_in_out(sine_low, fade_in_out_sec, sampling_rate)

    # left
    left_sound = np.zeros((total_sample), dtype=np.int16)
    for left_st_sec in left_st_sec_list:
        st_sample = sampling_rate * left_st_sec - offset_sample
        left_sound[st_sample:st_sample+sine_low.shape[0]] = sine_low

    # right
    right_sound = np.zeros((total_sample), dtype=np.int16)
    for right_st_sec in right_st_sec_list:
        st_sample = sampling_rate * right_st_sec - offset_sample
        right_sound[st_sample:st_sample+sine_low.shape[0]] = sine_low

    stereo = np.dstack((left_sound, right_sound)).reshape((total_sample, 2))
    wavfile.write("./wav/countdown_99s.wav", sampling_rate, stereo)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_countdown_sound()
    # make_countdown_sound_99s()

# -*- coding: utf-8 -*-
"""
音声ファイルを作る
===================
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from scipy.io import wavfile
import turbo_colormap
from scipy import interpolate

# import my libraries
import test_pattern_generator2 as tpg

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
    sample_per_one_cycle = sampling_rate / freq
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


def read_wav_file(wav_file_name):
    sampling_rate, data = wavfile.read(wav_file_name)
    return sampling_rate, data / np.iinfo(data.dtype).max


def make_sine_wave(freq=440, sec=5, sampling_rate=48000):
    """
    """
    total_sample_num, sample_per_one_cycle, cycle_num\
        = calc_cycle_param(freq=freq, sec=sec, sampling_rate=sampling_rate)

    x = np.linspace(0, 2*np.pi, sample_per_one_cycle, endpoint=False)
    one_sine = np.sin(x)
    all_sine = np.tile(one_sine, cycle_num) * 0.9
    fname = f"./wav/sine_{freq:d}Hz_{sampling_rate/1000:.1f}kHz.wav"

    write_wav_file(fname, sampling_rate, all_sine)


def make_triangle_wave(freq=440, sec=5, sampling_rate=48000):
    total_sample_num, sample_per_one_cycle, cycle_num\
        = calc_cycle_param(freq=freq, sec=sec, sampling_rate=sampling_rate)

    x = np.linspace(0, 4, sample_per_one_cycle, endpoint=False)
    x[x > 2] = x[x > 2] * (-1) + 4
    x = x - 1
    triangle = np.tile(x, cycle_num) * 0.9

    fname = f"./wav/triangle_{freq:d}Hz_{sampling_rate/1000:.1f}kHz.wav"

    write_wav_file(fname, sampling_rate, triangle)


def make_square_wave(freq=440, sec=5, sampling_rate=48000):
    total_sample_num, sample_per_one_cycle, cycle_num\
        = calc_cycle_param(freq=freq, sec=sec, sampling_rate=sampling_rate)

    x = np.linspace(0, 1, sample_per_one_cycle, endpoint=False)
    one_cycle = np.round(x)
    all_cycle = np.tile(one_cycle, cycle_num) * 0.9

    fname = f"./wav/square_{freq:d}Hz_{sampling_rate/1000:.1f}kHz.wav"

    write_wav_file(fname, sampling_rate, all_cycle)


def make_sawtooth_wave(freq=440, sec=5, sampling_rate=48000):
    total_sample_num, sample_per_one_cycle, cycle_num\
        = calc_cycle_param(freq=freq, sec=sec, sampling_rate=sampling_rate)

    x = np.linspace(-1, 1, sample_per_one_cycle, endpoint=False)

    all_cycle = np.tile(x, cycle_num) * 0.9
    fname = f"./wav/sawtooth_{freq:d}Hz_{sampling_rate/1000:.1f}kHz.wav"
    write_wav_file(fname, sampling_rate, all_cycle)


def main_func():
    # for idx in range(6):
    #     freq = 440 * (2 ** idx)
    #     make_sine_wave(freq=freq, sec=5, sampling_rate=48000)
    #     make_triangle_wave(freq=freq, sec=5, sampling_rate=48000)
    #     make_square_wave(freq=freq, sec=5, sampling_rate=48000)
    #     make_sawtooth_wave(freq=freq, sec=5, sampling_rate=48000)
    # make_time_freq_plane('./wav/sawtooth_440Hz_48.0kHz.wav')
    make_time_freq_plane("./voice/aaa.wav")


def get_turbo_colormap():
    """
    Turbo の Colormap データを Numpy形式で取得する
    以下のソースコードを利用。
    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    """
    return np.array(turbo_colormap.turbo_colormap_data)


def log_y_to_turbo(log_y):
    """
    輝度データを Turbo で色付けする。
    輝度データは Non-Linear かつ [0:1] の範囲とする。

    Turbo は 256エントリ数の LUT として定義されているが、
    log_y の値は浮動小数点であるため、補間計算が必要である。
    今回は scipy.interpolate.interp1d を使って、R, G, B の
    各種値を線形補間して使う。

    Parameters
    ----------
    log_y : ndarray
        A 1D-array luminance data.

    Returns
    -------
    ndarray
        Luminance data. The shape is (height, width, 3).
    """
    # Turbo データ準備
    turbo = get_turbo_colormap()
    if len(turbo.shape) != 2:
        print("warning: turbo shape is invalid.")

    # scipy.interpolate.interp1d を使って線形補間する準備
    x = np.linspace(0, 1, turbo.shape[0])
    func_rgb = [interpolate.interp1d(x, turbo[:, idx]) for idx in range(3)]

    # 線形補間の実行
    out_rgb = [func(log_y) for func in func_rgb]

    return np.dstack(out_rgb)


def plot_time_freq_plane(data, freq):
    img = data / np.max(data)
    img = log_y_to_turbo(data ** (1/2.4))
    preview_img = img[::-1, :, :][-img.shape[0]//2:]
    tpg.preview_image(preview_img)


def make_time_freq_plane(wav_file_name):
    sampling_rate, data = read_wav_file(wav_file_name)
    total_sample = data.shape[0]
    data = data[..., 0] if len(data.shape) > 1 else data

    window_sample = 2048
    shift_sample = window_sample // 2
    shift_num = (total_sample // window_sample) * 2 - 1
    fft_positive_freq_num = window_sample // 2

    window = np.hamming(window_sample)
    freq = fft.fftfreq(
        window_sample, 1/sampling_rate)[:fft_positive_freq_num]

    buf = []

    for w_idx in range(shift_num):
        st_sample = shift_sample * w_idx
        target_data = data[st_sample:st_sample+window_sample] * window
        original_signal_power = np.sum(target_data ** 2) * window_sample
        fft_data = fft.fft(target_data)[:fft_positive_freq_num]
        fft_power_spectrum = (np.abs(fft_data) ** 2) / original_signal_power
        fft_power_spectrum = np.reshape(
            fft_power_spectrum, (fft_positive_freq_num, 1, 1)) ** 0.5
        fft_power_spectrum = np.tile(fft_power_spectrum, (4, 1))
        buf.append(fft_power_spectrum)

    data = np.hstack(buf)
    plot_time_freq_plane(data, freq)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

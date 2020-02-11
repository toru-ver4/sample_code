# -*- coding: utf-8 -*-
"""
Countdown動画を作る
===================


```
# 動作イメージ

```
bg_img_maker = BgImage(bg_image_param)
bg_img_maker.make()
bg_image_maker.save()

count_down_seq = CountDownSequence(count_down_param)
count_down_seq.make_and_save()

composite = Composite(composite_param)
composite.composite()
```
# 各クラスの概要

```
class BackgroundImage():
    # バックグラウンドを作る

    ## Input
    typedef struct{
        None;
    }coordinate_param;
    typedef struct{
        str transfer_function;
        int bg_luminance;
        int fg_luminance;
    }color_param;
    str fname;

class CountDownSequence():
    # カウントダウンの静止画シーケンスファイルを作る

    ## Input
    typedef struct{
        float frame_rate;
        int sec;
        int frame;
        int fill_color;
        int font_color;
        int bg_color;
        str fname_base;  // ファイル名のプレフィックス的なアレ
    }input_param;

class Compoite():
    # 背景と前景の静止画シーケンスを合成する

    ## Input
    typedef struct{
        image bg_image;
        image fg_image[];
        str fname_bg;  // ファイル名のプレフィックス的なアレ
        str fname_fg_base;  // ファイル名のプレフィックス的なアレ
    }
```
"""

# import standard libraries
import os

# import third-party libraries
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

# import my libraries
from countdown_movie import BackgroundImageColorParam,\
    BackgroundImageCoodinateParam, CountDownImageColorParam,\
    CountDownImageCoordinateParam
from countdown_movie import BackgroundImage, CountDownSequence
import transfer_functions as tf
import test_pattern_generator2 as tpg
from font_control import NOTO_SANS_MONO_EX_BOLD
from make_sound_file import make_countdown_sound

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

SDR_BG_COLOR_PARAM = BackgroundImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=90.0,
    sound_lumiannce=22.0,
    object_outline_luminance=1.0,
    step_ramp_code_values=([x * 64 for x in range(16)] + [1023])
)


BG_COODINATE_PARAM = BackgroundImageCoodinateParam(
    width=1920,
    height=1080,
    crosscross_line_width=4,
    outline_width=2,
    ramp_pos_v_from_center=400,
    ramp_height=84,
    ramp_outline_width=4,
    step_ramp_font_size=24,
    step_ramp_font_offset_x=5,
    step_ramp_font_offset_y=5,
    sound_text_font_size=200
)


SDR_COUNTDOWN_COLOR_PARAM = CountDownImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=60.0,
    object_outline_luminance=1.0,
)


COUNTDOWN_COORDINATE_PARAM = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=313,
    radius4=315,
    fps=24,
    crosscross_line_width=4,
    font_size=570,
    font_path=NOTO_SANS_MONO_EX_BOLD
)


SDR_BG_FILENAME_BASE = "./bg_img/backgraound_{}_{}x{}.tiff"
SDR_COUNTDOWN_FILENAME_BASE = "./fg_img/countdown_{}_{}x{}_{:06d}.tiff"


def calc_merge_st_pos(bg_image_maker, count_down_seq_maker):
    pos_h = bg_image_maker.width // 2 - count_down_seq_maker.img_width // 2
    pos_v = bg_image_maker.height // 2 - count_down_seq_maker.img_height // 2

    return (pos_h, pos_v)


def composite_sequence(
        sec, frame, counter, count_down_seq_maker, bg_image, merge_st_pos,
        dynamic_range):
    if sec > 0:
        fg_img = count_down_seq_maker.draw_countdown_seuqence_image(
            sec=sec, frame=frame)
        img = tpg.merge_with_alpha(
            bg_image, fg_img, tf_str=count_down_seq_maker.transfer_function,
            pos=merge_st_pos)
    else:
        if frame % count_down_seq_maker.fps == 0:
            img = bg_image.copy()
        else:
            img = np.zeros_like(bg_image)
    fname = "./movie_seq/movie_{:}_{:}x{:}_{:}fps_{:04d}.tiff".format(
        dynamic_range, bg_image.shape[1], bg_image.shape[0],
        count_down_seq_maker.fps, counter)
    print(fname)
    cv2.imwrite(fname, np.uint16(np.round(img * 0xFFFF)))


def thread_wrapper_composite_sequence(args):
    composite_sequence(**args)


def make_sdr_countdown_movie(
        bg_color_param, bg_coordinate_param,
        cd_color_param, cd_coordinate_param,
        dynamic_range='sdr', scale_factor=1):
    # background image
    bg_filename_base\
        = f"./bg_img/backgraound_{dynamic_range}_{{}}_{{}}x{{}}.tiff"
    bg_image_maker = BackgroundImage(
        color_param=bg_color_param, coordinate_param=bg_coordinate_param,
        fname_base=bg_filename_base, dynamic_range='sdr',
        scale_factor=scale_factor)

    # foreground image
    cd_filename_base\
        = f"./fg_img/countdown_{dynamic_range}_{{}}_{{}}x{{}}_{{:06d}}.tiff"
    count_down_seq_maker = CountDownSequence(
        color_param=cd_color_param,
        coordinate_param=cd_coordinate_param,
        fname_base=cd_filename_base,
        dynamic_range='sdr',
        scale_factor=scale_factor)

    # get merge pos
    merge_st_pos = calc_merge_st_pos(bg_image_maker, count_down_seq_maker)

    # composite
    counter = 0
    sec_list = [3, 2, 1, 0]
    sound_text_list = ["L", "R", "C", " "]
    for sec, sound_text in zip(sec_list, sound_text_list):
        bg_image_maker.sound_text = " "
        bg_image_maker.make()
        bg_image_without_sound = bg_image_maker.img.copy()
        bg_image_maker.sound_text = sound_text
        bg_image_maker.make()
        bg_image_with_sound = bg_image_maker.img.copy()

        args = []
        for frame in range(cd_coordinate_param.fps):
            if frame < int(cd_coordinate_param.fps * 0.5 + 0.5):
                bg_image = bg_image_without_sound
            else:
                bg_image = bg_image_with_sound
            args.append(dict(sec=sec, frame=frame, counter=counter,
                             count_down_seq_maker=count_down_seq_maker,
                             bg_image=bg_image, merge_st_pos=merge_st_pos,
                             dynamic_range=dynamic_range))
            # composite_sequence(
            #     sec=sec, frame=frame, counter=counter,
            #     count_down_seq_maker=count_down_seq_maker,
            #     bg_image=bg_image, merge_st_pos=merge_st_pos,
            #     dynamic_range=dynamic_range)
            counter += 1
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_composite_sequence, args)


def make_sdr_hd_sequence():
    make_sdr_countdown_movie(
        bg_color_param=SDR_BG_COLOR_PARAM,
        cd_color_param=SDR_COUNTDOWN_COLOR_PARAM,
        dynamic_range='sdr',
        bg_coordinate_param=BG_COODINATE_PARAM,
        cd_coordinate_param=COUNTDOWN_COORDINATE_PARAM,
        scale_factor=1)
    # make_countdown_sound()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_sdr_hd_sequence()

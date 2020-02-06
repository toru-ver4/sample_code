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

# import my libraries
from countdown_movie import BackgroundImageColorParam,\
    BackgroundImageCoodinateParam, CountDownImageColorParam,\
    CountDownImageCoordinateParam
from countdown_movie import BackgroundImage, CountDownSequence
import transfer_functions as tf
import test_pattern_generator2 as tpg

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

SDR_COLOR_PARAM = BackgroundImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=90.0,
    object_outline_luminance=1.0,
    step_ramp_code_values=([x * 64 for x in range(16)] + [1023])
)


COODINATE_PARAM = BackgroundImageCoodinateParam(
    width=1920,
    height=1080,
    crosscross_line_width=4,
    outline_width=2,
    ramp_pos_v_from_center=400,
    ramp_height=84,
    ramp_outline_width=4,
    step_ramp_font_size=24,
    step_ramp_font_offset_x=5,
    step_ramp_font_offset_y=5
)


COUNTDOWN_COLOR_PARAM = CountDownImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=60.0,
    object_outline_luminance=1.0,
)


COUNTDOWN_COORDINATE_PARAM = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=310,
    radius4=315,
    fps=24,
    crosscross_line_width=4,
    font_size=570,
)


SDR_BG_FILENAME_BASE = "./bg_img/backgraound_{}_{}x{}.tiff"
SDR_COUNTDOWN_FILENAME_BASE = "./fg_img/countdown_{}_{}x{}_{:06d}.tiff"


def calc_merge_st_pos(bg_image_maker, count_down_seq_maker):
    pos_h = bg_image_maker.width // 2 - count_down_seq_maker.img_width // 2
    pos_v = bg_image_maker.height // 2 - count_down_seq_maker.img_height // 2

    return (pos_h, pos_v)


def make_sdr_countdown_movie():
    bg_image_maker = BackgroundImage(
        color_param=SDR_COLOR_PARAM, coordinate_param=COODINATE_PARAM,
        fname_base=SDR_BG_FILENAME_BASE, dynamic_range='sdr',
        scale_factor=1)
    bg_image_maker._debug_dump_param()
    bg_image_maker.make()
    bg_image_maker.save()
    bg_image = bg_image_maker.img

    count_down_seq_maker = CountDownSequence(
        color_param=COUNTDOWN_COLOR_PARAM,
        coordinate_param=COUNTDOWN_COORDINATE_PARAM,
        fname_base=SDR_COUNTDOWN_FILENAME_BASE,
        dynamic_range='sdr',
        scale_factor=1)
    merge_st_pos = calc_merge_st_pos(bg_image_maker, count_down_seq_maker)
    counter = 0
    for sec in [9, 8, 7, 6, 5, 4, 3, 2, 1]:
        for frame in range(24):
            fg_img = count_down_seq_maker.draw_countdown_seuqence_image(
                sec=sec, frame=frame)
            img = tpg.merge_with_alpha(
                bg_image, fg_img, tf_str=SDR_COLOR_PARAM.transfer_function,
                pos=merge_st_pos)
            fname = "./movie_seq/movie_{:04d}.tiff".format(counter)
            cv2.imwrite(fname, np.uint16(np.round(img * 0xFFFF)))
            counter += 1


def main_func():
    make_sdr_countdown_movie()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

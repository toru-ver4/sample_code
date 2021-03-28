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
from colour import Lab_to_XYZ, XYZ_to_RGB
from colour.models import BT709_COLOURSPACE, BT2020_COLOURSPACE

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

# REVISION = 2  # added sound information.
# REVISION = 3  # added signal information at the bottom.
# REVISION = 4  # added Three types of test patterns.

# improved chroma-subsampling chaker patterns.
# added checker board pattern to distinguish 10bit.
# REVISION = 5

REVISION = 6  # added 8bit-10bit identification pattern.


SDR_BG_COLOR_PARAM = BackgroundImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=90.0,
    sound_lumiannce=22.0,
    object_outline_luminance=1.0,
    step_ramp_code_values=([x * 64 for x in range(16)] + [1023]),
    gamut='ITU-R BT.709',
    text_info_luminance=50,
    crosshatch_luminance=28.0,
    checker_board_levels=[
        [398, 400], [400, 402], [402, 404], [404, 406]],
    ramp_10bit_levels=[384, 448],
    dot_droped_luminance=90.0
)


HDR_BG_COLOR_PARAM = BackgroundImageColorParam(
    transfer_function=tf.ST2084,
    bg_luminance=18.0,
    fg_luminance=90.0,
    sound_lumiannce=22.0,
    object_outline_luminance=1.0,
    step_ramp_code_values=([x * 64 for x in range(16)] + [1023]),
    gamut='ITU-R BT.2020',
    text_info_luminance=50,
    crosshatch_luminance=28.0,
    ramp_10bit_levels=[384, 448],
    dot_droped_luminance=90.0
)


BG_COODINATE_PARAM = BackgroundImageCoodinateParam(
    width=1920,
    height=1080,
    crosscross_line_width=4,
    outline_width=2,
    ramp_pos_v_from_center=400,
    ramp_height=84,
    ramp_outline_width=4,
    step_ramp_font_size=20,
    step_ramp_font_offset_x=5,
    step_ramp_font_offset_y=5,
    sound_text_font_size=200,
    info_text_font_size=25,
    limited_text_font_size=96,
    crosshatch_size=128,
    dot_dropped_text_size=133,
    lab_patch_each_size=48,
    even_odd_info_text_size=16,
    ramp_10bit_info_text_size=22
)


SDR_COUNTDOWN_COLOR_PARAM = CountDownImageColorParam(
    transfer_function=tf.GAMMA24,
    bg_luminance=18.0,
    fg_luminance=60.0,
    object_outline_luminance=1.0,
)


HDR_COUNTDOWN_COLOR_PARAM = CountDownImageColorParam(
    transfer_function=tf.ST2084,
    bg_luminance=18.0,
    fg_luminance=60.0,
    object_outline_luminance=1.0,
)


COUNTDOWN_COORDINATE_PARAM_24P = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=313,
    radius4=315,
    fps=24,
    crosscross_line_width=4,
    font_size=570,
    font_path=NOTO_SANS_MONO_EX_BOLD
)

COUNTDOWN_COORDINATE_PARAM_30P = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=313,
    radius4=315,
    fps=30,
    crosscross_line_width=4,
    font_size=570,
    font_path=NOTO_SANS_MONO_EX_BOLD
)

COUNTDOWN_COORDINATE_PARAM_04P = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=313,
    radius4=315,
    fps=4,
    crosscross_line_width=4,
    font_size=570,
    font_path=NOTO_SANS_MONO_EX_BOLD
)

COUNTDOWN_COORDINATE_PARAM_60P = CountDownImageCoordinateParam(
    radius1=360,
    radius2=320,
    radius3=313,
    radius4=315,
    fps=60,
    crosscross_line_width=4,
    font_size=470,
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
    fname_prefix = "/work/overuse/2020/005_make_countdown_movie/movie_seq/"
    fname = fname_prefix + "movie_{:}_{:}x{:}_{:}fps_{:04d}.png".format(
        dynamic_range, bg_image.shape[1], bg_image.shape[0],
        count_down_seq_maker.fps, counter)
    print(fname)
    cv2.imwrite(fname, np.uint16(np.round(img[..., ::-1] * 0xFFFF)))


def thread_wrapper_composite_sequence(args):
    composite_sequence(**args)


def make_countdown_movie(
        bg_color_param, bg_coordinate_param,
        cd_color_param, cd_coordinate_param,
        dynamic_range='sdr', scale_factor=1):
    """
    Make the sequence files.
    """
    # background image
    bg_filename_base\
        = f"./bg_img/backgraound_{dynamic_range}_{{}}_{{}}x{{}}.tiff"
    bg_image_maker = BackgroundImage(
        color_param=bg_color_param, coordinate_param=bg_coordinate_param,
        fname_base=bg_filename_base, dynamic_range=dynamic_range,
        scale_factor=scale_factor, fps=cd_coordinate_param.fps,
        revision=REVISION)
    bg_image_maker.sound_text = " "
    bg_image_maker.is_even_number = True
    bg_image_maker.make()  # dummy make for 8bit 10bit id pattern
    generator_list = make_8bit_10bit_id_pat_generator(
        bg_image_maker, scale_factor, dynamic_range)

    # foreground image
    cd_filename_base\
        = f"./fg_img/countdown_{dynamic_range}_{{}}_{{}}x{{}}_{{:06d}}.tiff"
    count_down_seq_maker = CountDownSequence(
        color_param=cd_color_param,
        coordinate_param=cd_coordinate_param,
        fname_base=cd_filename_base,
        dynamic_range=dynamic_range,
        scale_factor=scale_factor)

    # get merge pos
    merge_st_pos = calc_merge_st_pos(bg_image_maker, count_down_seq_maker)

    # composite
    counter = 0
    sec_list = [x for x in range(100)][::-1]
    sound_text_list = ["L", "R"] * 50
    sound_text_list[-1] = " "
    g_frame_cnt = 0
    for sec, sound_text in zip(sec_list, sound_text_list):
        # for chroma-subsampling pattern
        is_even_number = (sec % 2) == 0
        bg_image_maker.is_even_number = is_even_number

        # for audio L-R-C indicator
        bg_image_maker.sound_text = " "
        bg_image_maker.make()
        bg_image_without_sound_indicator = bg_image_maker.img.copy()
        bg_image_maker.save()

        bg_image_maker.sound_text = sound_text
        bg_image_maker.make()
        bg_image_with_sound_indicator = bg_image_maker.img.copy()

        args = []
        for frame in range(cd_coordinate_param.fps):
            if frame < int(cd_coordinate_param.fps * 0.5 + 0.5):
                bg_image = bg_image_without_sound_indicator
            else:
                bg_image = bg_image_with_sound_indicator

            # merge 8bit 10bit identification pattern
            make_8bit_10bit_pattern(
                bg_image_maker, bg_image, generator_list,
                cnt=g_frame_cnt)
            g_frame_cnt += 1

            d = dict(
                sec=sec, frame=frame, counter=counter,
                count_down_seq_maker=count_down_seq_maker,
                bg_image=bg_image.copy(), merge_st_pos=merge_st_pos,
                dynamic_range=dynamic_range)
            args.append(d)
            # composite_sequence(**d)
            counter += 1
            # break
        with Pool(cpu_count()) as pool:
            pool.map(thread_wrapper_composite_sequence, args)
        # break


def make_8bit_10bit_id_pat_generator(
        bg_image_maker, scale_factor, dynamic_range):
    id_param = bg_image_maker.get_8bit_10bit_id_pat_param()
    step = 24
    slide_step = 4
    hdr10 = True if dynamic_range == 'HDR' else False
    generator_l = tpg.IdPatch8bit10bitGenerator(
        width=id_param['patch_width'], height=id_param['patch_height'],
        total_step=step, level=tpg.L_LOW_C_HIGH,
        slide_step=slide_step*scale_factor,
        hdr10=hdr10)
    generator_m = tpg.IdPatch8bit10bitGenerator(
        width=id_param['patch_width'], height=id_param['patch_height'],
        total_step=step, level=tpg.L_MIDDLE_C_HIGH,
        slide_step=slide_step*scale_factor,
        hdr10=hdr10)
    generator_h = tpg.IdPatch8bit10bitGenerator(
        width=id_param['patch_width'], height=id_param['patch_height'],
        total_step=step, level=tpg.L_HIGH_C_HIGH,
        slide_step=slide_step*scale_factor,
        hdr10=hdr10)
    generator_list = [generator_l, generator_m, generator_h]

    return generator_list


def make_8bit_10bit_pattern(
        bg_image_maker, bg_image, generator_list, cnt=None, slide_step=4):
    id_param = bg_image_maker.get_8bit_10bit_id_pat_param()
    if cnt:
        cnt_actual = cnt * slide_step
    else:
        cnt_actual = cnt
    for idx, generator in enumerate(generator_list):
        img_8bit, img_10bit = generator.extract_8bit_10bit_img(cnt_actual)
        pos_h = id_param['patch_pos_h'][idx]
        pos_v = id_param['patch_pos_v'][idx]
        tpg.merge(bg_image, img_8bit, (pos_h, pos_v))

        pos_v += id_param['internal_margin_v'] + id_param['patch_height']
        tpg.merge(bg_image, img_10bit, (pos_h, pos_v))


def make_sequence():
    """
    Make the multiple types of sequence files at a time.
    """
    cd_coordinate_param_list = [
        COUNTDOWN_COORDINATE_PARAM_60P]
    for scale_factor in [1]:
        # for cd_coordinate_param in cd_coordinate_param_list:
        #     make_countdown_movie(
        #         bg_color_param=SDR_BG_COLOR_PARAM,
        #         cd_color_param=SDR_COUNTDOWN_COLOR_PARAM,
        #         dynamic_range='SDR',
        #         bg_coordinate_param=BG_COODINATE_PARAM,
        #         cd_coordinate_param=cd_coordinate_param,
        #         scale_factor=scale_factor)
        make_countdown_sound_99s()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    make_sequence()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NUKEなどのコンポジットツールで LOGファイルを読み込み
HDR表示を試すためのテストパターン画像を作成する
"""

import os
from TpgIO import TpgIO
from TpgDraw import TpgDraw
import transfer_functions as tf
import colour
from multiprocessing import Pool, cpu_count

# REVISION = 3  # 右上に色々と小物を追加
# REVISION = 4  # colour version 0.3.14
# REVISION = 5  # colour version 0.3.16
REVISION = 6  # colour version 0.4.3
BIT_DEPTH = 10


BT709_CS = colour.models.RGB_COLOURSPACE_BT709
BT2020_CS = colour.models.RGB_COLOURSPACE_BT2020
V_GAMUT_CS = colour.models.RGB_COLOURSPACE_V_GAMUT
ALEXA_WIDE_GAMUT_CS = colour.models.RGB_COLOURSPACE_ARRI_WIDE_GAMUT_3
S_GAMUT3_CINE_CS = colour.models.RGB_COLOURSPACE_S_GAMUT3_CINE
S_GAMUT3_CS = colour.models.RGB_COLOURSPACE_S_GAMUT3
V_LOG_CS = colour.models.RGB_COLOURSPACE_V_GAMUT
RED_WIDE_GAMUT_RGB_CS = colour.models.RGB_COLOURSPACE_RED_WIDE_GAMUT_RGB
DCI_P3_CS = colour.models.RGB_COLOURSPACE_P3_D65
SRGB_CS = colour.models.RGB_COLOURSPACE_sRGB
ACES_AP1_CS = colour.models.RGB_COLOURSPACE_ACESCG
ACES_AP0_CS = colour.models.RGB_COLOURSPACE_ACES2065_1

# PARAM_LIST = [{'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'},
#               {'tf': tf.GAMMA24, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.HLG, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.ST2084, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.ST2084, 'cs': DCI_P3_CS, 'wp': 'D65'},
#               {'tf': tf.SLOG3, 'cs': S_GAMUT3_CS, 'wp': 'D65'},
#               {'tf': tf.SLOG3_REF, 'cs': S_GAMUT3_CS, 'wp': 'D65'},
#               {'tf': tf.VLOG, 'cs': V_LOG_CS, 'wp': 'D65'},
#               {'tf': tf.LOGC, 'cs': ALEXA_WIDE_GAMUT_CS, 'wp': 'D65'},
#               {'tf': tf.LOGC, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.LOG3G10, 'cs': RED_WIDE_GAMUT_RGB_CS, 'wp': 'D65'},
#               {'tf': tf.LOG3G12, 'cs': RED_WIDE_GAMUT_RGB_CS, 'wp': 'D65'},
#               {'tf': tf.LOG3G10, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.LOG3G12, 'cs': BT2020_CS, 'wp': 'D65'},
#               {'tf': tf.ST2084, 'cs': ACES_AP1_CS, 'wp': 'D60'},
#               {'tf': tf.ST2084, 'cs': ACES_AP0_CS, 'wp': 'D60'}]
# PARAM_LIST = [{'tf': tf.LINEAR, 'cs': ACES_AP0_CS, 'wp': 'D60'}]
PARAM_LIST = [
    {'tf': tf.GAMMA24, 'cs': BT709_CS, 'wp': 'D65'},
    {'tf': tf.GAMMA24, 'cs': DCI_P3_CS, 'wp': 'D65'},
    {'tf': tf.GAMMA24, 'cs': BT2020_CS, 'wp': 'D65'},
    {'tf': tf.ST2084, 'cs': BT709_CS, 'wp': 'D65'},
    {'tf': tf.ST2084, 'cs': DCI_P3_CS, 'wp': 'D65'},
    {'tf': tf.ST2084, 'cs': BT2020_CS, 'wp': 'D65'}
]


class TpgControl:
    """
    必要なパラメータの受け取り。各種命令の実行。
    """
    def __init__(self, resolution='3840x2160', transfer_function=tf.GAMMA24,
                 color_space=BT709_CS, white_point="D65",
                 revision=REVISION):
        """
        white_point は 次のいずれか。'D50', 'D55', 'D60', 'D65', 'DCI-P3'
        """
        self.bg_color = 0.75  # unit is nits
        self.fg_color = 50  # unit is nits
        self.transfer_function = transfer_function
        self.parse_resolution(resolution)
        self.bit_depth = 10
        self.color_space = color_space
        self.white_point = white_point
        self.revision = revision
        self.draw_param = self.gen_keywords_for_draw()

    def parse_resolution(self, resolution):
        if resolution == '1920x1080':
            self.img_width = 1920
            self.img_height = 1080
        elif resolution == '3840x2160':
            self.img_width = 3840
            self.img_height = 2160
        else:
            raise ValueError("Invalid resolution parameter.")

    def gen_keywords_for_draw(self):
        """
        TpgDraw に渡すパラメータをまとめる
        """
        kwargs = {'bg_color': self.bg_color, 'fg_color': self.fg_color,
                  'img_width': self.img_width, 'img_height': self.img_height,
                  'bit_depth': self.bit_depth,
                  'transfer_function': self.transfer_function,
                  'color_space': self.color_space,
                  'white_point': self.white_point,
                  'revision': self.revision}

        return kwargs

    def draw_image_type1(self, preview=False):
        draw = TpgDraw(self.draw_param, preview)
        self.img = draw.draw_tpg_type1()

    def draw_image_type2(self, preview=False):
        draw = TpgDraw(self.draw_param, preview)
        self.img = draw.draw_tpg_type2()

    def save_image(self, fname, transfer_function):
        io = TpgIO(self.img, BIT_DEPTH, transfer_function)
        io.save_image(fname)

    def load_image(self, fname):
        io = TpgIO(BIT_DEPTH)
        self.load_img = io.load_image(fname)


def make_test_pattern(
        resolution, transfer_function, color_space, white_point, revision):

    fname_str_base = "./img/{}_{}_{}_{}_rev{:02d}_type1.{}"
    tpg_ctrl = TpgControl(resolution=resolution,
                          transfer_function=transfer_function,
                          color_space=color_space,
                          white_point=white_point,
                          revision=revision)
    tpg_ctrl.draw_image_type1(preview=False)
    fname = fname_str_base.format(transfer_function, color_space.name,
                                  white_point, resolution, revision, "dpx")
    tpg_ctrl.save_image(fname, transfer_function)
    fname = fname_str_base.format(transfer_function, color_space.name,
                                  white_point, resolution, revision, "exr")
    tpg_ctrl.save_image(fname, transfer_function)

    # # type2
    # fname_str_base = "./img/{}_{}_{}_{}_rev{:02d}_type2.{}"
    # tpg_ctrl.draw_image_type2(preview=False)
    # fname = fname_str_base.format(transfer_function, color_space.name,
    #                               white_point, resolution, revision, "dpx")
    # tpg_ctrl.save_image(fname, transfer_function)
    # fname = fname_str_base.format(transfer_function, color_space.name,
    #                               white_point, resolution, revision, "exr")
    # tpg_ctrl.save_image(fname, transfer_function)


def thread_wrapper(args):
    return make_test_pattern(*args)


def main_func():
    resolution_list = ['1920x1080', '3840x2160']

    with Pool(cpu_count()//2) as pool:
        args = [
            (x, y['tf'], y['cs'], y['wp'], REVISION)
            for x in resolution_list
            for y in PARAM_LIST
        ]
        pool.map(thread_wrapper, args)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

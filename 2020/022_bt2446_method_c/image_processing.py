# -*- coding: utf-8 -*-
"""
Image Processing for 'parameter adjustment tool'
================================================

"""

# import standard libraries
import os

# import third-party libraries
import cv2
import numpy as np

# import my libraries
from key_names import KeyNames
import bt2446_method_c as bmc
import transfer_functions as tf
import test_pattern_generator2 as tpg
import colormap as cmap

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []

kns = KeyNames()


TP_IMAGE_PATH = "./img/step_ramp_step_33.png"
LOW_IMAGE_PATH = "./img/dark.png"
MID_IMAGE_PATH = "./img/middle.png"
HIGH_IMAGE_PATH = "./img/high.png"


class ImageProcessing():
    """
    event controller for parameter_adjustment_tool.
    """
    def __init__(self, width=720, peak_luminance=1000):
        self.tp_img_path = TP_IMAGE_PATH
        self.low_image_path = LOW_IMAGE_PATH
        self.mid_image_path = MID_IMAGE_PATH
        self.high_image_path = HIGH_IMAGE_PATH
        self.peak_luminance = peak_luminance
        self.width = width

    def read_img(self, path):
        img = cv2.imread(
            path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1]
        max_value = np.iinfo(img.dtype).max
        img = img / max_value

        return img

    def read_and_concat_img(self):
        img_tp = self.resize_to_fixed_width(
            self.read_img(self.tp_img_path))
        img_low = self.resize_to_fixed_width(
            self.read_img(self.low_image_path))
        img_mid = self.resize_to_fixed_width(
            self.read_img(self.mid_image_path))
        img_high = self.resize_to_fixed_width(
            self.read_img(self.high_image_path))
        self.raw_img = np.vstack([img_tp, img_low, img_mid, img_high])

        # keep coordinate information
        tp_height = img_tp.shape[0]
        low_height = img_low.shape[0]
        mid_height = img_mid.shape[0]
        high_height = img_high.shape[0]
        self.img_v_pos_info = [
            dict(st=0, ed=tp_height),
            dict(st=tp_height, ed=tp_height+low_height),
            dict(st=tp_height+low_height, ed=tp_height+low_height+mid_height),
            dict(st=tp_height+low_height+mid_height,
                 ed=tp_height+low_height+mid_height+high_height)
        ]

    def extract_image(self, img, img_idx=0):
        st = self.img_v_pos_info[img_idx]['st']
        ed = self.img_v_pos_info[img_idx]['ed']
        devided_img = img[st:ed]

        return devided_img

    def non_linear_to_luminance(self, img):
        img_luminance = tf.eotf_to_luminance(img, tf.ST2084)
        return img_luminance

    def get_concat_raw_image(self):
        return self.raw_img

    def conv_8bit_int(self, img):
        return np.uint8(np.round(img * 0xFF))

    def conv_16bit_int(self, img):
        return np.uint16(np.round(img * 0xFFFF))

    def conv_io_stream(self, img):
        is_success, buffer = cv2.imencode(".png", img[..., ::-1])
        return buffer.tobytes()

    def resize_to_fixed_width(self, img):
        src_width = img.shape[1]
        src_height = img.shape[0]
        dst_width = self.width
        dst_height = int(self.width / src_width * src_height + 0.5)

        dst_img = cv2.resize(
            img, (dst_width, dst_height), interpolation=cv2.INTER_AREA)

        return dst_img

    def make_colormap_image(self, turbo_peak_luminance=1000):
        self.colormap_img = cmap.apply_st2084_to_srgb_colormap(
            self.raw_img, sdr_pq_peak_luminance=100,
            turbo_peak_luminance=turbo_peak_luminance)

    def get_colormap_image(self):
        return self.colormap_img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # im_pro = ImageProcessing()
    # im_pro.read_and_concat_img()
    # im_pro.apply_colormap(im_pro.raw_img, 4000)

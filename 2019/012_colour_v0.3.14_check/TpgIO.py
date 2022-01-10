#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# TpgIO モジュール

## 概要
TestPattern作成時の File への Write/Read を使いやすいように
カスタムしたモジュール。TyImageIO の用途を更に制限した感じ。
"""
import os
import OpenImageIO as oiio
import numpy as np
from TyImageIO import TyWriter, TyReader
import transfer_functions as tf


class TpgIO:
    """
    TestPattern作成に特化した File IO の提供
    """

    def __init__(self, img=None, bit_depth=10, transfer_function=None):
        self.img = img
        self.bit_depth = bit_depth
        self.img_max = (2 ** self.bit_depth) - 1
        self.tf = transfer_function

    def save_dpx_image(self, fname):
        attr = {"oiio:BitsPerSample": self.bit_depth}
        writer = TyWriter(self.img / self.img_max, fname, attr)
        writer.write()

    def save_exr_image(self, fname):
        attr = None
        img_temp = self.img / self.img_max
        img_temp = tf.eotf_to_luminance(img_temp, name=self.tf) / 100
        writer = TyWriter(img_temp, fname, attr)
        writer.write(out_img_type_desc=oiio.FLOAT)

    def save_tiff_image(self, fname):
        attr = {'png:compressionLevel': 7}
        writer2 = TyWriter(self.img / self.img_max, fname, attr)
        writer2.write(out_img_type_desc=oiio.UINT16)

    def save_image(self, fname):
        root, ext = os.path.splitext(fname)
        if ext == '.dpx':
            self.save_dpx_image(fname)

            # TIFFファイルも合わせて吐き出しておく
            self.save_tiff_image(root + '.png')
        elif ext == '.exr':
            self.save_exr_image(fname)

    def load_image(self, fname):
        """
        UIをどうするか、まだ決まっていない。
        """
        reader = TyReader(fname)
        self.load_img = reader.read()
        self.load_attr = reader.get_attr()
        self.load_img = self.load_img \
            / reader.get_max_value_from_dtype(self.load_img)
        img = np.uint16(np.round(self.load_img * self.img_max))

        return img


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Image IO モジュール

## 概要
OpenImageIO を使いやすいようにカスタムしたモジュール。
主にテストパターンの読み書きに使う。

## 設計思想
全部コミコミの TyImageIO クラスを作り、
そこから Writeクラス、Readerクラスを作ってみる。
"""

import os
import numpy as np
import OpenImageIO as oiio


class TyImageIO:
    """
    OpenImageIOを利用したファイルのWrite/Readを提供。
    拡張子でファイルの型を判別。最大値で正規化して保存する。
    """

    def __init__(self):
        pass

    def gen_out_img_spec(self, img, type_desk):
        xres = img.shape[1]
        yres = img.shape[0]
        nchannels = img.shape[2]
        img_spec = oiio.ImageSpec(xres, yres, nchannels, type_desk)

        return img_spec

    def get_max_value_from_dtype(self, img):
        try:
            img_max_value = np.iinfo(img.dtype).max
        except:
            img_max_value = 1.0

        return img_max_value

    def normalize_by_dtype(self, img):
        img_max_value = self.get_max_value_from_dtype(img)
        return np.double(img/img_max_value)

    def np_img_to_oiio_type_desc(self, img):
        """
        numpy の image data から
        OIIO の TypeDesk 情報を得る。

        Parameters
        ----------
        img : ndarray
            image data.

        Returns
        -------
        TypeDesk
            a type desctipter for oiio module.
        """

        data_type = img.dtype.type

        if data_type == np.int8:
            return oiio.INT8
        if data_type == np.int16:
            return oiio.INT16
        if data_type == np.int32:
            return oiio.INT32
        if data_type == np.int64:
            return oiio.INT64
        if data_type == np.uint8:
            return oiio.UINT8
        if data_type == np.uint16:
            return oiio.UINT16
        if data_type == np.uint32:
            return oiio.UINT32
        if data_type == np.uint64:
            return oiio.UINT64
        if data_type == np.float16:
            return oiio.HALF
        if data_type == np.float32:
            return oiio.FLOAT
        if data_type == np.float64:
            return oiio.DOUBLE

        raise TypeError("unknown img format.")

    def set_img_spec_attribute(self, img_spec, attr=None):
        """
        OIIO の ImageSpec に OIIO Attribute を設定する。

        Parameters
        ----------
        img_spec : OIIO ImageSpec
            specification of the image
        attr : dictionary
            attribute parameters.

        Returns
        -------
        -
        """

        if attr is None:
            return

        for key, value in attr.items():
            if key == "ICCProfile":
                img_spec.attribute(key, oiio.UINT8, value)
            elif isinstance(value, list) or isinstance(value, tuple):
                img_spec.attribute(key, value[0], value[1])
            else:
                img_spec.attribute(key, value)

    def get_img_spec_attribute(self, img_spec):
        """
        OIIO の ImageSpec から OIIO Attribute を取得する。

        Parameters
        ----------
        img_spec : OIIO ImageSpec
            specification of the image

        Returns
        -------
        attr : dictionary
            attribute parameters.
        """
        attr = {}
        for idx in range(len(img_spec.extra_attribs)):
            key = img_spec.extra_attribs[idx].name
            value = img_spec.extra_attribs[idx].value
            attr[key] = value
        return attr

    def save_img_using_oiio(self, img, fname,
                            out_img_type_desc=oiio.UINT16, attr=None):
        """
        OIIO を使った画像保存。

        Parameters
        ----------
        img : ndarray
            image data.
        fname : strings
            filename of the image.
        out_img_type_desc : oiio.desc
            type descripter of img
        attr : dictionary
            attribute parameters.

        Returns
        -------
        -

        Examples
        --------
        see ```_test_save_various_format()```

        """

        img_out = oiio.ImageOutput.create(fname)
        if not img_out:
            raise Exception("Error: {}".format(oiio.geterror()))
        out_img_spec = self.gen_out_img_spec(img, out_img_type_desc)
        self.set_img_spec_attribute(out_img_spec, attr)
        img_out.open(fname, out_img_spec)
        img_out.write_image(img)
        img_out.close()

    def load_img_using_oiio(self, fname):
        """
        OIIO を使った画像読込。

        Parameters
        ----------
        fname : strings
            filename of the image.

        Returns
        -------
        img : ndarray
            image data.
        attr : dictionary
            attribute parameters for dpx.

        Examples
        --------
        see ```_test_load_various_format()```

        """
        # データの読み込みテスト
        img_input = oiio.ImageInput.open(fname)
        if not img_input:
            raise Exception("Error: {}".format(oiio.geterror()))

        self.img_spec = img_input.spec()
        self.attr = self.get_img_spec_attribute(self.img_spec)
        self.typedesc = self.img_spec.format
        img_data = img_input.read_image(self.typedesc)

        img_input.close()

        return img_data

    def timecode_str_to_bcd(self, time_code_str):
        """
        '01:23:45:12' のようなタイムコードの文字列表記を
        0x01234512 に変換する。

        Examples
        --------
        >>> bcd = timecode_str_to_bcd(time_code_str='01:23:45:12')
        >>> print("0x{:08X}".format(bcd))
        0x12345612
        """
        temp_str = time_code_str.replace(':', '')
        if len(temp_str) != 8:
            raise TypeError('invalid time code str!')

        bcd = 0
        for idx in range(len(temp_str)):
            bcd += int(temp_str[idx]) << (8 - idx - 1) * 4

        return (int(bcd), int(0))


class TyWriter(TyImageIO):
    def __init__(self, img, fname, attr=None):
        super().__init__()
        self.is_float_dtype(img)
        self.img = img
        self.fname = fname
        self.attr = attr

    def is_float_dtype(self, img):
        float_set = {np.dtype(np.float16), np.dtype(np.float32),
                     np.dtype(np.float64)}
        if img.dtype not in float_set:
            raise ValueError('img is mut be float type')

    def write(self, out_img_type_desc=oiio.UINT16):
        self.save_img_using_oiio(img=self.img, fname=self.fname,
                                 out_img_type_desc=out_img_type_desc,
                                 attr=self.attr)


class TyReader(TyImageIO):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.attr = None

    def read(self):
        img = self.load_img_using_oiio(fname=self.fname)
        return img

    def get_attr(self):
        return self.attr


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

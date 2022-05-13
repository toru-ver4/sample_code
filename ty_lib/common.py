#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# 概要
どのモジュールからも使われそうな関数群

"""

import numpy as np
import math
import time

MODULE_IMPORT_TIME = time.time()


def is_numpy_module(data):
    return type(data).__module__ == np.__name__


def is_small_xy_array_shape(data):
    """
    # brief
    check whether the shape of data is (N, 2).
    """
    if not is_numpy_module(data):
        raise TypeError("data must be a numpy instance")
    if len(data.shape) != 2:
        return False
    if data.shape[1] != 2:
        return False
    return True


def is_img_shape(data):
    """
    # brief
    check whether the shape of data is (N, M, 3).
    """
    if not is_numpy_module(data):
        raise TypeError("data must be a numpy instance")
    if len(data.shape) != 3:
        return False
    if data.shape[2] != 3:
        return False
    return True


def is_numpy_color_module(data):
    """
    # brief
    check whether the data is numpy.
    and element num is 3.
    """
    if not is_numpy_module(data):
        raise TypeError("data must be a numpy instance")
    if data.shape[-1] != 3:
        return False
    return True


def is_correct_dtype(data, types={np.uint32, np.uint64}):
    """
    # brief
    for numpy instance only
    # note
    types must be a set.
    """
    if not is_numpy_module(data):
        raise TypeError("data must be a numpy instance")
    if not isinstance(types, set):
        raise TypeError("dtypes must be a set")

    return data.dtype.type in types


def equal_devision(length, div_num):
    """
    # 概要
    length を div_num で分割する。
    端数が出た場合は誤差拡散法を使って上手い具合に分散させる。
    """
    base = length / div_num
    ret_array = [base for x in range(div_num)]

    # 誤差拡散法を使った辻褄合わせを適用
    # -------------------------------------------
    diff = 0
    for idx in range(div_num):
        diff += math.modf(ret_array[idx])[0]
        if diff >= 1.0:
            diff -= 1.0
            ret_array[idx] = int(math.floor(ret_array[idx]) + 1)
        else:
            ret_array[idx] = int(math.floor(ret_array[idx]))

    # 計算誤差により最終点が +1 されない場合への対処
    # -------------------------------------------
    diff = length - sum(ret_array)
    if diff != 0:
        ret_array[-1] += diff

    # 最終確認
    # -------------------------------------------
    if length != sum(ret_array):
        raise ValueError("the output of equal_division() is abnormal.")

    return ret_array


def get_3d_grid_cube_format(grid_num=4):
    """
    # 概要
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), ...
    みたいな配列を返す。
    CUBE形式の3DLUTを作成する時に便利。
    """

    base = np.linspace(0, 1, grid_num)
    ones_x = np.ones((grid_num, grid_num, 1))
    ones_y = np.ones((grid_num, 1, grid_num))
    ones_z = np.ones((1, grid_num, grid_num))
    r_3d = base[np.newaxis, np.newaxis, :] * ones_x
    g_3d = base[np.newaxis, :, np.newaxis] * ones_y
    b_3d = base[:, np.newaxis, np.newaxis] * ones_z
    r_3d = r_3d.flatten()
    g_3d = g_3d.flatten()
    b_3d = b_3d.flatten()

    return np.dstack((r_3d, g_3d, b_3d))


class MeasureExecTime():
    def __init__(self):
        self.clear_buf()

    def clear_buf(self):
        self.st_time = 0.0
        self.lap_st = 0.0
        self.ed_time = 00

    def start(self):
        self.st_time = time.time()
        self.lap_st = self.st_time

    def lap(self, msg="", rate=1.0, show_str=True):
        current = time.time()
        if show_str:
            out_str = f"{msg}, lap {(current - self.lap_st)*rate:.4f} [s],"
            out_str += f" elapsed {(current - MODULE_IMPORT_TIME):.4f} [s]"
            print(out_str)
        self.lap_st = current

    def end(self, show_str=True):
        current = time.time()
        if show_str:
            out_str = f"total {current - self.st_time:.4f} [s], "
            out_str += f" elapsed {(current - MODULE_IMPORT_TIME):.4f} [s]"
            print(out_str)
        self.clear_buf()


if __name__ == '__main__':
    # print(equal_devision(10000, 1001))
    data = (100, 100, 3)
    print(is_img_shape(data))

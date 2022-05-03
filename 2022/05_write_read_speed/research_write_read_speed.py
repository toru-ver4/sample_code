# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.io import write_image, read_image
from colour.io.image import ImageAttribute_Specification

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
from common import MeasureExecTime

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2022 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def check_png_write_read_speed():
    """
    Measure write speed.
    """
    bit_options = [
        ImageAttribute_Specification("png:compressionLevel", x)
        for x in range(10)]
    met = MeasureExecTime()
    # width = 3840
    # height = 2160
    met.start()
    img = read_image("./img/src_4k.png")
    img = np.dstack([img, img, img])
    met.lap("created image")
    for idx, bit_option in enumerate(bit_options):
        met.lap(show_str=False)
        write_image(img, "./img_tmp/4k.png", 'uint16', attributes=[bit_option])
        met.lap(f"wrote image comp-lv: {idx}")
    met.end()


def check_tiff_write_read_speed():
    """
    Measure write speed.
    """
    bit_options = [
        ImageAttribute_Specification("tiff:zipquality", x)
        for x in range(1, 10)]
    met = MeasureExecTime()
    # width = 3840
    # height = 2160
    met.start()
    img = read_image("./img/src_4k.png")
    img = np.dstack([img, img, img])
    met.lap("created image")
    for idx, bit_option in enumerate(bit_options):
        met.lap(show_str=False)
        write_image(img, "./img_tmp/4k.tif", 'uint16', attributes=[bit_option])
        met.lap(f"wrote image comp-lv: {idx+1}")
    met.end()


def tr_tiff_write_using_multiprocessing_open_once_core(args):
    tiff_write_using_multiprocessing_open_once_core(**args)


def tiff_write_using_multiprocessing_open_once_core(idx, img):
    met = MeasureExecTime()
    met.start()
    fname = f"./img_tmp/idx_{idx:04d}.tif"
    met.lap(show_str=False)
    write_image(img, fname, 'uint16')
    met.lap(msg=f"idx={idx:04d}")
    met.end(show_str=False)


def tiff_write_using_multiprocessing_open_once():
    total_frame = 32
    total_process_num = total_frame
    block_process_num = int(cpu_count() * 1.0)
    block_num = int(round(total_process_num / block_process_num + 0.5))
    img = read_image("./img/src_4k.png")
    img = np.dstack([img, img, img])
    met = MeasureExecTime()
    met.start()
    for b_idx in range(block_num):
        args = []
        for p_idx in range(block_process_num):
            l_idx = b_idx * block_process_num + p_idx              # User
            print(f"b_idx={b_idx}, p_idx={p_idx}, l_idx={l_idx}")  # User
            if l_idx >= total_process_num:                         # User
                break
            d = dict(idx=l_idx, img=img.copy())
            args.append(d)
        #     render_and_save(**d)
        #     break
        # break
        with Pool(block_process_num) as pool:
            pool.map(tr_tiff_write_using_multiprocessing_open_once_core, args)
    met.end()


def debug_func():
    # check_png_write_read_speed()
    # check_tiff_write_read_speed()
    tiff_write_using_multiprocessing_open_once()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()

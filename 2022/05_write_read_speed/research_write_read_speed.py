# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from multiprocessing import Pool, cpu_count

# import third-party libraries
import numpy as np
from colour.io import write_image
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


def debug_func():
    check_png_write_read_speed()


def check_png_write_read_speed():
    """
    Measure write speed.
    """
    bit_options = [
        ImageAttribute_Specification("png:compressionLevel", x)
        for x in range(10)]
    met = MeasureExecTime()
    width = 3840
    height = 2160
    met.start()
    img = np.zeros((height, width, 3))
    met.lap("created image")
    for idx, bit_option in enumerate(bit_options):
        met.lap(show_str=False)
        write_image(img, "./img_tmp/4k.png", 'uint16', attributes=[bit_option])
        met.lap(f"wrote image comp-lv: {idx}")
    met.end()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_func()
    # main_func()

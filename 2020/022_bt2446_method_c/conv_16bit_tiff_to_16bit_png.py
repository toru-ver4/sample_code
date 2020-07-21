# -*- coding: utf-8 -*-
"""
convert from 16bit tiff to 16bit png.
======================================

"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2020 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


if __name__ == '__main__':
    src_name_list = [
        "./img/low.tif", "./img/high.tif", "./img/Chimera00216030.tif",
        "./img/Chimera00216111.tif"]
    dst_name_list = [
        "./img/low.png", "./img/high.png", "./img/middle.png",
        "./img/dark.png"]

    for src, dst in zip(src_name_list, dst_name_list):
        img = cv2.imread(src, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        cv2.imwrite(dst, img)

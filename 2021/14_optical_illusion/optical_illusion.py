# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
import cv2

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


fg_color = tf.eotf(np.array([0, 183, 255]) / 255, tf.SRGB)
bg_color = tf.eotf(np.array([199, 169, 0]) / 255, tf.SRGB)


def create_bg_dot_pattern(srgb_cv=bg_color):
    width = 640
    height = 640
    rate = 16
    dot_img_width = width // rate
    dot_img_height = height // rate

    mask = np.random.randint(0, 2, (dot_img_height, dot_img_width, 1))
    temp_img = mask
    dot_img = np.dstack([temp_img, temp_img, temp_img])
    img = cv2.resize(
        dot_img, (width, height), interpolation=cv2.INTER_NEAREST)

    center = (width//2, height//2)
    radius = int(width / 3)

    img = img * srgb_cv
    img = cv2.circle(img, center, radius, bg_color, -1)

    radius2 = int(width / 5)
    img = cv2.circle(img, center, radius2, fg_color, -1)

    tpg.img_wirte_float_as_16bit_int(
        "./img/test.png", tf.oetf(np.clip(img, 0, 1), tf.SRGB))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_bg_dot_pattern()

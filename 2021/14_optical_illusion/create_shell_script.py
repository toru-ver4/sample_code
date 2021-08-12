# -*- coding: utf-8 -*-
"""

```
"""

# import standard libraries
import os

# import third-party libraries
import numpy as np
from colour.utilities import tstack

# import my libraries
import test_pattern_generator2 as tpg
from optical_illusion import FPS, CYCLE_NUM, CYCLE_SEC

FRAME_NUM = FPS * CYCLE_NUM * CYCLE_SEC

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2021 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def calc_st_ed_frame_ed_for_block(len_list):
    """
    Examples
    --------
    >>> len_list = [4, 3, 5, 6, 2, 4, 1]
    >>> st_ed_list = calc_st_ed_frame_ed_for_blodk(len_list)
    [[  0.   4.]
     [  4.   7.]
     [  7.  12.]
     [ 12.  18.]
     [ 18.  20.]
     [ 20.  24.]
     [ 24.  25.]]
    """
    length = len(len_list)
    st_list = [0] + [sum(len_list[:idx]) for idx in range(1, length)]
    ed_list = [len_list[0]]\
        + [st_list[idx] + len_list[idx] for idx in range(1, length)]

    frame_st_ed_list = tstack(
        [np.uint16(st_list), np.int16(ed_list)])

    return frame_st_ed_list


def main_func():
    fname = "./multi_color_sample.sh"
    process_num = 8
    len_list = tpg.equal_devision(FRAME_NUM, process_num)
    st_ed_frame = calc_st_ed_frame_ed_for_block(len_list)
    src_file_name = "./optical_illusion.py"

    with open(fname, 'w') as f:
        f.write("#!/bin/bash\n")
        for st_ed in st_ed_frame:
            st_frame_idx = int(st_ed[0])
            ed_frame_idx = int(st_ed[1])
            buf = f"python {src_file_name} "
            buf += f"{st_frame_idx} {ed_frame_idx:d} &"
            f.write(buf + "\n")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main_func()

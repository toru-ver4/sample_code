# -*- coding: utf-8 -*-

# import standard libraries
import os
from pathlib import Path

# import third-party libraries
import numpy as np

# import my libraries
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tpg.add_clli_chunk_to_png(
        src_png_name="./debug/Photo_page_export_without_icc.png",
        dst_png_name="./debug/Photo_page_export_without_icc_with_clli.png",
        color_gamut=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        max_fall=203,
        max_cll=203
    )

# -*- coding: utf-8 -*-

# import standard libraries
import os

# import third-party libraries
from colour import (
    LUT1D,
    write_LUT
)

# import my libraries
import transfer_functions as tf

def create_st2084_80_to_100_nits_lut():
    fname_cube = "./lut/st2084_luminance_gain_1.25.cube"
    number_of_sample = 4096
    windows_sdr_nits = 80
    reference_sdr_nits = 100
    linear = tf.eotf_to_luminance(LUT1D.linear_table(size=number_of_sample), tf.ST2084)
    st2084_with_gain = tf.oetf_from_luminance(linear * (reference_sdr_nits/windows_sdr_nits), tf.ST2084)
    lut1d = LUT1D(st2084_with_gain, "80 nits to 100 nits conversion for DaVinci Resolve")
    write_LUT(lut1d, fname_cube)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_st2084_80_to_100_nits_lut()

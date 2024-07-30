# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os
from pathlib import Path
import subprocess

# import third-party libraries
import numpy as np
import test_pattern_generator2 as tpg
import transfer_functions as tf
import color_space as cs

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2024 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def encode_HDR_TP():
    src_file_list = [
        "./debug/src_tp/SMPTE ST2084_ITU-R BT.2020_D65_1920x1080_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_ITU-R BT.2020_D65_3840x2160_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_P3-D65_D65_1920x1080_rev07_type1.png",
        "./debug/src_tp/SMPTE ST2084_P3-D65_D65_3840x2160_rev07_type1.png",
    ]

    for png_fname in src_file_list:
        print(png_fname)
        if png_fname.find("BT.2020") > -1:
            color_space_name = cs.BT2020
        elif png_fname.find("P3-D65") > -1:
            color_space_name = cs.P3_D65
        else:
            raise ValueError("filename dosn't contain colorimetry infomation.")

        pp = Path(png_fname)
        parent = str(pp.parent)
        avif_fname = "./" + parent + "/" + pp.stem + ".avif"
        tpg.png_to_avif(
            png_fname=png_fname,
            avif_fname=avif_fname,
            bit_depth=10,
            color_space_name=color_space_name,
            transfer_characteristics=tf.ST2084,
            cll=None,
            pall=None
        )


def create_specific_luminance_small_patch(
        luminance=203, size=256, cll=10000, pall=10000):
    target_cv = tf.oetf_from_luminance(luminance, tf.ST2084)
    img = np.ones((size, size, 3)) * target_cv

    png_fname = f"./debug/src_tp/patch_{luminance:05d}-nits_{size}px.png"

    print(png_fname)
    tpg.img_wirte_float_as_16bit_int(png_fname, img)

    pp = Path(png_fname)
    parent = str(pp.parent)
    if (cll is None) or (pall is None):
        avif_fname =\
            "./" + parent + "/" + pp.stem + "_cll-auto" + ".avif"
    else:
        avif_fname =\
            "./" + parent + "/" + pp.stem + f"_cll-{cll}-{pall}-nits" + ".avif"

    tpg.png_to_avif(
        png_fname=png_fname,
        avif_fname=avif_fname,
        bit_depth=12,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=cll,
        pall=pall,
    )

    # cmd = [
    #     "avifenc", png_fname,
    #     "-d", "10",
    #     "-y", "444",
    #     "--cicp", "9/16/0",
    #     "-r", "full",
    #     "--clli", f"{cll},{pall}",
    #     "--lossless",
    #     "--ignore-exif",
    #     avif_fname
    # ]
    # print(" ".join(cmd))
    # subprocess.run(cmd)


def create_specific_luminance_small_patch_all(cll_luminance=None):
    lumiannce_list = [
        60, 80, 100, 120, 140, 160, 180,
        200, 203, 204, 220, 240, 260, 280, 300, 320,
        1000, 10000]
    patch_size = 64
    if cll_luminance is None:
        cll = None
        pall = None
    else:
        cll = cll_luminance
        pall = cll_luminance
    for lumiannce in lumiannce_list:
        create_specific_luminance_small_patch(
            luminance=lumiannce, size=patch_size,
            cll=cll, pall=pall
        )


def encode_av1_tp_video(pix_fmt='yuv444p12le'):
    in_fname = "./debug/src_tp/step_ramp_step_65_wo_alpha.png"
    out_fname = f"./debug/src_tp/step_ramp_step_65_{pix_fmt}.mp4"
    fps = 24
    cmd = "ffmpeg"
    ops = [
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-loop', '1',
        '-r', str(fps),
        '-i', in_fname,
        "-an",
        '-t', "10",
        '-c:v', 'libaom-av1',
        '-movflags', 'write_colr',
        '-pix_fmt', pix_fmt,
        '-crf', '0',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(out_fname), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def trim_3840x960():
    in_fname = "./debug/src_tp/SMPTE ST2084_ITU-R BT.2020_D65_3840x2160_rev07_type1.png"
    crop_fname = "./debug/src_tp/HDR_TP_3840x960.png"
    img = tpg.img_read(in_fname)
    img_crop = img[2158-960:2158]
    tpg.img_write(crop_fname, img_crop)

    pp = Path(crop_fname)
    parent = str(pp.parent)
    avif_fname = "./" + parent + "/" + pp.stem + ".avif"

    tpg.png_to_avif(
        png_fname=crop_fname,
        avif_fname=avif_fname,
        bit_depth=10,
        color_space_name=cs.BT2020,
        transfer_characteristics=tf.ST2084,
        cll=None,
        pall=None
    )


def _debug_calc_sdr_color_code():
    num_of_patch = 10
    max_8bit = 255
    cv = np.round(np.linspace(0, 1, num_of_patch) * max_8bit).astype(np.uint8)
    for x in cv:
        print(f"#{x:02X}{x:02X}{x:02X}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_specific_luminance_small_patch_all(cll_luminance=None)
    # create_specific_luminance_small_patch_all(cll_luminance=1000)
    # create_specific_luminance_small_patch_all(cll_luminance=10000)
    # encode_HDR_TP()
    # encode_av1_tp_video(pix_fmt="yuv420p10le")
    # trim_3840x960()
    _debug_calc_sdr_color_code()

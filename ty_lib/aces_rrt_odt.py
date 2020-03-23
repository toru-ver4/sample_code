# -*- coding: utf-8 -*-
"""
ACES の Output Transfrom について少し調べる
===============================================

Description.

"""

# import standard libraries
import os
from subprocess import run

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


def make_dst_name(src_name, suffix_list, dst_ext=".tiff"):
    """
    Examples
    --------
    >>> src_name = "./src_709_gamut.exr"
    >>> suffix_list = ["./ctl/rrt/RRT.ctl",
                       "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    >>> make_dst_name(src_name, suffix_list)
    ./src_709_gamut_RRT_ODT.Academy.sRGB_100nits_dim.exr
    """
    src_root, src_ext = os.path.splitext(src_name)
    suffix_bare_list = [os.path.basename(os.path.splitext(x)[0])
                        for x in suffix_list]
    out_name = src_root + "_" + "_".join(suffix_bare_list) + dst_ext

    return out_name


def apply_ctl_to_exr_image(
        img_list, ctl_list, ctl_module_path, out_ext=".tiff"):
    """
    Examples
    --------
    >>> img_list = ["./src_709_gamut.exr", "./src_2020_gamut.exr",
                    "./src_ap1.exr", "./src_ap0.exr"]
    >>> ctl_list = ["./ctl/rrt/RRT.ctl",
                    "./ctl/odt/sRGB/ODT.Academy.sRGB_100nits_dim.ctl"]
    >>> out_img_name_list = apply_ctl_to_exr_image(img_list, ctl_list)
    >>> print(out_img_name_list)
    ['./src_709_gamut_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_2020_gamut_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_ap1_RRT_ODT.Academy.sRGB_100nits_dim.tiff',
     './src_ap0_RRT_ODT.Academy.sRGB_100nits_dim.tiff']
    """
    cmd_base = "ctlrender -force "
    if len(ctl_list) < 2:
        ctl_ops = ["-ctl {}".format(ctl_list[0])]
    else:
        ctl_ops = ["-ctl {}".format(x) for x in ctl_list]
    if out_ext == ".tiff":
        format_ops = "-format tiff16"
    else:
        format_ops = "-format exr32"
    cmd_base += " ".join(ctl_ops) + " " + format_ops
    cmd_list = ["{} {} {}".format(cmd_base,
                                  src,
                                  make_dst_name(src, ctl_list, out_ext))
                for src in img_list]
    for cmd in cmd_list:
        print(cmd)
        os.environ['CTL_MODULE_PATH'] = ctl_module_path
        run(cmd.split(" "))

    return [make_dst_name(src, ctl_list, out_ext) for src in img_list]


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

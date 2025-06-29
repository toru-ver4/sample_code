# -*- coding: utf-8 -*-

# import standard libraries
import os

# import third-party libraries
import rawpy
import imageio

# import my libraries


def raw_to_tiff(raw_file, output_file):
    with rawpy.imread(raw_file) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1.0, 1.0),
            output_color=rawpy.ColorSpace.raw,  
        )
    imageio.imwrite(output_file, rgb)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    raw_to_tiff(
        raw_file="./camera_raw/DSC02079.ARW",
        output_file="./camera_raw/kanazawa_castle_01.tiff"
    )

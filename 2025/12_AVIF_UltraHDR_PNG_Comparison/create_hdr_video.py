from pathlib import Path
import os
import sys
import subprocess

import numpy as np

import test_pattern_generator2 as tpg
import color_space as cs


def xy_chromaticities_to_cta861_int(primaries: np.ndarray):
    return np.round(primaries / 0.00002).astype(np.uint32)


def calc_master_display_str(
        color_space_str: str = cs.BT2020,
        white_point: np.ndarray = cs.D65,
        min_lumiannce: float = 0.0001,
        max_lumiannce: float = 1000
):
    xy_primaries = cs.get_primaries(color_space_name=color_space_str)
    # RGB to GBR
    xy_primaries = np.roll(xy_primaries, shift=-1, axis=0)

    # add white
    xy_primaries = np.vstack([xy_primaries, white_point])
    int_primaries = xy_chromaticities_to_cta861_int(primaries=xy_primaries)
    print(xy_primaries)
    print(int_primaries)

    int_luminance = np.round(np.array([max_lumiannce, min_lumiannce]) / 0.0001).astype(np.uint32)

    master_display_str = "master-display=" \
        + f"G({int_primaries[0, 0]},{int_primaries[0, 1]})"\
        + f"B({int_primaries[1, 0]},{int_primaries[1, 1]})"\
        + f"R({int_primaries[2, 0]},{int_primaries[2, 1]})"\
        + f"WP({int_primaries[3, 0]},{int_primaries[3, 1]})"\
        + f"L({int_luminance[0]},{int_luminance[1]})"
    
    return master_display_str


def encode_hdr10_using_ffmpeg(
    mastering_display_color_space=cs.BT2020,
    mastering_display_white_point=cs.D65,
    mastering_display_min_luminance=0.0,
    mastering_display_max_luminance=1000,
    max_fall=10000,
    max_cll=10000,
    dst_mp4_name="./video/test.mp4"
):
    cmd = "ffmpeg"
    src_png_name = "./src_img/1920x1080_ST2084_Rec.2020.png"
    dst_bitstream_name = str(Path(dst_mp4_name).with_suffix(".h265"))
    length_sec = 10
    mastering_display_str = calc_master_display_str(
        color_space_str=mastering_display_color_space,
        white_point=mastering_display_white_point,
        min_lumiannce=mastering_display_min_luminance,
        max_lumiannce=mastering_display_max_luminance
    )
    max_fall_str = f"max-cll={max_cll},{max_fall}"
    if mastering_display_color_space == cs.BT2020:
        primary_str = "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"
    elif mastering_display_color_space == cs.P3_D65:
        primary_str = "colorprim=smpte432:transfer=smpte2084:colormatrix=bt709"
    elif mastering_display_color_space == cs.BT709:
        primary_str = "colorprim=bt709:transfer=smpte2084:colormatrix=bt709"
    else:
        raise ValueError("invalid mastering_display_color_space parameter")
    x265_params = f'"{primary_str}:{mastering_display_str}:{max_fall_str}"'

    ops = [
        '-loop', '1',
        '-framerate', '24',
        '-t', f"{length_sec}",
        '-i', src_png_name,
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-c:v', 'libx265',
        '-x265-params', x265_params,
        '-pix_fmt', 'yuv420p10le',
        '-qp', '0',
        '-movflags', '+write_colr',
        '-tag:v', 'hvc1',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        str(dst_mp4_name), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # extract bitstream data
    ops = [
        "-i", dst_mp4_name,
        "-c", "copy",
        "-an",
        "-bsf", "hevc_mp4toannexb",
        dst_bitstream_name
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def debug():
    pass
    # display_str = calc_master_display_str(
    #     color_space_str=cs.P3_D65,
    #     white_point=cs.D65,
    #     min_lumiannce=0.0,
    #     max_lumiannce=1000
    # )
    # print(display_str)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    encode_hdr10_using_ffmpeg(
        mastering_display_color_space=cs.BT2020,
        mastering_display_white_point=cs.D65,
        mastering_display_min_luminance=0.0,
        mastering_display_max_luminance=10000,
        max_fall=10000,
        max_cll=10000,
        dst_mp4_name="./video/test_10000-nits.mp4"
    )

    encode_hdr10_using_ffmpeg(
        mastering_display_color_space=cs.BT2020,
        mastering_display_white_point=cs.D65,
        mastering_display_min_luminance=0.0,
        mastering_display_max_luminance=0.0,
        max_fall=0,
        max_cll=0,
        dst_mp4_name="./video/test_00000-nits.mp4"
    )

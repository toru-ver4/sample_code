from pathlib import Path
import os
import subprocess
from itertools import product

import numpy as np

import test_pattern_generator2 as tpg
import color_space as cs

MDCV_PRIMARIES_LIST = [cs.BT709, cs.BT2020, None]
MDCV_LUMINANCE_LIST = [100, 10000, None]
CLLI_LUMINANCE_LIST = [100, 10000, None]

# MDCV_PRIMARIES_LIST = [cs.BT709]
# MDCV_LUMINANCE_LIST = [10000]
# CLLI_LUMINANCE_LIST = [None]

KIND_AV1 = "av1"
KIND_HEVC = "hevc"
KIND_AVIF = "avif"
KIND_PNG = "png"


def xy_chromaticities_to_cta861_int(primaries: np.ndarray):
    return np.round(primaries / 0.00002).astype(np.uint32)


def calc_master_display_str_hevc(
        color_space_str: str = cs.BT2020,
        white_point: np.ndarray = cs.D65,
        min_lumiannce: float = 0.0001,
        max_lumiannce: float = 1000
):
    if color_space_str is None and max_lumiannce is None:
        return None

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


def calc_master_display_str_av1(
        color_space_str: str = cs.BT2020,
        white_point: np.ndarray = cs.D65,
        min_lumiannce: float = 0.0001,
        max_lumiannce: float = 1000
):
    if (color_space_str is None) and (max_lumiannce is None):
        return None

    xy_primaries = cs.get_primaries(color_space_name=color_space_str)
    # RGB to GBR
    xy_primaries = np.roll(xy_primaries, shift=-1, axis=0)

    # add white
    xy_primaries = np.vstack([xy_primaries, white_point])
    print(xy_primaries)

    master_display_str = "mastering-display="\
        + f"G({xy_primaries[0, 0]},{xy_primaries[0, 1]})"\
        + f"B({xy_primaries[1, 0]},{xy_primaries[1, 1]})"\
        + f"R({xy_primaries[2, 0]},{xy_primaries[2, 1]})"\
        + f"WP({xy_primaries[3, 0]},{xy_primaries[3, 1]})"\
        + f"L({max_lumiannce},{min_lumiannce})"
    
    return master_display_str


def encode_hdr10_using_ffmpeg_mp4box_hevc_core(
    mastering_display_color_space=cs.BT2020,
    mastering_display_white_point=cs.D65,
    mastering_display_min_luminance=0.0,
    mastering_display_max_luminance=1000,
    max_fall=10000,
    max_cll=10000,
    framerate=24,
    dst_fname_without_ext="./video/test"
):
    def add_x265_params(param):
        return f":{param}" if param is not None else ""

    cmd = "ffmpeg"
    src_png_name = "./src_img/1920x1080_ST2084_Rec.2020.png"
    dst_mp4_name = dst_fname_without_ext + ".mp4"
    dst_mov_name = str(Path(dst_mp4_name).with_suffix(".mov"))
    dst_bitstream_name = str(Path(dst_mp4_name).with_suffix(".h265"))
    length_sec = 10
    mastering_display_str = calc_master_display_str_hevc(
        color_space_str=mastering_display_color_space,
        white_point=mastering_display_white_point,
        min_lumiannce=mastering_display_min_luminance,
        max_lumiannce=mastering_display_max_luminance
    )
    max_fall_str = f"max-cll={max_cll},{max_fall}" if max_fall is not None else "no-cll=1"
    cicp_str = "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"
    x265_params = f"{cicp_str}"
    x265_params += add_x265_params(mastering_display_str)
    x265_params += add_x265_params(max_fall_str)

    # create bitstream data
    ops = [
        '-hide_banner',
        '-loop', '1',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-framerate', f'{framerate}',
        '-t', f"{length_sec}",
        '-i', src_png_name,
        '-c:v', 'libx265',
        '-x265-params', x265_params,
        '-pix_fmt', 'yuv420p10le',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-qp', '0',
        '-f', 'hevc',
        str(dst_bitstream_name), '-y',
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # create mp4 container using MP4Box
    cmd = "MP4Box"
    param_str = f"{dst_bitstream_name}:fmt=hevc:fps={framerate}"
    ops = [
        "-new",
        "-add",
        param_str,
        dst_mp4_name
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # create mov container using MP4Box
    cmd = "MP4Box"
    param_str = f"{dst_bitstream_name}:fmt=hevc:fps={framerate}"
    ops = [
        "-new",
        "-add",
        param_str,
        dst_mov_name
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_hdr10_using_ffmpeg_mp4box_av1_core(
    mastering_display_color_space=cs.BT2020,
    mastering_display_white_point=cs.D65,
    mastering_display_min_luminance=0.0,
    mastering_display_max_luminance=1000,
    max_fall=10000,
    max_cll=10000,
    framerate=24,
    dst_fname_without_ext="./video/test",    
):
    """
    References:
    - https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/master/Docs/Parameters.md#2-av1-metadata
    """
    def add_svtav1_params(param):
        return f":{param}" if param is not None else ""

    cmd = "ffmpeg"
    src_png_name = "./src_img/1920x1080_ST2084_Rec.2020.png"
    dst_mp4_name= dst_fname_without_ext + ".mp4"
    dst_bitstream_name = str(Path(dst_mp4_name).with_suffix(".obu"))
    dst_mov_name = str(Path(dst_mp4_name).with_suffix(".mov"))
    length_sec = 10

    cicp_str = "color-primaries=9:transfer-characteristics=16:matrix-coefficients=9:color-range=0"

    mastering_display_str = calc_master_display_str_av1(
        color_space_str=mastering_display_color_space,
        white_point=mastering_display_white_point,
        min_lumiannce=mastering_display_min_luminance,
        max_lumiannce=mastering_display_max_luminance
    )

    content_light_str = f"content-light={max_cll},{max_fall}" if max_fall is not None else None

    # Note: do not include shell quotes here; pass the raw string as one argv token.
    svtav1_params = "crf=1"
    svtav1_params += add_svtav1_params(mastering_display_str)
    svtav1_params += add_svtav1_params(content_light_str)
    svtav1_params += add_svtav1_params(cicp_str)

    ops = [
        '-hide_banner',
        '-loop', '1',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-framerate', f'{framerate}',
        '-t', f"{length_sec}",
        '-i', src_png_name,
        '-c:v', 'libsvtav1',
        '-svtav1-params', svtav1_params,
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-pix_fmt', 'yuv420p10le',
        '-f', 'obu',
        str(dst_bitstream_name), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # create mp4 container using MP4Box
    cmd = "MP4Box"
    param_str = f"{dst_bitstream_name}:fmt=obu:fps={framerate}"
    ops = [
        "-new",
        "-add",
        param_str,
        dst_mp4_name
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # create mov container using MP4Box
    cmd = "MP4Box"
    param_str = f"{dst_bitstream_name}:fmt=obu:fps={framerate}"
    ops = [
        "-new",
        "-add",
        param_str,
        dst_mov_name
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def png_to_avif(
        png_fname: str,
        avif_fname: str,
        bit_depth: int = 10,
        cicp: list[int] = [1, 1, 1],
        cll: int | None = 0,
        pall: int | None = 0
):
    cmd = [
        "avifenc", png_fname,
        "-d", f"{bit_depth}",
        "--cicp", f"{cicp[0]}/{cicp[1]}/{cicp[2]}",
        "-c", "aom",
        "--lossless",
        "--ignore-exif",
        avif_fname
    ]
    if (cll is None) or (pall is None):
        pass
    else:
        cmd.insert(-2, "--clli")
        cmd.insert(-2, f"{cll},{pall}")

    print(" ".join(cmd))
    subprocess.run(cmd)


def extract_obu_from_avif_using_ffmpeg(avif_fname):
    output_fname = str(Path(avif_fname).with_suffix(".obu"))
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", avif_fname,
        "-map", "0:v:0",
        "-c", "copy",
        "-f", "obu",
        output_fname,
        '-y'
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


def encode_hdr10_using_avifenc_avif_core(
        cll_luminance: int,
        pall_luminance: int,
        dst_fname_without_ext: str):
    src_hdr_img_fname = "./src_img/1920x1080_ST2084_Rec.2020.png"

    # AVIF
    dst_avif_fname = dst_fname_without_ext + ".avif"
    png_to_avif(
        png_fname=src_hdr_img_fname,
        avif_fname=dst_avif_fname,
        bit_depth=10,
        cicp=[9, 16, 0],
        cll=cll_luminance,
        pall=pall_luminance
    )

    extract_obu_from_avif_using_ffmpeg(avif_fname=dst_avif_fname)


def make_media_file_name_without_ext(
        kind: str,
        suffix: str | None,
        mdcv_primaries: str | None,
        mdcv_luminance: int | None,
        clli_luminance: int):
    def convert_none_str_if_none(x):
        return x if x is not None else "None"
    suffix_str = suffix if suffix is not None else ""
    mdcv_primaries_str = convert_none_str_if_none(mdcv_primaries)
    mdcv_luminance_str = convert_none_str_if_none(mdcv_luminance)
    clli_luminance_str = convert_none_str_if_none(clli_luminance)
    dst_dir = "./hdr_media"
    file_name = f"{dst_dir}/{kind}_mdcv-p-{mdcv_primaries_str}_mdcv-l-{mdcv_luminance_str}_"
    file_name += f"clli-{clli_luminance_str}{suffix_str}"

    return file_name


def encode_hdr10_using_ffmpeg_mp4box_hevc(
        mdcv_primaries_list: list[str],
        mdcv_luminance_list: list[int],
        clli_luminance_list: list[int]):
    for mdcv_primaries, mdcv_luminance, clli_luminance\
        in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list):
        print(mdcv_primaries, mdcv_luminance, clli_luminance)

        if (mdcv_primaries is None) and (mdcv_luminance is not None):
            continue
        if (mdcv_primaries is not None) and (mdcv_luminance is None):
            continue

        file_name_without_ext = make_media_file_name_without_ext(
            kind=KIND_HEVC,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )

        encode_hdr10_using_ffmpeg_mp4box_hevc_core(
            mastering_display_color_space=mdcv_primaries,
            mastering_display_white_point=cs.D65,
            mastering_display_min_luminance=0,
            mastering_display_max_luminance=mdcv_luminance,
            max_fall=clli_luminance,
            max_cll=clli_luminance,
            framerate=24,
            dst_fname_without_ext=file_name_without_ext
        )


def encode_hdr10_using_ffmpeg_mp4box_av1(
        mdcv_primaries_list: list[str],
        mdcv_luminance_list: list[int],
        clli_luminance_list: list[int]):
    for mdcv_primaries, mdcv_luminance, clli_luminance\
        in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list):
        print(mdcv_primaries, mdcv_luminance, clli_luminance)

        if (mdcv_primaries is None) and (mdcv_luminance is not None):
            continue
        if (mdcv_primaries is not None) and (mdcv_luminance is None):
            continue

        file_name_without_ext = make_media_file_name_without_ext(
            kind=KIND_AV1,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )

        encode_hdr10_using_ffmpeg_mp4box_av1_core(
            mastering_display_color_space=mdcv_primaries,
            mastering_display_white_point=cs.D65,
            mastering_display_min_luminance=0,
            mastering_display_max_luminance=mdcv_luminance,
            max_fall=clli_luminance,
            max_cll=clli_luminance,
            framerate=24,
            dst_fname_without_ext=file_name_without_ext
        )


def encode_hdr10_using_avifenc_avif(
        mdcv_primaries_list: list[str] | None,
        mdcv_luminance_list: list[int] | None,
        clli_luminance_list: list[int] | None):
    def default_none_list(x):
        return x if x is not None else [None]
    mdcv_primaries_list = default_none_list(mdcv_primaries_list)
    mdcv_luminance_list = default_none_list(mdcv_luminance_list)
    for mdcv_primaries, mdcv_luminance, clli_luminance\
        in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list):

        file_name_without_ext = make_media_file_name_without_ext(
            kind=KIND_AVIF,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )

        encode_hdr10_using_avifenc_avif_core(
            cll_luminance=clli_luminance,
            pall_luminance=clli_luminance,
            dst_fname_without_ext=file_name_without_ext
        )


def encode_hdr10_using_ffmpeg_png_core(
    mastering_display_color_space=cs.BT2020,
    mastering_display_white_point=cs.D65,
    mastering_display_min_luminance=0.0,
    mastering_display_max_luminance=1000,
    max_fall=10000,
    max_cll=10000,
    framerate=24,
    dst_fname_without_ext="./video/test",    
):
    def add_x265_params(param):
        return f":{param}" if param is not None else ""

    cmd = "ffmpeg"
    src_png_name = "./src_img/1920x1080_ST2084_Rec.2020.png"
    dst_png_name= dst_fname_without_ext + ".png"
    dst_bitstream_name = str(Path(dst_png_name).with_name(f"{Path(dst_png_name).stem}.h265"))

    cicp_str = "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited"

    mastering_display_str = calc_master_display_str_hevc(
        color_space_str=mastering_display_color_space,
        white_point=mastering_display_white_point,
        min_lumiannce=mastering_display_min_luminance,
        max_lumiannce=mastering_display_max_luminance
    )
    # Note: do not include shell quotes here; pass the raw string as one argv token.
    max_fall_str = f"max-cll={max_cll},{max_fall}" if max_fall is not None else "no-cll=1"
    x265_params = f"{cicp_str}"
    x265_params += add_x265_params(mastering_display_str)
    x265_params += add_x265_params(max_fall_str)

    ops = [
        '-hide_banner',
        '-loop', '1',
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-framerate', f'{framerate}',
        '-i', src_png_name,
        '-frames:v', "1",
        '-c:v', 'libx265',
        '-x265-params', x265_params,
        '-color_primaries', 'bt2020',
        '-color_trc', 'smpte2084',
        '-colorspace', 'bt2020nc',
        '-pix_fmt', 'yuv444p12le',
        '-qp', '0',
        '-f', 'hevc',
        str(dst_bitstream_name), '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)

    # convert to PNG
    ops = [
        '-hide_banner',
        '-f', 'hevc',
        '-i', dst_bitstream_name,
        '-frames:v', '1',
        '-update', '1',
        dst_png_name, '-y'
    ]
    args = [cmd] + ops
    print(" ".join(args))
    subprocess.run(args)


def encode_hdr10_using_ffmpeg_png(
        mdcv_primaries_list: list[str] | None,
        mdcv_luminance_list: list[int] | None,
        clli_luminance_list: list[int] | None):

    for mdcv_primaries, mdcv_luminance, clli_luminance\
        in product(mdcv_primaries_list, mdcv_luminance_list, clli_luminance_list):

        if (mdcv_primaries is None) and (mdcv_luminance is not None):
            continue
        if (mdcv_primaries is not None) and (mdcv_luminance is None):
            continue

        file_name_without_ext = make_media_file_name_without_ext(
            kind=KIND_PNG,
            suffix=None,
            mdcv_primaries=mdcv_primaries,
            mdcv_luminance=mdcv_luminance,
            clli_luminance=clli_luminance
        )
        print(file_name_without_ext)

        encode_hdr10_using_ffmpeg_png_core(
            mastering_display_color_space=mdcv_primaries,
            mastering_display_white_point=cs.D65,
            mastering_display_min_luminance=0.0,
            mastering_display_max_luminance=mdcv_luminance,
            max_fall=clli_luminance,
            max_cll=clli_luminance,
            framerate=24,
            dst_fname_without_ext=file_name_without_ext
        )


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
    mdcv_primaries_list = MDCV_PRIMARIES_LIST
    mdcv_luminance_list = MDCV_LUMINANCE_LIST
    clli_luminance_list = CLLI_LUMINANCE_LIST

    # encode_hdr10_using_ffmpeg_mp4box_hevc(
    #     mdcv_primaries_list=mdcv_primaries_list,
    #     mdcv_luminance_list=mdcv_luminance_list,
    #     clli_luminance_list=clli_luminance_list
    # )

    # encode_hdr10_using_ffmpeg_mp4box_av1(
    #     mdcv_primaries_list=mdcv_primaries_list,
    #     mdcv_luminance_list=mdcv_luminance_list,
    #     clli_luminance_list=clli_luminance_list
    # )

    # encode_hdr10_using_avifenc_avif(
    #     mdcv_primaries_list=None,  # not supported by avifenc
    #     mdcv_luminance_list=None,  # not supported by avifenc
    #     clli_luminance_list=clli_luminance_list
    # )

    encode_hdr10_using_ffmpeg_png(
        mdcv_primaries_list=mdcv_primaries_list,
        mdcv_luminance_list=mdcv_luminance_list,
        clli_luminance_list=clli_luminance_list,
    )

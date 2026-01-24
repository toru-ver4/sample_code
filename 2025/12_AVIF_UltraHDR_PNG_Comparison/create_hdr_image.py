from pathlib import Path
import os
import subprocess

import numpy as np
from OpenImageIO import (
    ImageSpec,
    ImageOutput,
    UINT16
)
from colour.io import read_image


def png_to_avif(
        png_fname: str,
        avif_fname: str,
        bit_depth: int = 10,
        cicp: list[int] = [1, 1, 1],
        cll: int = 0,
        pall: int = 0
):
    cmd = [
        "avifenc", png_fname,
        "-d", f"{bit_depth}",
        "--cicp", f"{cicp[0]}/{cicp[1]}/{cicp[2]}",
        "-c", "aom",
        "--clli", f"{cll},{pall}",
        "--lossless",
        "--ignore-exif",
        avif_fname
    ]
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


def main():
    src_hdr_img_fname = "./src_img/1920x1080_ST2084_Rec.2020.png"
    mdcv_luminance_param_list = [0, 10000]
    cll_pall_param_list = [0, 10000]

    # # AVIF, PNG
    # dst_png_fname = "./dst_img/ISO_HDR_PNG_vs_AVIF_UltraHDR.png"
    # img = read_image(path="./src_img/1920x1080_ST2084_Rec.2020.png")
    # save_png_using_oiio(img=img, output_fname=dst_png_fname, cicp=[9, 16, 0, 1])

    for cll_pall_param in cll_pall_param_list:
        # AVIF
        dst_avif_fname = f"./dst_img/ISO_HDR_AVIF_MDCV-None_CLLI-{cll_pall_param}.avif"
        png_to_avif(
            png_fname=src_hdr_img_fname,
            avif_fname=dst_avif_fname,
            bit_depth=10,
            cicp=[9, 16, 0],
            cll=cll_pall_param,
            pall=cll_pall_param
        )

    extract_obu_from_avif_using_ffmpeg(avif_fname=dst_avif_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_sdr_image_for_ultrahdr()
    main()

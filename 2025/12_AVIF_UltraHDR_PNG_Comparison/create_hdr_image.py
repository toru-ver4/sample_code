from pathlib import Path
import os
import sys
import subprocess

import numpy as np
from OpenImageIO import (
    ImageSpec,
    ImageOutput,
    UINT16
)
from colour.io import read_image

tp_module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../08_UltraHDR_part2/"))
sys.path.insert(0, tp_module_path)

from create_ultrahdr_image import (
    craete_files_for_ultrahdr_app_scenario4
)


def png_to_avif(
        png_fname: str,
        avif_fname: str,
        bit_depth: int = 10,
        cicp: list[int] = [1, 1, 0],
        cll: int = 0,
        pall: int = 0
):
    cmd = [
        "avifenc", png_fname,
        "-d", f"{bit_depth}",
        "--cicp", f"{cicp[0]}/{cicp[1]}/{cicp[2]}",
        "-r", "full",
        "--clli", f"{cll},{pall}",
        "--lossless",
        "--ignore-exif",
        avif_fname
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)


def save_png_using_oiio(img: np.ndarray, output_fname: str, cicp: list[int] | None = None):
    output = ImageOutput.create(filename=output_fname)
    yres, xres, channels = img.shape
    image_spec = ImageSpec(xres, yres, channels, UINT16)
    if cicp is not None:
        image_spec.attribute("CICP", "int[4]", cicp)
    output.open(filename=output_fname, spec=image_spec)
    output.write_image(img)
    output.close()


def main():
    src_hdr_img_fname = "./src_img/1920x1080_ST2084_Rec.2020.png"
    src_sdr_img_fname = "./src_img/ultrahdr_sdr_image.png"

    # # AVIF, PNG
    # dst_png_fname = "./dst_img/ISO_HDR_PNG_vs_AVIF_UltraHDR.png"
    # img = read_image(path="./src_img/1920x1080_ST2084_Rec.2020.png")
    # save_png_using_oiio(img=img, output_fname=dst_png_fname, cicp=[9, 16, 0, 1])

    # AVIF
    dst_avif_fname = "./dst_img/ISO_HDR_AVIF_vs_UltraHDR_PNG_ll-10000.avif"
    png_to_avif(
        png_fname=src_hdr_img_fname,
        avif_fname=dst_avif_fname,
        bit_depth=10,
        cicp=[9, 16, 0, 1],
        cll=10000,
        pall=10000
    )

    # # Ultra HDR
    # craete_files_for_ultrahdr_app_scenario4(
    #     hdr_fname=src_hdr_img_fname,
    #     sdr_fname=src_sdr_img_fname,
    #     hdr_capacity_max=1.0
    # )


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # create_sdr_image_for_ultrahdr()
    main()

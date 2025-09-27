import os
import subprocess
import shutil

import numpy as np
from OpenImageIO import (
    ImageSpec,
    ImageOutput,
    UINT16
)

import test_pattern_generator2 as tpg


def debug_oiio_png_hdr():
    input_fname = "./_debug/1920x1080_ST2084_Rec.2020.png"
    output_fname = "./_debug/1920x1080_ST2084_Rec.2020_with_CICP_oiio.png"

    img = tpg.img_read_as_float(filename=input_fname)
    output = ImageOutput.create(filename=output_fname)
    yres, xres, channels = img.shape
    image_spec = ImageSpec(xres, yres, channels, UINT16)
    image_spec.attribute("CICP", "int[4]", [9, 16, 0, 1])
    output.open(filename=output_fname, spec=image_spec)
    output.write_image(img)
    output.close()


def debug_oiio_avif_hdr():
    input_fname = "./_debug/P3D65_ST2084_without_CICP.png"
    output_fname = "./_debug/P3D65_ST2084_with_CICP_10-bit_oiio.avif"

    img = tpg.img_read_as_float(filename=input_fname)
    output = ImageOutput.create(filename=output_fname)
    yres, xres, channels = img.shape
    image_spec = ImageSpec(xres, yres, channels, UINT16)
    image_spec.attribute("CICP", "int[4]", [12, 16, 0, 1])
    image_spec.attribute("oiio:BitsPerSample", 10)
    image_spec.attribute("Compression", "avif:100")
    output.open(filename=output_fname, spec=image_spec)
    output.write_image(img)
    output.close()


def debug_oiio_heic_hdr():
    input_fname = "./_debug/P3D65_ST2084_without_CICP.png"
    output_fname = "./_debug/P3D65_ST2084_with_CICP_10-bit_oiio.heic"

    img = tpg.img_read_as_float(filename=input_fname)
    output = ImageOutput.create(filename=output_fname)
    yres, xres, channels = img.shape
    image_spec = ImageSpec(xres, yres, channels, UINT16)
    image_spec.attribute("CICP", "int[4]", [12, 16, 0, 1])
    image_spec.attribute("oiio:BitsPerSample", 10)
    image_spec.attribute("Compression", "heic:100")
    output.open(filename=output_fname, spec=image_spec)
    output.write_image(img)
    output.close()


def debug_png_cicp_editor():
    input_fname = "./_debug/P3D65_ST2084_without_CICP.png"
    intermediate_fname = "./_debug/P3D65_ST2084_intermediate.png"
    output_fname = "./_debug/P3D65_ST2084_with_CICP_pngcheck.png"

    # prepare file
    shutil.copy(input_fname, intermediate_fname)

    cmd = [
        "png_cicp_editor",
        "overwrite",
        "--preset", "p3-d65-pq",
        "--full",
        intermediate_fname
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)

    os.rename(intermediate_fname, output_fname)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    debug_oiio_png_hdr()
    # debug_oiio_avif_hdr()
    # debug_oiio_heic_hdr()
    # debug_png_cicp_editor()

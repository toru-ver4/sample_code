# -*- coding: utf-8 -*-
import sys
from imagecodecs import JPEGXR, imread

if not JPEGXR.available:
    print("JPEG XR is not supported.")
    sys.exit()

jpeg_xr_fname = "./Windows_HDR_Capture/600.jxr"
image = imread(jpeg_xr_fname)
print(f"image.shape = {image.shape}")
print(f"image.dtype = {image.dtype}")
print(f"image[1080, 1920] = {image[1080, 1920]}")

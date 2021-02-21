#!/bin/sh

cavif -i SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -o PQ-BT2020-D65_mc-bt709_crf-0.avif --color-primaries bt2020 --transfer-characteristics smpte2084 --matrix-coefficients bt709 --enable-full-color-range --pix-fmt yuv444 --bit-depth 12 --profile 2 --crf 0

cavif -i movie_SDR_1920x1080_24fps_0000.png -o bt709_yuv444p12le_full_crf-0.avif --color-primaries bt709 --transfer-characteristics bt709 --matrix-coefficients bt709 --enable-full-color-range --pix-fmt yuv444 --bit-depth 12 --profile 2 --crf 0

cavif -i movie_SDR_1920x1080_24fps_0000.png -o bt709_yuv420p12le_full_crf-0.avif --color-primaries bt709 --transfer-characteristics bt709 --matrix-coefficients bt709 --enable-full-color-range --pix-fmt yuv420 --bit-depth 12 --profile 2 --crf 0

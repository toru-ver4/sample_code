#!/bin/bash
echo " "
echo " "
echo "avifenc_aom ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 12 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif"
avifenc_aom ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 12 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif
echo " "
echo " "
echo "avifenc_rav1e ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 12 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif"
avifenc_rav1e ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 12 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif
echo " "
echo " "
echo "avifdec_aom ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-aom_12-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_aom ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-aom_12-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_dav1d ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-dav1d_12-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_dav1d ./avif/aom_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-dav1d_12-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_aom ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-aom_12-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_aom ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-aom_12-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_dav1d ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-dav1d_12-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_dav1d ./avif/rav1e_12-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-dav1d_12-bit_yuv444_cicp-9-16-9_full-range.png

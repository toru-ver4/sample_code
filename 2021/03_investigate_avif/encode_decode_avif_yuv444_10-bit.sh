#!/bin/bash
echo " "
echo " "
echo "avifenc_aom ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 10 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif"
avifenc_aom ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 10 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif
echo " "
echo " "
echo "avifenc_rav1e ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 10 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif"
avifenc_rav1e ./png/SMPTE_ST2084_ITU-R_BT.2020_D65_1920x1080_rev04_type1.png -d 10 -y 444 --cicp 9/16/9 -r full --min 0 --max 0 ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif
echo " "
echo " "
echo "avifdec_aom ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-aom_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_aom ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-aom_10-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_dav1d ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-dav1d_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_dav1d ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-dav1d_10-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_libgav1 ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-libgav1_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_libgav1 ./avif/aom_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/aom-libgav1_10-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_aom ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-aom_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_aom ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-aom_10-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_dav1d ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-dav1d_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_dav1d ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-dav1d_10-bit_yuv444_cicp-9-16-9_full-range.png
echo " "
echo " "
echo "avifdec_libgav1 ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-libgav1_10-bit_yuv444_cicp-9-16-9_full-range.png"
avifdec_libgav1 ./avif/rav1e_10-bit_yuv444_cicp-9-16-9_full-range.avif ./decoded_png/rav1e-libgav1_10-bit_yuv444_cicp-9-16-9_full-range.png

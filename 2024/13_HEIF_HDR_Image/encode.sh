#!/bin/sh
# heif-enc --verbose --quality 100 --bit-depth 12 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-9_12bit.heic
# heif-enc --verbose --quality 100 --bit-depth 12 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_12bit.heic
# heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-9_10bit.heic
# heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_10bit.heic
# heif-enc --verbose --quality 100 --bit-depth 10 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_no-cicp_10bit.heic
# convert ./dst_heif/Rec2100-PQ_no-cicp_10bit.heic -profile ./IMG_0589.icc ./dst_heif/Rec2100-PQ_no-cicp-icc_10bit.heic

# heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 12 --transfer_characteristic 16 --matrix_coefficients 1 --full_range_flag 1 ./src_png/P3D65-PQ.png -o ./dst_heif/P3D65-PQ_cicp_12-16-1_10bit.heic
# heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 1 --transfer_characteristic 1 --matrix_coefficients 1 --full_range_flag 1 ./src_png/Rec709_gm24.png -o ./dst_heif/Rec709_gm24_12-16-1_10bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 -p chroma=444 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_RGB444_10bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 -p chroma=444 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr444_10bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 -p chroma=422 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr422_10bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 -p chroma=420 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr420_10bit.heic

heif-enc --verbose --quality 100 --bit-depth 12 -p chroma=444 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_RGB444_12bit.heic
heif-enc --verbose --quality 100 --bit-depth 12 -p chroma=444 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr444_12bit.heic
heif-enc --verbose --quality 100 --bit-depth 12 -p chroma=422 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr422_12bit.heic
heif-enc --verbose --quality 100 --bit-depth 12 -p chroma=420 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_YCbCr420_12bit.heic

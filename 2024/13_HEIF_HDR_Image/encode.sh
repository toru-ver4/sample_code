#!/bin/sh
heif-enc --verbose --quality 100 --bit-depth 12 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-9_12bit.heic
heif-enc --verbose --quality 100 --bit-depth 12 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_12bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-9_10bit.heic
heif-enc --verbose --quality 100 --bit-depth 10 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 0 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ_cicp_9-16-0_10bit.heic

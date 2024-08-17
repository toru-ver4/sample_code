#!/bin/sh
ultrahdr_app -m 0 -p /mnt/data/src_img/src_rec2100-pq_rgba1010102.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -C 2 -c 0 -t 2 -R 1 -z /mnt/data/ultra_hdr_img/rec2100-pq_scenario_0.jpeg
ultrahdr_app -m 0 -p /mnt/data/src_img/src_rec2100-hlg_rgba1010102.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -C 2 -c 0 -t 1 -R 1 -z /mnt/data/ultra_hdr_img/rec2100-hlg_scenario_0.jpeg

# scenario 4
ultrahdr_app -m 0 -i /mnt/data/src_img/src_rec2020_srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_src_rec2100-pq-src_rec2020_srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 2 -R 1 -f /mnt/data/metadata/metadata_src_rec2100-pq-src_rec2020_srgb.cfg -z /mnt/data/ultra_hdr_img/rec2100-pq_scenario_4.jpeg
ultrahdr_app -m 0 -i /mnt/data/src_img/src_rec2020_srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_src_rec2100-hlg-src_rec2020_srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 1 -R 1 -f /mnt/data/metadata/metadata_src_rec2100-hlg-src_rec2020_srgb.cfg -z /mnt/data/ultra_hdr_img/rec2100-hlg_scenario_4.jpeg
# ultrahdr_app -m 0 -i /mnt/data/src_img/river_rec2020-srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_river_rec2100-pq_2k-river_rec2020-srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 2 -R 1 -f /mnt/data/metadata/metadata_river_rec2100-pq_2k-river_rec2020-srgb.cfg -z /mnt/data/ultra_hdr_img/river_rec2100-pq_scenario_4.jpeg
ultrahdr_app -m 0 -i /mnt/data/src_img/shiga_sdr_rec2020_srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_shiga_rec2100-pq-shiga_sdr_rec2020_srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 2 -R 1 -f /mnt/data/metadata/metadata_shiga_rec2100-pq-shiga_sdr_rec2020_srgb.cfg -z /mnt/data/ultra_hdr_img/shiga_rec2100-pq_scenario_4.jpeg

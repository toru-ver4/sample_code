# Docker build for Ultra HDR

## build

```powershell
docker build -t takuver4/ultrahdr:rev02 .
```

## docker push

```powershell
docker push takuver4/ultrahdr:rev02
```

## create p010 file

```powershell
ffmpeg -i src_rec2100-pq.png -pix_fmt rgba1010102le -f rawvideo src_rec2100-pq_rgba1010102.raw
ffmpeg -i src_rec709.png -pix_fmt rgba -f rawvideo src_rec709_rgba8888.raw
```

## run

```powershell
docker run -it -P --name ultrahdr_rev01 -v /Users/toru/Work/sample_code/Temporary/06_ultrahdr:/mnt/data --rm takuver4/ultrahdr:rev01

docker run -it -P --name ultrahdr_rev02 -v C:\Users\toruv\OneDrive\work\sample_code\2024\11_Create_Ultra_HDR_Image:/mnt/data --rm takuver4/ultrahdr:rev02
```

## create Ultra HDR file

```powershell
# scenario 0
ultrahdr_app -m 0 -p /mnt/data/src_img/src_rec2100-pq_rgba1010102.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -C 2 -c 0 -t 2 -R 1 -z /mnt/data/ultra_hdr_img/rec2100-pq_senario_0.jpeg
ultrahdr_app -m 0 -p /mnt/data/src_img/src_rec2100-hlg_rgba1010102.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -C 2 -c 0 -t 1 -R 1 -z /mnt/data/ultra_hdr_img/rec2100-hlg_senario_0.jpeg

# scenario 4
ultrahdr_app -m 0 -i /mnt/data/src_img/src_rec2020_srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_src_rec2100-pq-src_rec2020_srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 2 -R 1 -f /mnt/data/metadata/metadata_src_rec2100-pq-src_rec2020_srgb.cfg -z /mnt/data/ultra_hdr_img/rec2100-pq_scenario_4.jpeg
ultrahdr_app -m 0 -i /mnt/data/src_img/src_rec2020_srgb_8bit.jpeg -g /mnt/data/gain_map_img/gain_map_src_rec2100-hlg-src_rec2020_srgb.jpeg -q 100 -Q 100 -C 2 -c 2 -t 1 -R 1 -f /mnt/data/metadata/metadata_src_rec2100-hlg-src_rec2020_srgb.cfg -z /mnt/data/ultra_hdr_img/rec2100-hlg_scenario_4.jpeg
```

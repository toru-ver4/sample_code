# Docker build for Ultra HDR

## build

```powershell
docker build -t takuver4/ultrahdr:rev01 .
```

## docker push

```powershell
docker push takuver4/ultrahdr:rev01
```

## create p010 file

```powershell
ffmpeg -i src_rec2100-pq.png -pix_fmt rgba1010102le -f rawvideo src_rec2100-pq_rgba1010102.raw
ffmpeg -i src_rec709.png -pix_fmt rgba -f rawvideo src_rec709_rgba8888.raw
```

## run

```powershell
docker run -it -P --name ultrahdr_rev01 -v /Users/toru/Work/sample_code/Temporary/06_ultrahdr:/mnt/data --rm takuver4/ultrahdr:rev01

docker run -it -P --name ultrahdr_rev01 -v C:\Users\toruv\OneDrive\work\sample_code\2024\11_Create_Ultra_HDR_Image:/mnt/data --rm takuver4/ultrahdr:rev01
```

## create Ultra HDR file

```powershell
/opt/ultrahdr/ultrahdr_app -m 0 -p /mnt/data/src_rec2100-pq_rgba1010102.raw -y /mnt/data/src_rec709_rgba8888.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -b 3 -C 2 -c 0 -t 2 -R 1  -z /mnt/data/rec2100-pq_cat_rec709.jpeg

/opt/ultrahdr/ultrahdr_app -m 0 -p /mnt/data/src_rec2100-pq_rgba1010102.raw -w 1920 -h 1080 -q 100 -Q 100 -a 5 -C 2 -c 0 -t 2 -R 1 -z /mnt/data/rec2100-pq_base.jpeg
```

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
docker run -it -P --name ultrahdr_rev01 -v /Users/toru/Work/sample_code/Temporary/06_ultrahdr:/mnt/mac --rm takuver4/ultrahdr:rev01
```



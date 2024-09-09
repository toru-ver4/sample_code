# Memo

## Build

### Ultra HDR

```powershell
docker build -f ./docker_files/00_Dockerfile_UltraHDR --no-cache -t takuver4/ultrahdr:rev03 .
docker run -it -P --name ultrahdr_rev03 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/ultrahdr:rev03
docker run -it -P --name ultrahdr_rev01 -v /Users/toru/Work/sample_code/Temporary/06_ultrahdr:/mnt/data --rm takuver4/ultrahdr:rev03
```

### OpenImageIO

```powershell
docker build -f ./docker_files/01_Dockerfile_OpenImageIO --no-cache -t takuver4/openimageio:rev01 .
docker run -it -P --name openimageio_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/openimageio:rev01
docker run -it -P --name openimageio_rev01 -v /Users/toru/Work/sample_code:/mnt/data --rm takuver4/openimageio:rev01
```

### OpenColorIO

```powershell
docker build --no-cache -f ./docker_files/02_Dockerfile_OpenColorIO -t takuver4/opencolorio:rev01 .
docker run -it -P --name opencolorio_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/opencolorio:rev01
docker run -it -P --name opencolorio_rev01 -v /Users/toru/Work/sample_code:/mnt/data --rm takuver4/opencolorio:rev01
```

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
docker run -it -P --name opencolorio_rev01 -v /Users/toru/Work/sample_code:/mnt/data -e PYTHONPATH=/work/src/ty_lib --rm takuver4/opencolorio:rev01
```

### CTL_ICC

```powershell
docker build --no-cache -f ./docker_files/03_Dockerfile_CTL_ICC -t takuver4/ctl_icc:rev01 .
docker run -it -P --name ctl_icc_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/ctl_icc:rev01
```

### STILL_HDR

```powershell
docker build -f ./docker_files/04_Dockerfile_STILL_HDR -t takuver4/still_hdr:rev01 .
docker run -it -P --name still_hdr_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/still_hdr:rev01
```

### png_cicp_editor

```powershell
docker build -f ./docker_files/05_png_cicp_editor -t takuver4/png_cicp_editor:rev01 .
docker run -it -P --name png_cicp_editor -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/png_cicp_editor:rev01
```

### Integrated Image

#### build

```powershell
docker build --no-cache -f ./docker_files/Dockerfile -t takuver4/ty_env_v2:rev06 .
```

## Push

```powershell
docker push takuver4/ty_env_v2:rev06
```

## run

```powershell
docker-compose up -d
  or
docker run -it -P --name ty_env_v2_rev06 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm takuver4/ty_env_v2:rev06 bash
docker run -itd -P --platform linux/amd64 --name ty_env_v2_rev06 -v /Users/toru/Work/sample_code:/work/src -e PYTHONPATH=/work/src/ty_lib --rm takuver4/ty_env_v2:rev04
```

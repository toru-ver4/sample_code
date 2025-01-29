# Memo

## docker pull

```powershell
docker pull aswf/ci-usd:2021.6; docker pull aswf/ci-usd:2022.4; docker pull aswf/ci-usd:2023.2; docker pull aswf/ci-usd:2024.2; 
```

## build type2

```powershell
docker build -f ./Dockerfile -t takuver4/ty_env_v2:ocio2.4 .
```

## run

```powershell
docker run -it -P --name ci-usd_ocio_v20 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm aswf/ci-usd:2021.6
docker run -it -P --name ci-usd_ocio_v21 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm aswf/ci-usd:2022.4
docker run -it -P --name ci-usd_ocio_v22 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm aswf/ci-usd:2023.2
docker run -it -P --name ci-usd_ocio_v23 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm aswf/ci-usd:2024.2

docker run -it -P --name ocio24 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src --rm takuver4/ty_env_v2:ocio2.4
```

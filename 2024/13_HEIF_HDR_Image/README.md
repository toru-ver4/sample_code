# Memo

```powershell
docker build -f ./docker/Dockerfile -t takuver4/libheif:rev01 .
docker run -it -P --name libheif_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/mnt/data --rm takuver4/libheif:rev01
```

```bash
heif-enc --verbose --quality 100 --colour_primaries 9 --transfer_characteristic 16 --matrix_coefficients 9 --full_range_flag 1 ./src_png/Rec2100-PQ.png -o ./dst_heif/Rec2100-PQ.heic
```

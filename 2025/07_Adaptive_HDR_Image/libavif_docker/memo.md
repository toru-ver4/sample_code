# Memo

## docker build

```powershell
docker build -t takuver4/still_hdr:rev02 .
docker push takuver4/still_hdr:rev02
```

## docker run

```powershell
docker-compose up -d
  or
docker run -itd -P --name still_hdr_rev01 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src -v D:\abuse:/work/overuse --rm takuver4/still_hdr:rev01
docker run -itd -P --platform linux/amd64 --name still_hdr_rev01 -v /Users/toru/Work/sample_code:/work/src -e PYTHONPATH=/work/src/ty_lib --rm takuver4/still_hdr:rev01
```

## build libavif

```bash
mkdir -p /work/local
cd /work/local
git clone https://github.com/toru-ver4/libavif.git
cd libavif/
chmod 755 ./my_build.sh
./my_build.sh
```

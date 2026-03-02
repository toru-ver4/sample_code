# Memo

rev02: 改造版、ローカルビルド
rev03: 改造版、Dockerfile でビルド
rev04: 最新版の公式ビルド
rev05: やっぱり v1.3.0 のビルド

## docker build

```powershell
docker build -t takuver4/still_hdr:rev03 .
docker push takuver4/still_hdr:rev03
```

## docker run

```powershell
docker-compose up -d
  or
docker run -itd -P --name still_hdr_rev04 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src -v D:\abuse:/work/overuse --rm takuver4/still_hdr:rev04
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

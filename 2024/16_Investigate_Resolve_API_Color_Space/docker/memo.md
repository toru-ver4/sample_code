# Memo

```
docker build -t takuver4/resolve:19.1 .
```

## WSL2 にて

以下の URL を参考に NVIDIA Container Toolkit をインストール
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo yum-config-manager --enable nvidia-container-toolkit-experimental
sudo yum install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
# Docker Desktop を再起動
```

```bash
docker run -it -P --gpus all --privileged --runtime=nvidia --name resolve_19_1 -v C:\Users\toruv\OneDrive\work\sample_code:/work/src -v D:\abuse:/work/overuse -e DISPLAY=host.docker.internal:0.0 --rm takuver4/resolve:19.1
```

```
./DaVinci_Resolve_Studio_19.1_Linux.run --appimage-extract
./installer -i -a .
```

## USB Dongle

以下のコマンドで BUSID を取得

```powershell
usbipd wsl list
```

BUSID  VID:PID    DEVICE
7-2    096e:0201  USB Input Device

```powershell_as_Administrator
usbipd.exe bind -b 7-2
usbipd.exe attach -w OracleLinux_9_1 -b 7-2
```

```powershell_as_Administrator
usbipd.exe detach -b 7-2
usbipd.exe unbind -b 7-2
```

# Python Environment

## Overview

-

## Requirements

-

## Build

```powershell
docker build -t takuver4/jupyter_dev:rev00 .
```

## Push

```powershell
docker push takuver4/jupyter_dev:rev00
```

## Create a container using `docker-compose up`

```powershell
docker-compose up -d
```

### (Option) Create a container using `docker run`

```powershell
$WORKING_DIR = "C:\Users\toruv\OneDrive\work\sample_code";
$DATA_DIR = "D:\abuse";
$PYTHON_LIB_DIR_ON_LINUX = "/home/jovyan/work/ty_lib";
docker run -it -d -P --name jupyter_dev_00 `
-p 10000:8888 `
-v ${WORKING_DIR}:/home/jovyan/work -v ${DATA_DIR}:/home/jovyan/abuse `
-e DISPLAY=docker.for.win.localhost:0.0 `
-e JUPYTER_ENABLE_LAB=yes `
-e PYTHONPATH=$PYTHON_LIB_DIR_ON_LINUX --rm takuver4/jupyter_dev:rev00
```

## Attach Visual Studio Code

![figure](./figures/attach_visual_studio_coode.png)

## Information

### pip list

### X Server

The [X410](https://x410.dev/) is recommended.

If you want to use the [VcXsrv](https://sourceforge.net/projects/vcxsrv/), please change the environment variable `DISPLAY`.


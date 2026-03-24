# INSTALL

## Install Visual Studio Build Tools 2022

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --accept-package-agreements --accept-source-agreements `
  --silent `
  --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

## Build

```powershell
# run "C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Visual Studio 2022\Visual Studio Tools\VC\x64 Native Tools Command Prompt for VS 2022.lnk"
cd capture_scRGB
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```powershell
.\build\my_capture_app.exe 1 output.jxr
```

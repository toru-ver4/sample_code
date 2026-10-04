# sample_code
sample cods.

## Setup

* Install uv

```PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

* Install Python 3.13

```PowerShell
uv python install 3.13 --default
```

* Install packages

```PowerShell
cd ${PROJECT_ROOT_DIRECTORY}
uv sync
```
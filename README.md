# sample_code
sample cods.

## Setup

* Install uv

```PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

* Install Python 3.13

```PowerShell
winget install --id Python.Python.3.13 --exact --source winget
```

* Install packages

Use the Python installed by winget explicitly so uv does not select a
uv-managed Python that Windows Application Control may block.

```PowerShell
cd ${PROJECT_ROOT_DIRECTORY}
uv sync --python "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" --no-managed-python --no-python-downloads
```

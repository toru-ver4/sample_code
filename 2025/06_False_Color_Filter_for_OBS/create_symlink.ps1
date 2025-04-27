# パラメータ
$src = ".\color.effect"
$dst = "C:\Program Files\obs-studio\data\obs-plugins\obs-filters\color.effect"

# 絶対パスに変換
$srcFullPath = (Resolve-Path -Path $src).Path

# ソースファイルの存在確認
if (-not (Test-Path -Path $srcFullPath)) {
    Write-Error "Source file does not exist: $srcFullPath"
    exit 1
}

# 既存リンクやファイルの確認
if (Test-Path -Path $dst) {
    Write-Host "Destination already exists: $dst"
    exit 1
}

try {
    cmd /c "mklink `"$dst`" `"$srcFullPath`""
    Write-Host "Symbolic link created successfully: $srcFullPath -> $dst"
} catch {
    Write-Error "Failed to create symbolic link. Run this script as administrator."
}

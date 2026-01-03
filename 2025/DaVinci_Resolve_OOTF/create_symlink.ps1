# パラメータ
$src = "./DCTL"
$dst = "C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\LUT\DCTL_OOTF"

# 絶対パスに変換
$srcFullPath = Resolve-Path -Path $src

# シンボリックリンク作成（管理者権限が必要）
if (-not (Test-Path -Path $srcFullPath)) {
    Write-Error "Source directory does not exist: $srcFullPath"
    exit 1
}

if (Test-Path -Path $dst) {
    Write-Host "Destination already exists: $dst"
    exit 1
}

try {
    cmd /c "mklink /D `"$dst`" `"$srcFullPath`""
    Write-Host "Symbolic link created successfully: $srcFullPath -> $dst"
} catch {
    Write-Error "Failed to create symbolic link. Run this script as administrator."
}

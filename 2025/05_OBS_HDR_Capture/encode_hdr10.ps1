# PowerShell スクリプト例

# 作成するディレクトリの一覧
$directories = @(
    "encode_data"
)

# 各ディレクトリが存在しない場合は作成
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
}

# ffmpeg コマンドの実行
ffmpeg -loop 1 -r 24 -i ".\img\src_hdr.png" -t 10 `
    -c:v libx265 -x265-params "range=full:colorprim=9:transfer=16:colormatrix=9:master-display=G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(100000000,1):max-cll=10000,10000" `
    -pix_fmt yuv420p10le -crf 10 -tag:v hvc1 -an `
    -color_primaries bt2020  -color_trc smpte2084  -colorspace bt2020nc "encode_data\src_hdr.mp4" -y

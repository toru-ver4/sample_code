# スクリプトのディレクトリから encode_data フォルダのパスを取得
$encodeDataPath = Join-Path $PSScriptRoot "encode_data"
Write-Host "Encode data directory: $encodeDataPath"

# "H265" を含む .mp4 または .mov ファイルを再帰的に検索
$videoFiles = Get-ChildItem -Path $encodeDataPath -Recurse -File | Where-Object {
    $_.Name -match "H265" -and ($_.Extension -eq ".mp4" -or $_.Extension -eq ".mov")
}
Write-Host "Found $($videoFiles.Count) video files matching criteria."

# 各ファイルに対して FFmpeg コマンドを実行し、.h265 の bitstream を作成
foreach ($video in $videoFiles) {
    Write-Host "---------------------------------------"
    Write-Host "Processing file: $($video.FullName)"
    
    # 出力ファイル名：元のファイル名の拡張子を除いたものに .h265 を追加
    $outputFile = Join-Path $video.DirectoryName ($video.BaseName + ".h265")
    Write-Host "Output file: $outputFile"
    
    # FFmpeg コマンドの引数を作成し表示
    $ffmpegArgs = "-i `"$($video.FullName)`" -c copy -bsf hevc_mp4toannexb `"$outputFile`""
    Write-Host "Executing FFmpeg command: ffmpeg $ffmpegArgs"
    
    # FFmpeg を利用して bitstream を生成
    ffmpeg -i $video.FullName -c copy -bsf hevc_mp4toannexb $outputFile -y > $null 2>&1
}

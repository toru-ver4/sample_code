# UltraHDR に関するメモ

## 用語の整理

なぜか分からないが、UltraHDR には独自の分からない用語が頻出する。
Adobe/Apple の用語との対応関係を明確にしておく。

| Ultra HDR 用語 | 説明 |
|:-----------|:-----------|
| pixel_gain | (Y_hdr + offset) / (Y_sdr + offset) の Linear値 |
| content_boost | SDR を何倍すれば HDR になるかを示した Linear値。pixel_gain に近い |
| min_content_boost, max_content_boost | **ユーザーが指定する** content_boost の最小値・最大値 |
| map_min_log2, map_max_log2 | min_content_boost, max_content_boost を log2 に変換した値 |
| log_recovery | pixel_gain を log2 に変換し、かつ map_min_log2, map_max_log2 で正規化した値 | 
| clamped_recovery | log_recovery を 0.0～1.0 でクランプした値 |
| recovery | clamped_recovery に対してオプションの gamma を適用した値 |
| encoded_recovery | recovery を 0～255 などの整数型に量子化した値  |
| gain_map_min, gain_map_max | map_min_log2, map_max_log2 と同義だが、これはデコード時に metadata として取得するもの |
| HDR white point | 表示デバイス上のHDRコンテンツの最大輝度 |
| SDR white point | 表示デバイス上のSDRコンテンツの最大輝度 |
| boost | HDR white point を SDR white point で割った値 |
| max_display_boost | 利用可能な最大の boost値。状況で変わる値 |
| display_boost | min(max_content_boost, max_display_boost) |
| hdr_capacity_min , hdr_capacity_max | display boost の最小値・最大値を log2 に変換したもの |

## これまでの流れ

* ./ffmpeg_prompt2.md の内容に従い作業をしてもらい、./ffmpeg_codex_report.md が出てきた
* これにより gray の問題は解決した

## 次の流れ

./scripts/check_10bit_diff.py で `check_color_diff` を有効化し、有彩色も許容誤差を設けた。
有彩色も許容誤差内に収まるように ffmpeg側のソースコード修正を試して欲しい。
なお、調査結果は ffmpeg_codex_report_color.md という名前で最後にまとめること。
また、既に ffmpeg_codex_report_color.md が存在していた場合は適当なサフィックスをつけてファイルを分けること。

## 注意点

* 前回の codex のコード修正結果は git log の `a1a821225e` を確認すること
* 今回追加した有彩色の誤差確認は、もしかすると chroma subsampling の問題で改善が不可能な可能性がある
  * テストが画像のカラーパッチサイズは 8x8 としている
* もしもカラーパッチが小さすぎて chroma subsampling の問題だと判明した場合は、テスト画像を「私が」作り直すので、その旨をレポートに書いて終了すること
  * 勝手にテスト画像作成の python コードを修正しないこと

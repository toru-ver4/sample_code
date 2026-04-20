# FFmpeg ソースコード解析依頼プロンプト

## 背景

別プロジェクトで FFmpeg を使用した際に、FFmpeg でエンコードした HEVC や AV1 のファイルを DaVinci Resolve や Google Chrome などの別アプリでデコードすると、3/1023 ～ 4/1023 CV 程度の誤差が生じることを経験した。このズレの原因を本格的に調査することにした。

## これまでの流れ

[ffmpeg_prompt.md](./ffmpeg_prompt.md) を参照。
なお、このファイルは参考情報として下さい。具体的な命令は以下で行う。

## お願いしたいこと

次のステップとしてソースコード解析の修正により、この問題を解決できるか確認したい。
生成AI に確認させるための環境として以下を準備した。

* FFmpeg のソースコード
  * /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation/ffmpeg_8.1_src
* FFmpeg のビルドと yuv420p10le ファイルの生成
  * まず `docker run -it -P --name ffmpeg_investigation -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src --rm takuver4/ffmpeg_investigation:rev02 bash` をコールして docker コンテナ起動
  * docker コンテナ内で `cd /work/src/2026/02_ffmpeg_deviation/ && ./scripts/build_ffmpeg.sh` を実行して ffmpeg をビルド
  * docker コンテナ内で `cd /work/src/2026/02_ffmpeg_deviation/ && ./scripts/encode.sh` を実行してカスタムした ffmpeg で yuv420p10le のファイルを生成
* 生成した yuv420p10le ファイルの精度確認
  * docker コンテナ内で `exit` をしてコンテナから離れる
  * `cd /mnt/c/Users/toruv/OneDrive/work/sample_code/2026/02_ffmpeg_deviation && python3 ./scripts/check_10bit_diff.py` を実行して "OK" となれば成功。"NG" ならば失敗。

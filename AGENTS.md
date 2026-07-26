# AGENTS.md

## 実装の方針

**個人開発** の環境であることを念頭に、異常系や非機能要件を作り込まなすぎないようにする。
正常系が正しく動けば良く、少しでも異常を観測したらエラーメッセージを吐いて何も処理を実行せずにプログラムを停止する方針とする。

## ドックストリング

Python の関数ドックストリングは NumPy docstring style で書く。言語は英語とする。
セクションは `Parameters`, `Returns`, `Notes`, `Examples` の順にする。
簡潔な関数だった場合は Notes は省略しても良い。
戻り値なしでも `Returns: None` を明記し、Examples には最小例を書く。

## テストコード

テストコードを書いた際は、ファイルの冒頭に以下の内容を書いておくこと

* どのディレクトリで実行すればよいか
* どのコマンド、コマンドライン引数で実施すればよいか

## プログラムの実行環境

コマンドが見つからない場合は、現状の環境に新規インストールを試すのではなく、
以下の docker コンテナを使って実行できるか試すこと。それでも無理な場合はインストールしてよいか許可を取ること。
なお、`takuver4/ty_env_v2:rev12` が見つからない場合は `takuver4/ty_env_v2:rev13` など新しい revision を探すこと。

```bash
docker run --rm \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  -w /work/src \
  takuver4/ty_env_v2:rev12 \
  iccFromXml input.xml output.icc
```

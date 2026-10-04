# Resolve API samples

`api_sapmle.py` は Python 3.13 / Resolve Studio 21.1 向けのサンプルです。
更新した `ty_davinci_resolve` の editable install 環境を使います。
各サンプルは同名の既存サンプルプロジェクトを削除して作り直します。
保持したいプロジェクトがある場合は `project_name` に別名を指定してください。

## 実行

`C:\Users\toruv\OneDrive\work\sample_code` で実行します。

```powershell
& ./ty_lib/TY_DaVinci_Resolve_Control_Lib/.venv313/Scripts/python.exe ./2026/06_DaVinci_Resolve_API_Test_V3/api_sapmle.py
```

現在のエントリーポイントは矩形サンプルです。他のサンプルは末尾の呼び出しを選択するか、関数を import して呼び出します。
`encode_test(output_dir=...)` で既存の出力ディレクトリを指定できます。省略時は Downloads に出力します。

## 21.1 更新内容

- 全設定取得は `GetSettings()` を優先します。メソッドが存在しない場合だけ旧形式を使い、失敗時の再試行はしません。snapshot は数値等の元の型を保持します。
- 現在のTimeline取得は `get_current_timeline()` を使います。
- Timeline解像度は `TimelineSetting`、作業輝度モードは `WorkingLuminanceMode` を使います。Monitor/色管理の共有キーには既存の `ProjectSetting` 定数を使います。
- RCMは Automatic off / Custom と関連値を明示します。プリセット名だけの変更による設定維持は保証しません。
- 59.94 fpsはTimeline作成前にProjectへ設定して継承します。空のCustom Timelineに対する整数fps設定と、クリップ追加後の制限は別の条件です。
- RCMで使うFusionサンプルには `refresh_fusion_color_management()` を入れています。アニメーション作成サンプルは更新後にEditページへ移り、保存します。
- 矩形サイズが奇数のとき、中心の半ピクセル位置を保持してから丸めるように修正しました。

API形式、設定の維持制限、Fusion処理の必要性の根拠は[ライブラリの21.1検証記録](../../ty_lib/TY_DaVinci_Resolve_Control_Lib/docs/resolve-21.1-verification.md)にあります。
21.0.4でのPython 3.13実機検証は行っていません。

## 検証

Resolveを使わない設定取得の単体チェック:

```powershell
& ./ty_lib/TY_DaVinci_Resolve_Control_Lib/.venv313/Scripts/python.exe -m pytest ./2026/06_DaVinci_Resolve_API_Test_V3/test_api_sample.py -q -x -p no:cacheprovider
```

Resolve起動中の実機チェック:

```powershell
$env:RUN_RESOLVE_SAMPLE_TESTS = '1'
& ./ty_lib/TY_DaVinci_Resolve_Control_Lib/.venv313/Scripts/python.exe -m pytest ./2026/06_DaVinci_Resolve_API_Test_V3/test_api_sample.py -q -x -p no:cacheprovider
Remove-Item Env:RUN_RESOLVE_SAMPLE_TESTS
```

実機チェックはUUID名のプロジェクトを使用して後処理で削除し、動画はpytestの一時ディレクトリへ出力します。
Project作成、Project/Timeline設定取得・変更、保存・再読込後の主要設定の維持、Fusion作成、奇数サイズの矩形中心、ProRes動画の生成を確認します。

2026-10-04、Windows / Python 3.13.16 / Resolve Studio 21.1.0.17で、単体3件・実機7件の合計10件が成功しました（68.32秒）。
動画のピクセル一致・旧版との描画互換性の判定はこのサンプル検証には含めません。

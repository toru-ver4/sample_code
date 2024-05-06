# Direct X はさっぱりなので理解した内容をメモする

https://chat.openai.com/share/9a9e181d-b4b7-44b0-9bbb-9ffe88fb643d

## 基本用語確認

* D3D12
  * Direct3D 12 の略
* Direct3D 12
  * DirextX 12 においてリアルタイム3Dグラフィックスのレンダリングを行うためのコンポーネントまたは DirectX API
  * 【参考】DirextX 12 には Direct2D、DirectCompute、DirectWrite など様々なコンポーネント/API が存在する
* DXGI
  * DirectX Graphics Infrastructure Factory
  * DirectXのグラフィックスコンポーネントを管理するための最重要オブジェクト
  * 主に以下を管理
    * アダプタ（ビデオカード）の列挙
    * スワップチェーンの作成 (ビデオバッファ的なやつ)
    * ディスプレイの管理

* commmand queue
  * GPUへの命令を整理して送信するのに使う
  * DirectX 12 では必須。使わないプログラミングは不可らしい。

* Swap Chain
  * ビデオバッファ的なやつ

* DXGI_FORMAT
  * bit深度や int/float を指定するやつ
  * 例は以下
    * `DXGI_FORMAT_R16G16B16A16_FLOAT`
    * `DXGI_FORMAT_R10G10B10A2_UNORM`
      * 余談だが UNORM は 0.0 - 1.0 が 0 - 1023 に正規化されてる意味

* Root Signature
  * シェーダープログラムがグラフィックスパイプラインに必要なリソースへどのようにアクセスするかを定義するコンポーネント

* CD3DX12_ROOT_PARAMETER
  * ルートパラメータのこと
  * SRV, UAV, サンプラーなどに関する基本的な？パラメータを設定するくさい
    * SRV の場合は Descriptor Range を設定してるっぽいね

* Graphics Pipeline State Object (PSO)
  * シェーダー、レンダーターゲットとか色々と多岐にわたってやるやつ
  * 

## 理解したこと

* Win32Application::Run
  * このデモ用に作られたコード

* D3D12HDRクラス
  * サンプルコードのメイン部分のクラス
  * DXSample クラスを継承

* D3D12HDR::OnInit
  * D312HDR クラスのコンストラクタとは別の初期化関数
  * 初期化プロセスの複雑性を軽減するために分けられている（by ChatGPT）
  * 今回のコードでは Win32Application::Run から呼ばれてますね

* MSG msg
  * メッセージをやりとり？するための情報を入れる構造体
  * メンバに HWND があり、これで該当アプリケーションを指定するっぽい
  * 厳密には アプリケーションのウィンドウプロシージャに渡すっぽい
  
* `CreateDXGIFactory2`関数
  * DXGI Factory を用意する
  * ここから GPUリソースの取得とか描画用バッファ (Swap Chain)とか準備するっぽい

* つながっているモニターが HDR かどうか
  * DXGI_OUTPUT_DESC1 の ColorSpace で確認できるっぽい
    * https://github.com/microsoft/DirectX-Graphics-Samples/blob/51d0c1c5e225186a279bcdf15b7dbf68745301db/Samples/Desktop/D3D12HDR/src/D3D12HDR.cpp#L1118-L1124

* .hlsli は Include 用のファイル。なので末尾に "i" がついてるっぽい

## 全体の流れ

* LoadPipeline
  * DXGI Factory `m_dxgiFactory` を取得
  * `m_dxgiFactory` を使って GPUリソース `m_device` を取得
  * GPUへの命令用の `queueDesc` を用意

* LoadAssets
  * root signature を作る
    * root signature はシェーダープログラムがグラフィックスパイプラインに必要なリソースへどのようにアクセスするかを定義するコンポーネント
    * それには root signature descriptor が必要
      * リソースのバインディング や パフォーマンスの最適化などを担当
      * 具体的には root parameters と sampler を定義する
        * 定数バッファとか直接バッファとかを設定？
        * sampler はテクスチャの扱いに関するものを記述？
  * graphics pileine state を作る。なお、PSO = Pipeline State Object
    * グラフィックスパイプラインの全ての状態を一つのオブジェクトにカプセル化します
    * ここでは 3種類の PSO を作っている
      * グラデーション描画用
      * 三角形描画用
      * 最終ブレンドバッファ？用
  * Command List を作る
    * よくわからんがワンライナーで済んでる
  * Vertex Buffer を作る
    * Vertex の座標とか色指定をしている
    * CreateCommittedResource 以下を確保
      * 実際にGPUで使用される頂点バッファ（m_vertexBuffer）
      * CPUからデータをアップロードするための一時バッファ（m_vertexBufferUpload）
    * 作成したバッファにデータをコピー (memcpy)
    * Buffer Views の初期化
      * バッファのアドレスとかサイズとか色々と指定
  * ウィンドウサイズに依存する者たちの初期化
    * 各フレーム用にレンダーターゲットビューを作成
      * レンダーバッファ（ターゲット）ができあがる？
    * 中間レンダーターゲットの作成
    * UIレンダーターゲットの作成
    * ビューポートとシザー矩形の設定
    * UIレイヤーの更新
  * レンダリングについて
    * 最初に色々とセットアップ
    * スワップチェーンに描画を適用
      * Swap Chain とはダブルバッファリングとかトリプルバッファリングとかのシステム
      * 大きく以下に分けられる
        * 画面表示（Read）に使われる フロントバッファ
        * 画面描画 (Write) に使われる バックバッファ
      
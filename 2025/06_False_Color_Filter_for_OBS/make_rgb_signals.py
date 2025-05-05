#!/usr/bin/env python3
import numpy as np


def make_tesselation(gsv):
    """
    Python 移植版 of MATLAB make_tesselation.m
    入力:
      gsv: 1D iterable of gray-scale values (e.g. [0, 64, 128, 192, 255])
    出力:
      TRI_ref: (M×3) ndarray of triangle vertex indices (0-based)
      RGB_ref: (6*N^2 × 3) ndarray of RGB corner coordinates
    """
    # グレースケール値ベクトル
    gsv = np.asarray(gsv, dtype=int)
    N = gsv.size

    # J, K を作成し一次元化
    J, K = np.meshgrid(gsv, gsv, indexing='xy')
    J = J.ravel()
    K = K.ravel()

    # 下底面と上底面の定義
    Lower = np.full_like(J, gsv[0])
    Upper = np.full_like(J, gsv[-1])

    # 6面分の RGB_ref を積み重ね
    face1 = np.column_stack((Lower, J, K))
    face2 = np.column_stack((K, Lower, J))
    face3 = np.column_stack((J, K, Lower))
    face4 = np.column_stack((Upper, K, J))
    face5 = np.column_stack((J, Upper, K))
    face6 = np.column_stack((K, J, Upper))
    RGB_ref = np.vstack((face1, face2, face3, face4, face5, face6))

    # 三角形のインデックスを生成 (MATLAB 側は 1-based なので Python では -1)
    tri_count = 12 * (N - 1)**2
    TRI_ref = np.zeros((tri_count, 3), dtype=int)
    idx = 0
    for s in range(6):
        for q in range(N - 1):
            for p in range(N - 1):
                # MATLAB: m = N^2*(s) + N*q + p + 1
                m = N**2 * s + N * q + p
                # 2つの三角形を定義 (1-based)
                t1 = np.array([m,     m + N,     m + 1    ])  # A-B-C
                t2 = np.array([m + N, m + N + 1, m + 1    ])  # B-D-C
                # Python: 0-based に変換
                TRI_ref[idx    ] = t1 - 1
                TRI_ref[idx + 1] = t2 - 1
                idx += 2

    return TRI_ref, RGB_ref


def make_rgb_signals(n=11, max_val=255):
    # V の初期化
    V = None

    # V 未指定なら 0..max_val を n 個の等間隔で生成
    if V is None or V.size == 0:
        # n が None の場合はデフォルト 11
        if n is None:
            n = 11
        V = np.round(np.linspace(0, 1, int(n)) * float(max_val)).astype(int)

    # tesselation して RGB 値を取得
    _, rgb = make_tesselation(V)

    # 重複を除く
    rgb = np.unique(rgb, axis=0)
    return rgb


if __name__ == "__main__":
    # コマンドライン引数処理
    rgb = make_rgb_signals(n=3, max_val=1023)
    print(rgb)

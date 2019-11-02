import numpy as np
from colour import xyY_to_XYZ, XYZ_to_RGB, RGB_to_XYZ, XYZ_to_xyY
from colour.models import BT709_COLOURSPACE

# BT.709色域外の xyY データ。Y は Green の Primariy に合わせた。
xyY_bt2020 = np.array([[0.26666249, 0.47998497, 0.67799807],
                       [0.25055208, 0.5328208,  0.67799807],
                       [0.23444166, 0.58565664, 0.67799807],
                       [0.21833125, 0.63849248, 0.67799807],
                       [0.20222083, 0.69132832, 0.67799807],
                       [0.18611042, 0.74416416, 0.67799807],
                       [0.17,       0.797,      0.67799807]])

d65 = np.array([0.3127, 0.3290])

if __name__ == '__main__':
    # とりあえず XYZ する
    large_xyz_bt2020 = xyY_to_XYZ(xyY_bt2020)

    # BT.2020 --> BT.709 のRGB値へ変換
    rgb_linear_bt709 = XYZ_to_RGB(
        XYZ=large_xyz_bt2020, illuminant_XYZ=d65, illuminant_RGB=d65,
        XYZ_to_RGB_matrix=BT709_COLOURSPACE.XYZ_to_RGB_matrix)

    # BT.709 の式域内にクリッピング
    print(rgb_linear_bt709)
    rgb_linear_bt709_clipped = np.clip(rgb_linear_bt709, 0.0, 1.0)

    # xyY に変換して最終出力する
    large_xyz_bt709_clipped = RGB_to_XYZ(
        RGB=rgb_linear_bt709_clipped, illuminant_RGB=d65, illuminant_XYZ=d65,
        RGB_to_XYZ_matrix=BT709_COLOURSPACE.RGB_to_XYZ_matrix)
    xyY_bt709_clipped = XYZ_to_xyY(large_xyz_bt709_clipped)
    print(xyY_bt709_clipped)

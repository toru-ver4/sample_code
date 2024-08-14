import cv2
import numpy as np

# PNG画像を16ビットで読み込む
img = cv2.imread('path_to_image.png', cv2.IMREAD_UNCHANGED)

# OpenCVでは画像はBGR形式で読み込まれるため、順序を調整
if img.shape[2] == 3:  # BGRの場合
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画像の高さと幅を取得
height, width = img.shape[:2]

# Alphaチャンネルを全ピクセルで最大値（16ビットなので65535）に設定
alpha_channel = np.full((height, width), 65535, dtype=np.uint16)  # 最大値でフルAlpha

# RGBとAlphaチャンネルを結合
rgba_image = np.dstack((img, alpha_channel))

# 16ビットデータを10ビットと2ビットに適切にスケーリング
r = (rgba_image[:,:,0] >> 6).astype(np.uint16)  # Redを16ビットから10ビットに
g = (rgba_image[:,:,1] >> 6).astype(np.uint16)  # Greenを16ビットから10ビットに
b = (rgba_image[:,:,2] >> 6).astype(np.uint16)  # Blueを16ビットから10ビットに
a = (rgba_image[:,:,3] >> 14).astype(np.uint16) # Alphaを16ビットから2ビットに

# 新しい配列に結合
rgba1010102 = (r << 22) | (g << 12) | (b << 2) | a

# ファイルに保存
rgba1010102.tofile('output_file.raw')

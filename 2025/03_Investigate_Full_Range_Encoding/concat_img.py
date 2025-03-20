import os
import glob
from PIL import Image

# 画像フォルダと画像ファイルの取得（PNGファイル）
image_folder = "./debug"
image_paths = glob.glob(os.path.join(image_folder, "*.png"))
image_paths.sort()  # 任意：ファイル名順にソート

# タイルの設定
tile_cols = 6
tile_rows = 3
tile_width = 800
tile_height = 800

# 出力画像のサイズ (背景は黒色)
result_width = tile_cols * tile_width  # 4000
result_height = tile_rows * tile_height  # 3200
result_image = Image.new("RGB", (result_width, result_height), "black")

# 画像をタイル状に貼り付け
for index, image_path in enumerate(image_paths):
    if index >= tile_cols * tile_rows:
        break  # タイル枠を超えたら終了
    img = Image.open(image_path)
    # 貼り付け先の位置 (列と行)
    col = index % tile_cols
    row = index // tile_cols
    x = col * tile_width
    y = row * tile_height
    result_image.paste(img, (x, y))

# 結合画像の保存
result_image.save("./img/combined.png")

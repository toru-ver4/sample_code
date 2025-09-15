import os
import re

# 対象フォルダ
folder = "./debug/rename"

# フォルダ内のファイルを走査
for filename in os.listdir(folder):
    if filename.endswith(".tif"):
        # 正規表現で最後の "_" 以降の文字列を削除
        new_name = re.sub(r'_(\d+\.\d+\.\d+)\.tif$', '.tif', filename)
        
        # 変更がある場合のみ rename 実行
        if new_name != filename:
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

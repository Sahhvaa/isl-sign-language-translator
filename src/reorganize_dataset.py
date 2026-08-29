import os
import re
import shutil

SRC = "data/Greetings"
DST = "data/raw_videos"

os.makedirs(DST, exist_ok=True)

for folder_name in os.listdir(SRC):
    src_path = os.path.join(SRC, folder_name)
    if not os.path.isdir(src_path):
        continue

    # "48. Hello" -> "Hello" -> "hello"
    match = re.match(r"^\d+\.\s*(.+)$", folder_name)
    word = match.group(1) if match else folder_name
    clean_word = word.strip().lower().replace(" ", "_")

    dst_path = os.path.join(DST, clean_word)
    os.makedirs(dst_path, exist_ok=True)

    for file in os.listdir(src_path):
        shutil.copy2(os.path.join(src_path, file), os.path.join(dst_path, file))

    print(f"{folder_name}  ->  {clean_word}")

print("Done.")
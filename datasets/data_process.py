#%%
from pathlib import Path
import shutil

coco_pose_train_txt = 'coco-pose/train2017.txt'
coco_pose_val_txt = 'coco-pose/val2017.txt'
coco_pose_train_img_dir = 'coco-pose/images/train2017'
coco_pose_val_img_dir = 'coco-pose/images/val2017'

def copy_from_txt(txt_file, dest_dir):
    """把 txt 中每行给出的图片路径复制到 dest_dir"""
    count = 0
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            src_img = Path(line.strip())
            src_img_path = Path(
                'coco') / src_img  # Path('coco')根据自己存train2017或val2017图片进行更改(比如我的在coco/images/train2017)
            if src_img_path.is_file():
                count += 1
                shutil.copy2(src_img_path, Path(dest_dir) / Path(src_img_path.name))
            else:
                print(f'{src_img_path} not exists, check it')
    return count

if __name__ == "__main__":
    count = copy_from_txt(coco_pose_val_txt, coco_pose_val_img_dir)
    print(f"val count:{count}")
    count = copy_from_txt(coco_pose_train_txt, coco_pose_train_img_dir)
    print(f"train count:{count}")
#%% md
# 原始的yolov8s-pose
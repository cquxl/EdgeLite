# train_student.py
from ultralytics import YOLO
import os
from pathlib import Path
# -----------------------------
# 配置
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
student_weights = "weights/yolov8s-pose-prune-sp0.5.pt"  # Student 模型
# data_path = "datasets/coco-pose"                          # COCO 格式数据集目录
data_yaml = "datasets/my-coco-pose.yaml"
save_dir = "runs/student_train"                            # 保存训练结果目录

epochs = 100       # 训练轮数
batch_size = 16    # Batch size
img_size = 1088     # 输入图片尺寸
device = "1"       # 使用 GPU id

# -----------------------------
# 创建保存目录
# -----------------------------
os.makedirs(save_dir, exist_ok=True)


if __name__ == "__main__":
    # -----------------------------
    # 加载 Student 模型
    # -----------------------------
    model = YOLO(student_weights)
    
    # -----------------------------
    # 开始训练
    # -----------------------------
    model.train(
        data=data_yaml,        # 数据集路径
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        amp=True,
        save_dir=save_dir
    )
    
    print(f"训练完成，结果保存在 {save_dir}")

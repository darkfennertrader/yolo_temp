import os
from datetime import datetime
import torch
import ultralytics
from ultralytics import YOLO
from kill_gpu_proc import clear_gpus_memory


ultralytics.checks()

os.environ["OMP_NUM_THREADS"] = "8"

print("Ultralytics version:", ultralytics.__version__)
print("PyTorch version:", torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
print("Available GPU count:", torch.cuda.device_count())


clear_gpus_memory()

# Load a pretrained YOLO model (recommended for training)

name = "test"
# Get the current datetime
now = datetime.now()
# Format the datetime as a string in the format YYYYMMDDHHMMSS
datetime_str = now.strftime("%Y-%m-%d-%H%M%S")

print(os.getcwd())

model = YOLO("models/yolov8s.pt")
print()
print(model.info())
print()

results = model.train(
    task="detect",
    data="yolo_dataset/dataset.yaml",
    epochs=3,
    imgsz=640,
    workers=8,
    batch=16,
    device=[0],
    amp=False,
    project="runs",
    name=f"{name}_{datetime_str}",
)

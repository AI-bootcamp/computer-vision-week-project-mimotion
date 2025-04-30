!pip install kagglehub ultralytics pyyaml

import kagglehub
from pathlib import Path

# Download the “Colour Classification” data
path = kagglehub.dataset_download("trushraut18/colour-classification")

# Point to the folder that contains train/ & validation/
data_dir = Path(path)
if (data_dir / "Data").is_dir():
    data_dir = data_dir / "Data"

print("Using data directory:", data_dir)
print("Train classes:", [p.name for p in (data_dir/"train").iterdir() if p.is_dir()])
print("Validation classes:", [p.name for p in (data_dir/"validation").iterdir() if p.is_dir()])


!yolo train task=classify \
    data={data_dir} \
    model=yolov8n-cls.pt \
    epochs=20 \
    batch=16 \
    imgsz=224 \
    lr0=0.01 \
    lrf=0.2 \
    project=/content/runs/colour_classify \
    name=yolov8n-colour \
    exist_ok=True


!ls /content/runs/colour_classify/yolov8n-colour/weights

!yolo val task=classify \
    data={data_dir} \
    model=/content/runs/colour_classify/yolov8n-colour/weights/best.pt \
    batch=16 \
    imgsz=224


#Manual accuracy calculation

from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load best checkpoint
weights_path = Path("/content/runs/colour_classify/yolov8n-colour/weights/best.pt")
model = YOLO(str(weights_path))
val_dir    = data_dir / "validation"
train_dir  = data_dir / "train"
class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])

# Loop over every image in each class folder
y_true, y_pred = [], []
for idx, cls in enumerate(class_names):
    for img_path in (val_dir/cls).iterdir():
        y_true.append(idx)
        res = model.predict(str(img_path), imgsz=224, verbose=False)[0]
        y_pred.append(int(res.probs.argmax()))

# Compute & print metrics
acc = accuracy_score(y_true, y_pred)
print(f"\nManual Validation Accuracy: {acc:.2%}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


'''
Validation top accuracy: 98 %
With a batch of 300 images


Train Top-1 Acc: ≈ 99.8 %
Val Top-1 Acc: 98 %
Test Top-1 Acc: 98 %

There is ~1.8 % gap between train and val/test, so there’s no significant over-fitting.
'''

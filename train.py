from ultralytics import YOLO

EPOCHS = 10
IMGSZ = 96

RUNS = [
    ("baseline", "datasets/emotions/data.yaml"),
    ("clahe", "datasets/emotions_clahe/data.yaml"),
]

for name, data in RUNS:
    model = YOLO("yolov8n.pt")
    model.train(
        data=data,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=16,
        device="cpu",
        seed=42,
        name=name,
    )
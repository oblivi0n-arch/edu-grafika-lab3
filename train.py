from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="datasets/emotions/data.yaml",
    epochs=10,
    imgsz=320,
    batch=16,
    device="cpu",
    name="baseline",
)
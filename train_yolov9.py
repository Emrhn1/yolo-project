from ultralytics import YOLO

model = YOLO("yolov9s.pt")  # YOLOv9 small model

model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    device=0,   # 👈 GPU
    batch=8
)

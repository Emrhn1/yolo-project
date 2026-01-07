from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # küçük model, hızlı

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    device=0,   # 👈 GPU
    batch=16
)

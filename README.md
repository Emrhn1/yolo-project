# Brain Tumor Detection Using YOLO Object Detection Models

<div align="center">

**Comparative Analysis of YOLOv8, YOLOv9, and YOLOv10 for Medical Image Object Detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

</div>

---

## 📋 Project Overview

This project implements and compares three state-of-the-art YOLO (You Only Look Once) object detection models for automated brain tumor detection in medical imaging. The goal is to evaluate the performance trade-offs between **YOLOv8s**, **YOLOv9n**, and **YOLOv10s** on the same brain tumor dataset.

### Medical Context

Brain tumors are among the most serious medical conditions requiring early and accurate detection for effective treatment planning. This project leverages deep learning-based object detection to automatically identify and classify three types of brain tumors:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **Pituitary** - A tumor that forms in the pituitary gland

Unlike traditional classification approaches, object detection provides both tumor localization and classification, which is crucial for medical diagnosis and treatment planning.

---

## 📊 Dataset

### Source
**Brain Tumor Dataset** from [Roboflow Universe](https://universe.roboflow.com/academia-keleu/brain-tumor-bb6yj/dataset/1)

- **Workspace**: academia-keleu
- **Project**: brain-tumor-bb6yj
- **Version**: 1
- **License**: CC BY 4.0
- **Date**: 2024-07-04

### Classes
The dataset contains **3 tumor classes**:
```python
names: ['glioma', 'meningioma', 'pituitary']
```

### Dataset Structure
```
dataset/
├── train/
│   ├── images/     # ~2,062 training images
│   └── labels/     # YOLO format annotations
├── valid/
│   ├── images/     # ~612 validation images
│   └── labels/     # YOLO format annotations
└── test/
    ├── images/     # ~308 test images
    └── labels/     # YOLO format annotations
```

### Data Configuration (`data.yaml`)
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['glioma', 'meningioma', 'pituitary']
```

---

## 🛠️ Training Setup

### Environment
- **Python Version**: 3.8+
- **Framework**: Ultralytics YOLO
- **Hardware**: NVIDIA GPU (CUDA enabled)
  - Device: `cuda:0` (GPU acceleration)
- **OS**: Windows

### Installation
```bash
pip install ultralytics
```

### Training Configuration

All three models were trained with consistent parameters for fair comparison:

| Parameter | YOLOv8n | YOLOv9s | YOLOv10s |
|-----------|---------|---------|----------|
| **Epochs** | 30 | 30 | 30 |
| **Batch Size** | 16 | 8 | 8 |
| **Image Size** | 640×640 | 640×640 | 640×640 |
| **Device** | GPU (cuda:0) | GPU (cuda:0) | GPU (cuda:0) |
| **Optimizer** | Auto | Auto | Auto |
| **Pretrained** | Yes (COCO) | Yes (COCO) | Yes (COCO) |
| **Workers** | 8 | 8 | 8 |
| **AMP** | True | True | True |

### Training Commands

#### YOLOv8n
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load pretrained model
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    batch=16
)
```

#### YOLOv9s
```python
from ultralytics import YOLO

model = YOLO("yolov9s.pt")
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    batch=8
)
```

#### YOLOv10s
```python
from ultralytics import YOLO

model = YOLO("yolov10s.pt")
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    batch=8
)
```

---

## 🏗️ Model Architectures

### YOLOv8s (Small)
- **Variant**: YOLOv8 Nano (lightweight version)
- **Focus**: Balance between speed and accuracy
- **Architecture**: CSPDarknet backbone with PAN-FPN neck
- **Why chosen**: Ideal baseline for real-time applications with minimal computational overhead

### YOLOv9n (Nano)
- **Variant**: YOLOv9 Small
- **Focus**: Improved feature extraction with GELAN architecture
- **Architecture**: Programmable Gradient Information (PGI) + GELAN
- **Why chosen**: Latest advancements in object detection with better gradient flow

### YOLOv10s (Small)
- **Variant**: YOLOv10 Small
- **Focus**: NMS-free architecture for end-to-end detection
- **Architecture**: Dual assignments for NMS-free training
- **Why chosen**: Cutting-edge model with reduced post-processing overhead

---

## 📈 Training Results

### YOLOv8n Training Performance

**Final Metrics (Epoch 30)**
| Metric | Value |
|--------|-------|
| Precision (B) | 92.17% |
| Recall (B) | 83.87% |
| mAP50 (B) | 91.78% |
| **mAP50-95 (B)** | **71.73%** |
| Box Loss (val) | 0.9375 |
| Class Loss (val) | 0.6628 |
| DFL Loss (val) | 1.1952 |

**Training Curves**

![Training Results](runs/detect/yolov8s/train3/results.png)

**Confusion Matrix**

![Confusion Matrix](runs/detect/yolov8s/train3/confusion_matrix.png)

![Normalized Confusion Matrix](runs/detect/yolov8s/train3/confusion_matrix_normalized.png)

**Performance Curves**

| Precision | Recall | F1-Score | PR Curve |
|:---------:|:------:|:--------:|:--------:|
| ![P](runs/detect/yolov8s/train3/BoxP_curve.png) | ![R](runs/detect/yolov8s/train3/BoxR_curve.png) | ![F1](runs/detect/yolov8s/train3/BoxF1_curve.png) | ![PR](runs/detect/yolov8s/train3/BoxPR_curve.png) |

---

### YOLOv9s Training Performance

**Final Metrics (Epoch 30)**
| Metric | Value |
|--------|-------|
| Precision (B) | 88.96% |
| Recall (B) | 86.19% |
| mAP50 (B) | 91.35% |
| **mAP50-95 (B)** | **70.79%** |
| Box Loss (val) | 0.9453 |
| Class Loss (val) | 0.6577 |
| DFL Loss (val) | 1.3971 |

**Training Curves**

![Training Results](runs/detect/yolov9n/train/results.png)

**Confusion Matrix**

![Confusion Matrix](runs/detect/yolov9n/train/confusion_matrix.png)

![Normalized Confusion Matrix](runs/detect/yolov9n/train/confusion_matrix_normalized.png)

**Performance Curves**

| Precision | Recall | F1-Score | PR Curve |
|:---------:|:------:|:--------:|:--------:|
| ![P](runs/detect/yolov9n/train/BoxP_curve.png) | ![R](runs/detect/yolov9n/train/BoxR_curve.png) | ![F1](runs/detect/yolov9n/train/BoxF1_curve.png) | ![PR](runs/detect/yolov9n/train/BoxPR_curve.png) |

---

### YOLOv10s Training Performance

**Final Metrics (Epoch 30)**
| Metric | Value |
|--------|-------|
| Precision (B) | 86.10% |
| Recall (B) | 85.74% |
| mAP50 (B) | 90.54% |
| **mAP50-95 (B)** | **68.44%** |
| Box Loss (val) | 1.9826 |
| Class Loss (val) | 1.3560 |
| DFL Loss (val) | 2.5260 |

**Training Curves**

![Training Results](runs/detect/yolov10/results.png)

**Confusion Matrix**

![Confusion Matrix](runs/detect/yolov10/confusion_matrix.png)

![Normalized Confusion Matrix](runs/detect/yolov10/confusion_matrix_normalized.png)

**Performance Curves**

| Precision | Recall | F1-Score | PR Curve |
|:---------:|:------:|:--------:|:--------:|
| ![P](runs/detect/yolov10/BoxP_curve.png) | ![R](runs/detect/yolov10/BoxR_curve.png) | ![F1](runs/detect/yolov10/BoxF1_curve.png) | ![PR](runs/detect/yolov10/BoxPR_curve.png) |

---

## 💾 Model Weights

Each trained model produces two weight files:

### `best.pt` - Best Performing Model
- Saved when validation mAP50 achieves peak performance
- **Recommended for deployment and inference**
- Used for final model evaluation

### `last.pt` - Last Epoch Checkpoint
- Model state after completing all 30 epochs
- Useful for resuming training or debugging
- May not have optimal performance

### Weight Locations
```
runs/detect/
├── yolov8s/train3/weights/
│   ├── best.pt        # YOLOv8 best model
│   └── last.pt
├── yolov9n/train/weights/
│   ├── best.pt        # YOLOv9 best model
│   └── last.pt
└── yolov10/weights/
    ├── best.pt        # YOLOv10 best model
    └── last.pt
```

### Why Use `best.pt`?
The `best.pt` model is preferred because:
1. **Optimal Performance**: Represents the epoch with highest validation mAP
2. **Generalization**: Avoids overfitting that may occur in later epochs
3. **Production Ready**: Pre-validated for real-world deployment

---

## 🔍 Evaluation

### Validation Commands

#### YOLOv8n
```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolov8s/train3/weights/best.pt")
metrics = model.val(data="dataset/data.yaml")
```

#### YOLOv9s
```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolov9n/train/weights/best.pt")
metrics = model.val(data="dataset/data.yaml")
```

#### YOLOv10s
```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolov10/weights/best.pt")
metrics = model.val(data="dataset/data.yaml")
```

### Evaluation Metrics Explained

| Metric | Description |
|--------|-------------|
| **Precision** | Ratio of correct positive predictions to total positive predictions |
| **Recall** | Ratio of correct positive predictions to total actual positives |
| **mAP50** | Mean Average Precision at IoU threshold 0.5 |
| **mAP50-95** | Mean Average Precision averaged over IoU thresholds 0.5 to 0.95 (primary metric) |
| **Box Loss** | Bounding box regression loss |
| **Class Loss** | Classification loss |
| **DFL Loss** | Distribution Focal Loss for bounding box quality |

---

## 🚀 Inference

### Run Detection on New Images

#### Using YOLOv8 (Best Model)
```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/yolov8s/train3/weights/best.pt")

# Run inference
results = model.predict(
    source="path/to/brain/scan.jpg",
    conf=0.25,      # Confidence threshold
    save=True,      # Save annotated images
    device=0        # Use GPU
)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    print(f"Detected {len(boxes)} tumors")
```

#### CLI Inference
```bash
# YOLOv8
yolo detect predict model=runs/detect/yolov8s/train3/weights/best.pt source=path/to/images conf=0.25

# YOLOv9
yolo detect predict model=runs/detect/yolov9n/train/weights/best.pt source=path/to/images conf=0.25

# YOLOv10
yolo detect predict model=runs/detect/yolov10/weights/best.pt source=path/to/images conf=0.25
```

### Batch Inference on Test Set
```python
from ultralytics import YOLO

model = YOLO("runs/detect/yolov8s/train3/weights/best.pt")
results = model.predict(
    source="dataset/test/images",
    save=True,
    save_txt=True,  # Save labels
    conf=0.25
)
```

---

## 📊 Model Comparison

### Performance Summary

| Model | mAP50-95 | mAP50 | Precision | Recall | Training Time* | Batch Size |
|-------|----------|-------|-----------|--------|----------------|------------|
| **YOLOv8n** | **71.73%** ✅ | **91.78%** | **92.17%** | 83.87% | ~1,170s | 16 |
| **YOLOv9s** | **70.79%** | **91.35%** | 88.96% | **86.19%** | ~3,286s | 8 |
| **YOLOv10s** | 68.44% | 90.54% | 86.10% | **85.74%** | ~3,363s | 8 |

<sub>*Training time for 30 epochs on GPU (total elapsed time)</sub>

### Key Observations

#### 🏆 Best Overall Accuracy: YOLOv8n
- **Highest mAP50-95** (71.73%) - best at precise localization
- **Highest mAP50** (91.78%) - excellent at basic detection
- **Highest Precision** (92.17%) - most reliable positive predictions
- **Fastest Training** - 2.8× faster than YOLOv9 and YOLOv10

#### ⚡ Best Recall: YOLOv9s
- **Highest Recall** (86.19%) - detects more tumors, fewer false negatives
- Competitive mAP50-95 (70.79%)
- Better at not missing tumors (critical for medical use)

#### 🔄 Most Balanced: YOLOv10s
- Best **Precision-Recall balance** (86.10% / 85.74%)
- NMS-free architecture reduces post-processing
- Slightly lower mAP but more consistent predictions

### Performance vs. Speed Trade-offs

```
YOLOv8n:  ████████████ 71.73% mAP | Speed: ★★★★★ (Fastest)
YOLOv9s:  ███████████  70.79% mAP | Speed: ★★☆☆☆ (Slower)
YOLOv10s: ██████████   68.44% mAP | Speed: ★★☆☆☆ (Slower)
```

### Recommendations by Use Case

| Use Case | Recommended Model | Reasoning |
|----------|------------------|-----------|
| **Clinical Deployment** | YOLOv9s | High recall minimizes missed detections |
| **Real-time Screening** | YOLOv8n | Best speed-accuracy trade-off |
| **Research/Analysis** | YOLOv8n | Highest precision for detailed studies |
| **Embedded Systems** | YOLOv8n | Lightweight, efficient, fast inference |
| **Production Pipeline** | YOLOv10s | Balanced performance, NMS-free |

---

## 🎯 Conclusion

This comparative study demonstrates that:

1. **YOLOv8n achieves the best overall performance** with 71.73% mAP50-95, making it ideal for most brain tumor detection applications where both accuracy and speed matter.

2. **YOLOv9s offers superior recall** (86.19%), which is critical in medical imaging where missing a tumor (false negative) is more costly than a false positive.

3. **YOLOv10s provides the most balanced predictions** with near-equal precision and recall, though at slightly lower overall accuracy.

4. **Training efficiency varies significantly**: YOLOv8n trains nearly 3× faster than YOLOv9 and YOLOv10 with similar or better performance.

### Medical Implications

For medical applications, the choice depends on priorities:
- **Maximize detection rate**: Choose YOLOv9s (highest recall)
- **Minimize false positives**: Choose YOLOv8n (highest precision)  
- **Balance both**: Choose YOLOv10s (balanced precision-recall)

All three models achieve >90% mAP50 and >68% mAP50-95, demonstrating the viability of YOLO-based object detection for automated brain tumor detection tasks.

---

## 📁 Project Structure

```
yolo-project/
│
├── dataset/                       # Brain tumor dataset
│   ├── train/                    # Training set (~2,062 images)
│   │   ├── images/
│   │   └── labels/
│   ├── valid/                    # Validation set (~612 images)
│   │   ├── images/
│   │   └── labels/
│   ├── test/                     # Test set (~308 images)
│   │   ├── images/
│   │   └── labels/
│   ├── data.yaml                 # Dataset configuration
│   ├── README.dataset.txt
│   └── README.roboflow.txt
│
├── runs/                          # Training outputs
│   └── detect/
│       ├── yolov8s/train3/       # YOLOv8 experiment
│       │   ├── weights/
│       │   │   ├── best.pt       # Best YOLOv8 model
│       │   │   └── last.pt
│       │   ├── confusion_matrix.png
│       │   ├── results.png
│       │   ├── results.csv
│       │   └── args.yaml
│       │
│       ├── yolov9n/train/        # YOLOv9 experiment
│       │   ├── weights/
│       │   │   ├── best.pt       # Best YOLOv9 model
│       │   │   └── last.pt
│       │   ├── confusion_matrix.png
│       │   ├── results.png
│       │   ├── results.csv
│       │   └── args.yaml
│       │
│       ├── yolov10/              # YOLOv10 experiment
│       │   ├── weights/
│       │   │   ├── best.pt       # Best YOLOv10 model
│       │   │   └── last.pt
│       │   ├── confusion_matrix.png
│       │   ├── results.png
│       │   ├── results.csv
│       │   └── args.yaml
│       │
│       ├── val/                  # Validation results
│       └── val2/
│
├── train_yolov8.py               # Training script example
├── yolov8n.pt                    # Pretrained YOLOv8 weights
├── yolov9s.pt                    # Pretrained YOLOv9 weights
├── yolov10s.pt                   # Pretrained YOLOv10 weights
├── yolo11n.pt                    # Pretrained YOLO11 weights
└── README.md                      # This file
```

---

## 📚 References

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **YOLOv9 Paper**: [YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](https://arxiv.org/abs/2402.13616)
- **YOLOv10 Paper**: [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- **Dataset**: [Brain Tumor - Roboflow Universe](https://universe.roboflow.com/academia-keleu/brain-tumor-bb6yj/dataset/1)

---

## 📝 License

- **Code**: This project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) under AGPL-3.0 license
- **Dataset**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## 👥 Acknowledgments

- **Dataset Provider**: academia-keleu @ Roboflow Universe
- **Framework**: Ultralytics YOLO team
- **YOLO Creators**: Redmon et al., and subsequent contributors to YOLOv8/v9/v10

---

<div align="center">

**🧠 Brain Tumor Detection with YOLO**

*Leveraging State-of-the-Art Object Detection for Medical Imaging*

</div>

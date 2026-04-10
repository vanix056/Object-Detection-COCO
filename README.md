# Object Detection with SSD MobileNet V3 on COCO

## Overview

This project implements **real-time object detection** using a pre-trained SSD MobileNet V3 model on the COCO dataset. It captures a live webcam feed, runs inference using OpenCV's DNN module, and overlays bounding boxes, class labels, and confidence scores directly on the video frames. The solution is lightweight, dependency-minimal, and runs entirely on CPU without requiring a dedicated GPU — making it accessible for prototyping and edge deployments.

## Key Features

- Real-time object detection via webcam using a pre-trained SSD MobileNet V3 model
- Detects **80 object categories** from the COCO dataset (people, vehicles, animals, household items, etc.)
- Displays bounding boxes, class labels, and confidence percentages on live frames
- Configurable confidence threshold to filter low-confidence detections
- Pure OpenCV-based inference — no TensorFlow runtime required at execution time
- Interactive Jupyter Notebook for experimentation and prototyping

## Tech Stack

| Category     | Technology                          |
|--------------|-------------------------------------|
| Language     | Python 3.11                         |
| Framework    | Jupyter Notebook                    |
| Libraries    | OpenCV (`cv2`) with DNN module      |
| Model Format | TensorFlow frozen graph (`.pb`)     |
| Dataset      | COCO (Common Objects in Context)    |
| Tools        | Jupyter Lab / Notebook              |

## Installation

### Prerequisites

- Python 3.8 or higher
- A connected webcam
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/MAbdullahWaqar/Object-Detection-COCO.git
cd Object-Detection-COCO

# 2. Install dependencies
pip install opencv-python jupyter
```

> All model files (`frozen_inference_graph.pb`, `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, `coco.names`) are already included in the repository.

## Usage

### Running via Jupyter Notebook

```bash
jupyter notebook OD.ipynb
```

Open `OD.ipynb` in the browser and execute the cell. The notebook will:

1. Open the default webcam (`cv2.VideoCapture(1)` — change to `0` if your primary camera is not detected)
2. Load the SSD MobileNet V3 model and COCO class labels
3. Run inference on each captured frame
4. Display a live window titled **Output** with annotated detections

**To stop detection:** Press `q` while the output window is focused.

### Webcam Index

If the detection window does not open, change the capture index in the notebook:

```python
# Try index 0 for the default built-in webcam
cap = cv2.VideoCapture(0)
```

### Confidence Threshold

To adjust sensitivity, modify the `confThreshold` parameter:

```python
ClassIds, confs, bbox = model.detect(img, confThreshold=0.5)
```

Increase the value (e.g., `0.6`) to reduce false positives, or decrease it (e.g., `0.3`) to detect more objects.

## Project Structure

```
Object-Detection-COCO/
├── OD.ipynb                                    # Main Jupyter Notebook with detection pipeline
├── frozen_inference_graph.pb                   # Pre-trained SSD MobileNet V3 model weights
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt # Model configuration / network definition
└── coco.names                                  # 80 COCO object class labels
```

## Model Architecture

| Property           | Details                                      |
|--------------------|----------------------------------------------|
| Architecture       | SSD (Single Shot MultiBox Detector)          |
| Backbone           | MobileNet V3 Large                           |
| Input Resolution   | 320 × 320 pixels                             |
| Input Normalization| Scale: `1/127.5`, Mean: `(127.5, 127.5, 127.5)`, RGB swap enabled |
| Output             | Class IDs, confidence scores, bounding boxes |
| Model Date         | January 14, 2020                             |

The model uses a single forward pass to simultaneously predict object classes and bounding box locations, making it well-suited for real-time inference.

## Dataset

The model is pre-trained on the **COCO (Common Objects in Context)** dataset, which contains over 330,000 images across **80 object categories**, including:

- People and body parts
- Vehicles (car, bus, truck, bicycle, etc.)
- Animals (dog, cat, bird, elephant, etc.)
- Food items (pizza, banana, sandwich, etc.)
- Household objects (chair, laptop, clock, etc.)

Full class list is available in [`coco.names`](coco.names).

## Configuration

No environment variables are required. All configuration is handled within the notebook:

| Parameter        | Location in Code                         | Description                          |
|------------------|------------------------------------------|--------------------------------------|
| `confThreshold`  | `model.detect(..., confThreshold=0.5)`   | Minimum confidence to display a detection |
| Input size       | `model.setInputSize(320, 320)`           | Model input resolution               |
| Camera index     | `cv2.VideoCapture(1)`                    | Webcam device index                  |
| Model files      | Top of detection cell                    | Paths to `.pb`, `.pbtxt`, `.names`   |

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Open a Pull Request describing what was changed and why

Please ensure your changes are tested against a live webcam feed before submitting.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Author

**Muhammad Abdullah Waqar**
- GitHub: [@MAbdullahWaqar](https://github.com/MAbdullahWaqar)

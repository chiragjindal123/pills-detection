# 🔬 Pills Detection System

A real-time capsule/pill detection system using YOLOv8 and ONNX Runtime for high-performance inference. This project provides live camera detection with optimizations for both CPU and GPU acceleration.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Detection Scripts](#detection-scripts)
- [Model Information](#model-information)
- [Performance Optimization](#performance-optimization)
- [Field of View Measurement](#field-of-view-measurement)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- **Real-time Detection**: Live camera feed with capsule/pill detection
- **High Performance**: Optimized for both CPU and GPU inference
- **Multiple Detection Modes**: Basic, optimized, GPU-accelerated, and FOV measurement
- **Field of View Calculation**: Measure detection area and capsule sizes in real-world units
- **Frame Rate Optimization**: Adaptive frame skipping for smooth performance
- **ONNX Model Support**: Cross-platform compatibility with ONNX Runtime
- **Configurable Parameters**: Adjustable confidence thresholds and NMS settings

## 📁 Project Structure

```
pills-detection/
├── 📜 README.md                    # Project documentation
├── 📋 requirements.txt             # Python dependencies
├── 🤖 best_model.onnx             # Trained YOLOv8 model
├── 🔧 detection.py                # Basic detection script
├── ⚡ optimize_detection.py        # CPU-optimized detection
├── 🚀 gpu.py                      # GPU-accelerated detection
├── 📏 table.py                    # FOV measurement detection
├── 🖼️  raw_images_processing.py    # Image preprocessing utilities
├── 📓 yolo_training.ipynb          # Model training notebook
└── 🙈 .gitignore                  # Git ignore file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- (Optional) NVIDIA GPU with CUDA for acceleration

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd pills-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU acceleration** (optional):
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

4. **Verify camera access**:
   ```bash
   python -c "import cv2; print('Camera available:', cv2.VideoCapture(0).isOpened())"
   ```

## 🎯 Usage

### Quick Start
```bash
python optimize_detection.py
```

### Available Scripts

#### 1. **Basic Detection** (`detection.py`)
Standard detection with default settings:
```bash
python detection.py
```
- **Resolution**: 640×480
- **Frame Processing**: Every frame
- **Best For**: High accuracy, slower performance

#### 2. **Optimized Detection** (`optimize_detection.py`) ⭐ **Recommended**
CPU-optimized with performance enhancements:
```bash
python optimize_detection.py
```
- **Resolution**: 480×360 (or camera default)
- **Frame Processing**: Every 3rd frame
- **Optimizations**: Vectorized processing, reduced resolution
- **Best For**: Balanced performance and accuracy

#### 3. **GPU Accelerated** (`gpu.py`)
GPU-powered detection for maximum performance:
```bash
python gpu.py
```
- **Resolution**: Adaptive (higher with GPU)
- **Acceleration**: CUDA/GPU if available
- **Fallback**: CPU optimization if no GPU
- **Best For**: Maximum performance with compatible hardware

#### 4. **FOV Measurement** (`table.py`)
Detection with field-of-view measurement capabilities:
```bash
python table.py
```
- **Features**: Real-world size measurement, grid overlay
- **Calibration**: Camera height input required
- **Best For**: Quality control, size verification

## 🎮 Controls

### Universal Controls (All Scripts)
- **`q`**: Quit application
- **`Esc`**: Alternative quit method

### FOV Measurement Controls (`table.py`)
- **`g`**: Toggle measurement grid
- **`r`**: Recalibrate camera height

### GPU Detection Controls (`gpu.py`)
- **`+/=`**: Zoom in
- **`-/_`**: Zoom out
- **`r`**: Reset zoom to 1.0x

## 🤖 Model Information

### YOLOv8 ONNX Model (`best_model.onnx`)
- **Architecture**: YOLOv8n (Nano) optimized for real-time inference
- **Input Size**: 640×640 pixels
- **Classes**: Capsule/Pill detection
- **Format**: ONNX for cross-platform compatibility
- **Batch Size**: 8 (automatically handled)

### Model Performance
| Device | FPS Range | Resolution | Frame Skip |
|--------|-----------|------------|------------|
| CPU Only | 8-15 FPS | 480×360 | Every 3rd |
| GPU (RTX/GTX) | 20-30 FPS | 640×480 | Every 1-2 |
| High-end CPU | 12-20 FPS | 480×360 | Every 2nd |

## ⚡ Performance Optimization

### Automatic Optimizations
1. **Adaptive Frame Skipping**: Based on camera resolution
2. **Vectorized Processing**: NumPy operations for speed
3. **Memory Management**: Optimized ONNX Runtime settings
4. **Resolution Scaling**: Automatic downscaling for better FPS

### Manual Tuning
```python
# In any detection script, adjust these parameters:
confidence_threshold = 0.5  # Lower = more detections, higher = fewer false positives
nms_threshold = 0.5        # Non-maximum suppression threshold
frame_skip = 3             # Process every Nth frame (higher = better FPS)
```

### Camera Resolution Override
```python
# Force specific resolution (in main function)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Lower for better FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Lower for better FPS
```

## 📏 Field of View Measurement

The `table.py` script provides real-world measurement capabilities:

### Setup
1. **Measure camera height** above the table surface
2. **Run the script**: `python table.py`
3. **Enter height** when prompted
4. **View calibration results**

### Measurement Features
- **Table Coverage**: Shows total area visible to camera
- **Grid Overlay**: 5cm measurement grid
- **Capsule Sizing**: Real-time size measurement in cm
- **Pixel Density**: Pixels per centimeter calculation

### Example Output
```
📏 FOV Calibration Complete:
  Camera height: 29.0 cm
  Table coverage: 37.0 × 19.2 cm
  Resolution: 52.0 × 56.4 pixels/cm
  Total area: 708 cm²
```

## 🔧 Configuration

### Detection Parameters
```python
class CapsuleDetector:
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.5,  # Detection confidence
                 nms_threshold: float = 0.5):        # NMS threshold
```

### Camera Settings
```python
# Resolution (adjust based on performance needs)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Performance settings
cap.set(cv2.CAP_PROP_FPS, 30)           # Target FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # Reduce lag
```

## 🐛 Troubleshooting

### Common Issues

#### Low FPS Performance
```bash
# Solution 1: Use optimized detection
python optimize_detection.py

# Solution 2: Lower camera resolution (edit script)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Solution 3: Increase frame skipping
frame_skip_counter % 5 == 0  # Process every 5th frame
```

#### Camera Resolution Issues
```python
# Check actual camera resolution
ret, frame = cap.read()
print(f"Actual resolution: {frame.shape[1]}x{frame.shape[0]}")

# Force resize if camera won't change resolution
frame = cv2.resize(frame, (480, 360))
```

#### GPU Not Detected
```bash
# Check ONNX Runtime providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Install GPU version
pip uninstall onnxruntime
pip install onnxruntime-gpu

# Verify CUDA installation
nvidia-smi
```

#### Model Loading Errors
```bash
# Check model file exists
ls -la best_model.onnx

# Verify ONNX Runtime installation
pip install --upgrade onnxruntime
```

### Performance Benchmarking
```python
# Add timing to any script
import time

start_time = time.time()
detections = detector.detect(frame)
inference_time = time.time() - start_time
print(f"Inference time: {inference_time*1000:.1f}ms")
```

## 📊 Expected Performance

### Typical Results
| Script | Resolution | FPS | Use Case |
|--------|------------|-----|----------|
| `detection.py` | 640×480 | 5-10 | High accuracy |
| `optimize_detection.py` | 480×360 | 10-15 | **Recommended** |
| `gpu.py` | 640×480 | 20-30 | Maximum performance |
| `table.py` | 480×360 | 8-12 | Quality control |

### Hardware Requirements
- **Minimum**: Intel i5/AMD Ryzen 5, 8GB RAM, integrated graphics
- **Recommended**: Intel i7/AMD Ryzen 7, 16GB RAM, dedicated GPU
- **Optimal**: Modern CPU + NVIDIA RTX/GTX GPU with CUDA support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing GitHub issues
3. Create a new issue with detailed description

## 🔗 Related Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

**Made with ❤️ for real-time computer vision applications**
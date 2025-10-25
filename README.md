# 🧠 ObjectDetectionFromScratch

This project explores multiple approaches to object detection from scratch using OpenCV and deep learning models. It starts with edge-based heuristics and evolves into feature map extraction and classification using pretrained CNNs. Future updates will extend these methods to video streams and real-time applications.

---

## 📦 File Overview

### `Simple.py`
A pure OpenCV-based approach:
- Detects objects by identifying continuous edges.
- Uses Canny edge detection and contour bounding boxes.
- No deep learning involved—ideal for fast, naive detection.

### `Naive.py`
A hybrid edge + classification pipeline:
- Detects edges and crops regions with contours.
- Passes each cropped region to a pretrained model (e.g., ResNet50).
- Classifies each region and annotates the original image.

### `Advanced.py`
Feature-driven object detection:
- Extracts feature maps from `block1_conv1` of VGG19.
- Uses contours on the feature map to generate bounding boxes.
- Filters small regions and classifies each using ResNet50.
- Annotates predictions with bounding boxes and labels.

### `test.py`
Custom-layer enhanced feature extraction:
- Adds a custom `Conv2D` layer (e.g., `Conv2D(1024, (3,3), activation='relu')`) after a selected VGG19 layer.
- Improves feature richness for downstream detection.
- Useful for experimentation and fine-tuning feature maps.

---

## 🧠 Workflow Summary (Advanced.py)

1. **Feature Extraction**  
   Extracts spatial features from VGG19 (without top layer).

2. **Bounding Box Generation**  
   Converts feature maps to grayscale, applies blur and edge detection, and finds contours.

3. **Region Filtering**  
   Filters out regions with area < 500 pixels.

4. **Classification**  
   Cropped regions are resized, preprocessed, and classified using ResNet50. Predictions with confidence ≥ 80% are annotated.

---

## 🧪 Dependencies

- Python 3.x  
- OpenCV (`cv2`)  
- TensorFlow / Keras  
- NumPy  

Install with:
```bash
pip install opencv-python tensorflow numpy

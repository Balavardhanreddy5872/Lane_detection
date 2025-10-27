# Lane Detection using U-Net

This project implements **lane detection** using a **U-Net deep learning model**.  
The model segments lane markings from road images and overlays them on videos, highlighting detected lanes in **green**.

---

## Project Overview

The system uses a **U-Net Convolutional Neural Network** trained to identify lane markings in real road images.
The model is trained on the original **TuSimple Lane Detection dataset**, which contains real highway driving videos with lane annotations.
Once trained, the model can be used to run inference on any road video to detect and **visualize lane lines** accurately.


```

## Dataset

This project uses the **TuSimple Lane Detection Dataset**, a widely used benchmark for autonomous driving lane detection.

- **Source:** [TuSimple Lane Detection Challenge Dataset](https://github.com/TuSimple/tusimple-benchmark)  
- **Description:** The dataset contains over **6,000 highway driving images** captured under various lighting and weather conditions.  
- **Annotations:** Each image has corresponding **lane line coordinates** labeled across multiple frames, allowing temporal and spatial lane tracking.  

### 🧹 Data Preprocessing
- All input images were **resized to 256×256 pixels** to fit the U-Net input size.  
- The data was divided into **training (80%)** and **validation (20%)** sets.  
- Images were **normalized** to a [0, 1] range for stable neural network training.  
- Data augmentation techniques like **horizontal flips** and **brightness adjustments** were optionally applied to improve model generalization.

```



## 🚀 How to Run

### 1️⃣ Train the Model (optional)

If you want to train from scratch:
```bash
# Run the Jupyter notebook
U_net_implementation.ipynb
```
This will:
- Generate synthetic data.
- Train the U-Net model.
- Save it as `lane_detection_unet.h5`.

---

### 2️⃣ Run Inference with Visualization

To see intermediate results (frames, masks, overlays):
```bash
python lane_detection_inference.py
```
- Input video: `input.mp4`
- Output video: `input_video_with_bottom_45_inference.mp4`
- Displays debug plots (predicted mask, preprocessed frame, etc.)

---




## 🧩 Model Architecture

The **U-Net** consists of:
- **Encoder**: Downsampling layers (Conv + MaxPooling) to extract features.  
- **Decoder**: Upsampling layers (Conv + UpSampling + Skip connections) to reconstruct lane masks.  
- **Output Layer**: 1-channel sigmoid for binary mask prediction.

---


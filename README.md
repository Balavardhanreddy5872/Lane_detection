# 🛣️ Lane Detection using U-Net

This project implements **lane detection** using a **U-Net deep learning model**.  
The model segments lane markings from road images and overlays them on videos, highlighting detected lanes in **green**.

---

## 🧠 Project Overview

The system uses a **U-Net Convolutional Neural Network** trained to identify lane markings in road images.  
Due to the large size of real datasets (like TuSimple), a **synthetic dataset** is generated for training.  
Once trained, the model can be used to run inference on any road video to detect and visualize lanes.


## ⚙️ Requirements

Install dependencies before running any file:

```
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

---

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

### 3️⃣ Run Clean Inference (Final Output)

For fast, clean lane detection:
```bash
python main.py
```
- Input video: `input.mp4`
- Output video: `output_video.mp4`
- Shows final lane detection result only (no debug visuals).


## 🧩 Model Architecture

The **U-Net** consists of:
- **Encoder**: Downsampling layers (Conv + MaxPooling) to extract features.  
- **Decoder**: Upsampling layers (Conv + UpSampling + Skip connections) to reconstruct lane masks.  
- **Output Layer**: 1-channel sigmoid for binary mask prediction.

---

## 📈 Output Example

After running inference, the system outputs a video where:
- Lane markings are highlighted in **green**.
- The model focuses on the **bottom 45% of the frame**, improving road-level accuracy.

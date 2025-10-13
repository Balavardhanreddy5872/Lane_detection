import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('unet_lane_detection_tusimple.h5')


def preprocess_frame(frame, img_size=(256, 256)):
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame / 255.0 
    return resized_frame, normalized_frame.reshape((1, img_size[0], img_size[1], 3))


def postprocess_mask(predicted_mask, original_shape):
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    predicted_mask_binarized = (predicted_mask_resized > 0.5).astype(np.uint8) * 255

    updated_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    bottom_45_percent_start = int(original_shape[0] * 0.65)
    updated_mask[bottom_45_percent_start:, :] = predicted_mask_binarized[bottom_45_percent_start:, :]
    
    return updated_mask


input_video_path = 'input.mp4'
output_video_path = 'input_video_with_bottom_45_inference.mp4'

cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    resized_frame, preprocessed_frame = preprocess_frame(frame)
    if frame_count == 1:
        plt.imshow(preprocessed_frame[0])
        plt.title("Preprocessed Frame")
        plt.show()

    predicted_mask = model.predict(preprocessed_frame)[0]

    if frame_count == 1:
        plt.imshow(predicted_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()

    predicted_mask_resized = postprocess_mask(predicted_mask, frame.shape[:2])

    if frame_count == 1:
        plt.imshow(predicted_mask_resized, cmap='gray')
        plt.title("Resized Mask")
        plt.show()

    overlay = frame.copy()
    overlay[predicted_mask_resized > 0] = [0, 255, 0] 

    out.write(overlay)


cap.release()
out.release()

print(f'Output video saved as {output_video_path}')

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('unet_lane_detection_tusimple.h5')


def preprocess_frame(frame, img_size=(256, 256)):
    frame_resized = cv2.resize(frame, img_size)
    frame_normalized = frame_resized / 255.0 
    return frame_normalized.reshape((1, img_size[0], img_size[1], 3))


def postprocess_mask(predicted_mask, original_shape):
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return predicted_mask_resized


input_video_path = 'input.mp4'
output_video_path = 'output_video.mp4'
cap = cv2.VideoCapture(input_video_path)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    preprocessed_frame = preprocess_frame(frame)

   
    predicted_mask = model.predict(preprocessed_frame)[0]

    
    predicted_mask_resized = postprocess_mask(predicted_mask, frame.shape)

    
    overlay = frame.copy()
    overlay[predicted_mask_resized > 0.5] = [0, 255, 0] 

    
    out.write(overlay)


cap.release()
out.release()

print(f'Output video saved as {output_video_path}')

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import streamlit as st
from threading import Thread
from queue import Queue
import time

# Constants
MODEL_PATH = "/Users/sonam/development/Agahash/facial_emotion_detection_model.h5"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FRAME_SIZE = (48, 48)
UPDATE_FREQUENCY = 2  # Update the emotion text every 2 seconds


# Initialize Streamlit and MediaPipe
st.title("Facial Emotion Recognition")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Load the model with Streamlit cache for efficiency
@st.experimental_singleton
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except OSError as e:
        st.error(f"Model could not be loaded: {e}")
        return None


model = load_model()

# Adjusted face detection for performance
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)


# Preprocess the face ROI
def preprocess_roi(face_roi):
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = cv2.resize(face_roi, FRAME_SIZE)
    face_roi = face_roi / 255.0
    return np.reshape(face_roi, (1, *FRAME_SIZE, 1))


# Predict emotion
def predict_emotion(face_roi):
    processed_roi = preprocess_roi(face_roi)
    emotion_prediction = model.predict(processed_roi)
    emotion_label = np.argmax(emotion_prediction)
    return EMOTIONS[emotion_label]


# Initialize variables
last_emotion_update_time = 0  # Track when the emotion was last updated
current_emotion = None  # Track the current emotion to display


# Function to draw landmarks and annotations
def draw_annotations(image, bbox):
    if bbox:
        center_x, center_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
        radius = max(bbox[2], bbox[3]) // 2
        cv2.circle(image, (center_x, center_y), radius, (255, 105, 180), 2)  # Pink circle


# Frame capture thread
def capture_frames(cap, frame_queue):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame_queue.put(frame)


# Initialize the webcam and frame queue
cap = cv2.VideoCapture(0)
frame_queue = Queue(maxsize=5)

# Start the frame capture thread
capture_thread = Thread(target=capture_frames, args=(cap, frame_queue), daemon=True)
capture_thread.start()


# Processing and displaying frames with a delay for smoothness
def process_and_display(frame_queue, face_detection):
    global last_emotion_update_time, current_emotion
    stframe = st.empty()
    emotion_display = st.empty()  # Placeholder to display emotion text

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            if results.detections:
                detection = results.detections[0]  # Use only the most prominent face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                face_roi = frame[
                    max(0, bbox[1]) : max(0, bbox[1] + bbox[3]),
                    max(0, bbox[0]) : max(0, bbox[0] + bbox[2]),
                ]
                if face_roi.size == 0:
                    continue
                if time.time() - last_emotion_update_time > UPDATE_FREQUENCY:
                    predicted_emotion = predict_emotion(face_roi)
                    current_emotion = predicted_emotion
                    last_emotion_update_time = time.time()
                draw_annotations(frame, bbox)  # Draw pink circle around the face
            stframe.image(frame, channels="BGR", use_column_width=True)
            # Display the current emotion text larger under the camera feed
            emotion_display.markdown(f"<h2 style='text-align: center; color: white;'>{current_emotion}</h2>", unsafe_allow_html=True)
            time.sleep(0.01)  # Adjust for smoother display

# Start processing and display
process_and_display(frame_queue, face_detection)



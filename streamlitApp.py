import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import streamlit as st
from threading import Thread
from queue import Queue
import time
import pandas as pd

# Constants
MODEL_PATH = "/Users/sonam/development/emotionstreamlit/facial_emotion_detection_model.h5"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FRAME_SIZE = (48, 48)
UPDATE_FREQUENCY = 2  # Update the emotion text every 2 seconds

# Initialize Streamlit and MediaPipe
st.title("Facial Emotion Recognition")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except OSError as e:
        st.error(f"Model could not be loaded: {e}")
        return None

model = load_model()

# Page Navigation
page = st.sidebar.selectbox("Navigate", options=["Home", "Live Detection"])

if page == "Home":
    st.write("Utilizing Facial Recognition to Analyze Emotions in Real Time")
    st.markdown("""
        <p style='color:white;'>
            This capstone project, AR07: Real-Time Facial Emotion Recognition, focuses on improving human-computer interaction by accurately recognizing and classifying human emotions from facial expressions in real-time. We are leveraging advanced technologies such as TensorFlow, OpenCV, and MediaPipe to introduce a robust facial expression recognition (FER) system that can operate under diverse environmental conditions and across different demographics.
            <br><br>
            The system's core is a Convolutional Neural Network (CNN), which we have meticulously trained with the FER2013 dataset to recognize seven distinct emotional states: anger, disgust, fear, happiness, neutral, sadness, and surprise. Our model, detailed in the system design, uses real-time processing algorithms to detect and analyze facial expressions with minimal latency accurately. Immediate feedback is crucial for applications requiring dynamic user interaction.
        </p>
        """, unsafe_allow_html=True)
    st.image("demo.png", caption="")



elif page == "Live Detection":
    # Adjusted face detection for performance
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )

    def preprocess_roi(face_roi):
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, FRAME_SIZE)
        face_roi = face_roi / 255.0
        return np.reshape(face_roi, (1, *FRAME_SIZE, 1))

    def predict_emotion_with_prob(face_roi):
        processed_roi = preprocess_roi(face_roi)
        emotion_probabilities = model.predict(processed_roi)[0]
        emotion_index = np.argmax(emotion_probabilities)
        emotion_label = EMOTIONS[emotion_index]
        return emotion_label, emotion_probabilities[emotion_index], emotion_probabilities

    last_emotion_update_time = 0
    current_emotion = None

    def draw_annotations(image, bbox):
        if bbox:
            center_x, center_y = bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2
            radius = max(bbox[2], bbox[3]) // 2
            cv2.circle(image, (center_x, center_y), radius, (255, 105, 180), 2)

    cap = cv2.VideoCapture(0)
    frame_queue = Queue(maxsize=5)

    def capture_frames(cap, frame_queue):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            frame_queue.put(frame)

    capture_thread = Thread(target=capture_frames, args=(cap, frame_queue), daemon=True)
    capture_thread.start()

    def process_and_display(frame_queue, face_detection):
        global last_emotion_update_time, current_emotion
        stframe = st.empty()
        emotion_display = st.empty()
        chart_placeholder = st.empty()

        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                if results.detections:
                    detection = results.detections[0]
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    bbox = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )
                    face_roi = frame[max(0, bbox[1]):max(0, bbox[1] + bbox[3]),
                                    max(0, bbox[0]):max(0, bbox[0] + bbox[2])]
                    if face_roi.size == 0:
                        continue
                    if time.time() - last_emotion_update_time > UPDATE_FREQUENCY:
                        emotion_label, highest_prob, emotion_probabilities = predict_emotion_with_prob(face_roi)
                        current_emotion = emotion_label
                        last_emotion_update_time = time.time()

                        chart_data = pd.DataFrame({
                            'Emotion': EMOTIONS,
                            'Probability': emotion_probabilities
                        }).set_index('Emotion')
                        chart_placeholder.bar_chart(chart_data)

                    draw_annotations(frame, bbox)
                stframe.image(frame, channels="BGR", use_column_width=True)
                emotion_display.markdown(f"<h2 style='text-align: center; color: white;'>{current_emotion}</h2>",
                                         unsafe_allow_html=True)
                time.sleep(0.01)  # Adjust for smoother display

    # Call the function to start processing and displaying frames
    process_and_display(frame_queue, face_detection)


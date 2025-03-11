import streamlit as st
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Emojify - Emotion Recognition", page_icon="😊", layout="wide"
)


# Load the pre-trained emotion classification model
@st.cache_resource
def load_model():
    model = keras.models.load_model("emotion_classifier.keras")
    return model


# Load face detector from dlib
@st.cache_resource
def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    return detector


# Load emoji images
@st.cache_resource
def load_emoji_images():
    emojis = {}
    emoji_dir = "emojis"
    emoji_files = {
        0: "angry.png",
        1: "disgusted.png",
        2: "fearful.png",
        3: "happy.png",
        4: "sad.png",
        5: "surprised.png",
        6: "neutral.png",
    }

    for emotion_id, file_name in emoji_files.items():
        img_path = os.path.join(emoji_dir, file_name)
        if os.path.exists(img_path):
            emojis[emotion_id] = Image.open(img_path)
        else:
            st.warning(f"Emoji file {img_path} not found!")

    return emojis


# Emotion labels
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


# Function to preprocess face for prediction
def preprocess_face(face_img):
    # Resize to model input size (48x48)
    face_img = cv2.resize(face_img, (48, 48))

    # Convert to grayscale if it's not already
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to [0, 1]
    face_img = face_img / 255.0

    # Reshape to model input format
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    return face_img


def main():
    st.title("Emojify - Real-time Emotion Recognition")
    st.write("This app recognizes your facial emotion and displays it as an emoji!")

    # Load the model, detector, and emojis
    try:
        model = load_model()
        detector = load_face_detector()
        emojis = load_emoji_images()

        # Add some space
        st.write("")

        # Create a placeholder for the camera feed
        camera_col, emoji_col = st.columns([3, 1])

        with camera_col:
            video_placeholder = st.empty()

        with emoji_col:
            st.subheader("Detected Emotion")
            emoji_placeholder = st.empty()
            emotion_text = st.empty()
            confidence_text = st.empty()

        start_btn = st.button("Start Camera")
        stop_btn = st.button("Stop Camera")

        if start_btn:
            st.session_state.run_camera = True

        if stop_btn:
            st.session_state.run_camera = False

        # Initialize webcam
        if "run_camera" not in st.session_state:
            st.session_state.run_camera = False

        if st.session_state.run_camera:
            cap = cv2.VideoCapture(0)

            while st.session_state.run_camera:
                # Read frame from webcam
                ret, frame = cap.read()

                if not ret:
                    st.error(
                        "Failed to access webcam. Please check your camera and try again."
                    )
                    break

                # Convert to RGB for display in Streamlit
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get faces from the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray_frame)

                # Display a message if no face is detected
                if len(faces) == 0:
                    video_placeholder.image(
                        rgb_frame, channels="RGB", use_container_width=True
                    )
                    emoji_placeholder.empty()
                    emotion_text.write("No face detected")
                    confidence_text.empty()
                    continue

                # If more than one face is detected, show a warning
                if len(faces) > 1:
                    for face in faces:
                        x, y, w, h = (
                            face.left() - 10,
                            face.top() - 10,
                            face.width() + 10,
                            face.height() + 10,
                        )
                        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    video_placeholder.image(
                        rgb_frame, channels="RGB", use_container_width=True
                    )
                    emoji_placeholder.empty()
                    emotion_text.write(
                        "⚠️ Multiple faces detected! Please ensure only one person is in the frame."
                    )
                    confidence_text.empty()
                    continue

                # Process the single face
                face = faces[0]
                x, y, w, h = (
                    face.left() - 10,
                    face.top() - 10,
                    face.width() + 10,
                    face.height() + 10,
                )

                # Extract face ROI
                face_roi = gray_frame[y : y + h, x : x + w]

                if face_roi is None:
                    emotion_text.write("⚠️ No face detected!")
                    continue

                # Preprocess face for the model
                processed_face = preprocess_face(face_roi)

                # Make prediction
                prediction = model.predict(processed_face)[0]
                emotion_id = np.argmax(prediction)
                confidence = prediction[emotion_id] * 100

                # Draw rectangle around face
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the frame
                video_placeholder.image(
                    rgb_frame, channels="RGB", use_container_width=True
                )

                # Display the corresponding emoji
                if emotion_id in emojis:
                    emoji_placeholder.image(emojis[emotion_id], width=150)
                    emotion_text.write(f"Emotion: {emotion_labels[emotion_id]}")
                    confidence_text.write(f"Confidence: {confidence:.2f}%")

                # Slight delay to reduce resource usage
                time.sleep(0.1)

            # Release the webcam when done
            cap.release()

        st.write("---")
        st.write("Click 'Start Camera' to begin emotion detection.")
        st.write(
            "Note: This app works best in good lighting conditions and when your face is clearly visible."
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write(
            "Please make sure the required model files and emoji images are in the correct locations."
        )


if __name__ == "__main__":
    main()

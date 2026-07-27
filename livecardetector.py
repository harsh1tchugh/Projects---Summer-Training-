import cv2
import numpy as np
import streamlit as st

st.title("🚗 Vehicle Detection App")

# Upload video file
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Load Haar cascade for car detection
    car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_car.xml")

    # Open video
    cap = cv2.VideoCapture("temp_video.mp4")

    stframe = st.empty()  # Placeholder for video frames
    vehicle_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            vehicle_detected = True

        # Show frame in Streamlit
        stframe.image(frame, channels="BGR")

    cap.release()

    # Show detection message
    if vehicle_detected:
        st.markdown("<h3 style='color:green;'>✅ Vehicle detected</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red;'>❌ No vehicle detected</h3>", unsafe_allow_html=True)

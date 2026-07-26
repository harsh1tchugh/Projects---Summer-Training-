import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.title("😊 Live Smile Detection App")

st.markdown("Allow camera access below. Detection runs live on each frame.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


class SmileDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Improves contrast, which helps a LOT with smile detection accuracy
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Restrict smile search to the lower half of the face
            # (mouth region) — this alone fixes most false negatives
            roi_gray = gray[y + int(h * 0.5): y + h, x: x + w]
            roi_color = img[y + int(h * 0.5): y + h, x: x + w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.5,
                minNeighbors=15,
                minSize=(25, 25)
            )

            if len(smiles) > 0:
                cv2.putText(img, "Smile Detected!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="smile-detection",
    video_processor_factory=SmileDetector,
    media_stream_constraints={"video": True, "audio": False},
)

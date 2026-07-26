import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp

st.title("😊 Live Smile Detection App (Enhanced)")
st.markdown("Uses facial landmark geometry instead of Haar cascades — much more accurate.")

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for mouth corners and lips (MediaPipe Face Mesh)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_FACE = 234
RIGHT_FACE = 454


class SmileDetector(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        # Smoothing buffer to prevent flicker/false triggers
        self.smile_history = []
        self.history_len = 6

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        is_smiling = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pts = face_landmarks.landmark

                def coord(idx):
                    return np.array([pts[idx].x * w, pts[idx].y * h])

                left_mouth = coord(LEFT_MOUTH)
                right_mouth = coord(RIGHT_MOUTH)
                top_lip = coord(TOP_LIP)
                bottom_lip = coord(BOTTOM_LIP)
                left_face = coord(LEFT_FACE)
                right_face = coord(RIGHT_FACE)

                mouth_width = np.linalg.norm(right_mouth - left_mouth)
                mouth_height = np.linalg.norm(bottom_lip - top_lip)
                face_width = np.linalg.norm(right_face - left_face)

                # Normalize by face width so it works regardless of distance from camera
                mouth_width_ratio = mouth_width / face_width
                mouth_open_ratio = mouth_height / face_width

                # A real smile widens the mouth significantly relative to face width
                # Tune these two thresholds if needed for your face/lighting
                smile_score = mouth_width_ratio > 0.42 and mouth_open_ratio < 0.18

                self.smile_history.append(smile_score)
                if len(self.smile_history) > self.history_len:
                    self.smile_history.pop(0)

                # Require majority of recent frames to agree -> kills flicker/false positives
                is_smiling = sum(self.smile_history) > (self.history_len // 2)

                # Draw face bounding box from landmarks
                xs = [p.x * w for p in pts]
                ys = [p.y * h for p in pts]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw mouth points for visual debugging
                for p in [left_mouth, right_mouth, top_lip, bottom_lip]:
                    cv2.circle(img, tuple(p.astype(int)), 2, (0, 255, 255), -1)

                label = "Smile Detected!" if is_smiling else "No Smile"
                color = (0, 255, 0) if is_smiling else (0, 0, 255)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Optional: show live ratio values for debugging/tuning
                cv2.putText(img, f"width_ratio={mouth_width_ratio:.2f}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="smile-detection",
    video_processor_factory=SmileDetector,
    media_stream_constraints={"video": True, "audio": False},
)

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Vehicle Detection & Counter", page_icon="🚗")

st.title("🚗Live Vehicle Detection & Counting App")
st.markdown("Upload a video to detect and count vehicles using Background Subtraction.")

# --- 1. STREAMLIT UI FOR VIDEO UPLOAD ---
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Read video file into memory
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    temp_filename = "temp_video.mp4"
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)

    cap = cv2.VideoCapture(temp_filename)

    # --- 2. BACKGROUND SUBTRACTOR SETUP ---
    # MOG2 isolates moving objects from the static background
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100, varThreshold=50, detectShadows=True
    )

    # Morphological kernel for noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # --- 3. COUNTING SETUP ---
    line_position_factor = 0.80  # Position line at 80% height
    offset = 8                   # Pixel tolerance buffer around line

    vehicles_tracked = []
    total_vehicles = 0

    stframe = st.empty()
    status_placeholder = st.empty()
    frame_counter = 0

    # --- 4. VIDEO PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        height, width, _ = frame.shape
        line_height = int(height * line_position_factor)

        # 1. Apply Background Subtraction to generate motion mask
        fg_mask = bg_subtractor.apply(frame)

        # 2. Remove shadows (shadows are shaded grey in MOG2, true motion is white = 255)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # 3. Apply Erosion & Dilation to remove small noise dots and seal car shapes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # 4. Find contours of moving blobs
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw green counting line across the road
        cv2.line(frame, (0, line_height), (width, line_height), (0, 255, 0), 2)

        # --- 5. VEHICLE DETECTION & TRACKING ---
        for contour in contours:
            # Filter out small noise blobs (adjust minimum area threshold if needed)
            if cv2.contourArea(contour) < 1200:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Draw Blue Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Calculate Centroid
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)

            # Draw Green Centroid Dot
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # --- 6. LINE CROSSING LOGIC ---
            line_top = line_height - offset
            line_bottom = line_height + offset

            if line_top < cy < line_bottom:
                is_new = True
                for old_cx, old_cy, old_frame in vehicles_tracked:
                    # Prevent double counting if detected across consecutive frames
                    if frame_counter - old_frame < 10 and abs(cx - old_cx) < 60:
                        is_new = False
                        break

                if is_new:
                    total_vehicles += 1
                    vehicles_tracked.append([cx, cy, frame_counter])

        # --- 7. DISPLAY COUNTER ON VIDEO FRAME ---
        cv2.putText(
            frame,
            f"Total Vehicles Detected: {total_vehicles}",
            (20, 80),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0, 255, 0),
            2,
        )

        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

    status_placeholder.success(
        f"Processing complete! Final Count: {total_vehicles} vehicles detected."
    )

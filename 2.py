import cv2
import logging

# Configuration constants
CONFIG = {
    'camera_index': 0,
    'frame_skip': 1,  # Process every frame for smooth detection
    'scale_factor': 0.8,  # Less aggressive downscaling for better accuracy
    'face_scale_factor': 1.3,
    'face_min_neighbors': 5,
    'smile_scale_factor': 1.7,
    'smile_min_neighbors': 15,
    'smile_min_size': (25, 25),
    'smile_persistence_frames': 1,  # Show smile immediately on detection
    'roi_start_ratio': 0.5,  # Start ROI at 50% of face height
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_camera():
    """Initialize camera with error handling"""
    try:
        cap = cv2.VideoCapture(CONFIG['camera_index'])
        if not cap.isOpened():
            logger.error("Cannot connect to camera")
            return None
        logger.info("Camera initialized successfully")
        return cap
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        return None

def initialize_cascades():
    """Initialize Haar cascade classifiers"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        
        if face_cascade.empty() or smile_cascade.empty():
            logger.error("Failed to load cascade classifiers")
            return None, None
            
        logger.info("Cascade classifiers loaded successfully")
        return face_cascade, smile_cascade
    except Exception as e:
        logger.error(f"Cascade initialization failed: {e}")
        return None, None

def detect_faces(gray_frame, face_cascade):
    """Detect faces in grayscale frame"""
    return face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=CONFIG['face_scale_factor'],
        minNeighbors=CONFIG['face_min_neighbors']
    )

def detect_smiles(roi_gray, smile_cascade):
    """Detect smiles in region of interest"""
    return smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=CONFIG['smile_scale_factor'],
        minNeighbors=CONFIG['smile_min_neighbors'],
        minSize=CONFIG['smile_min_size']
    )

def draw_results(frame, faces, smile_detections):
    """Draw bounding boxes and text on frame"""
    for i, (x, y, w, h) in enumerate(faces):
        # Draw face bounding box (Blue)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Show smile if detected
        if i < len(smile_detections) and len(smile_detections[i]) > 0:
            cv2.putText(
                frame,
                "Smile Detected!",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            
            # Draw green bounding box around the mouth
            roi_color_lower = frame[y + int(h * CONFIG['roi_start_ratio']): y + h, x: x + w]
            for (sx, sy, sw, sh) in smile_detections[i]:
                cv2.rectangle(
                    roi_color_lower,
                    (sx, sy),
                    (sx + sw, sy + sh),
                    (0, 255, 0),
                    2
                )
    return frame


def main():
    """Main detection loop"""
    # Initialize components
    cap = initialize_camera()
    if cap is None:
        return
    
    face_cascade, smile_cascade = initialize_cascades()
    if face_cascade is None or smile_cascade is None:
        cap.release()
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                break
            
            frame_count += 1
            
            # Frame skipping for performance
            if frame_count % CONFIG['frame_skip'] != 0:
                cv2.imshow("Smile Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            
            # Downscale for detection
            small_frame = cv2.resize(frame, (0, 0), fx=CONFIG['scale_factor'], fy=CONFIG['scale_factor'])
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detect_faces(gray, face_cascade)
            
            # Scale coordinates back to original frame size
            faces = [(int(x / CONFIG['scale_factor']), int(y / CONFIG['scale_factor']), 
                     int(w / CONFIG['scale_factor']), int(h / CONFIG['scale_factor'])) 
                    for (x, y, w, h) in faces]
            
            # Detect smiles
            smile_detections = []
            
            for (x, y, w, h) in faces:
                roi_gray_lower = gray[
                    int(y * CONFIG['scale_factor']) + int(h * CONFIG['scale_factor'] * CONFIG['roi_start_ratio']): 
                    int((y + h) * CONFIG['scale_factor']),
                    int(x * CONFIG['scale_factor']): 
                    int((x + w) * CONFIG['scale_factor'])
                ]
                
                smiles = detect_smiles(roi_gray_lower, smile_cascade)
                smile_detections.append(smiles)
            
            # Draw results
            frame = draw_results(frame, faces, smile_detections)
            
            cv2.imshow("Smile Detector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        logger.info("Detection stopped by user")
    except Exception as e:
        logger.error(f"Error during detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")

if __name__ == "__main__":
    main()
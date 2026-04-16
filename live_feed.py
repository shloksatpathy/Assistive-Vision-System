import cv2
import torch
import time
import threading
from PIL import Image
import numpy as np

from models.vlm import VLMCaptionGenerator
from models.yolo_detector import YoloDetector
from models.caption_optimizer import CaptionOptimizer

# Global variables for thread synchronization
latest_frame = None
latest_caption = "Initializing models..."
latest_boxes = []
is_running = True

def inference_thread(captioner, detector, optimizer):
    """
    Background thread that continuously grabs the latest frame,
    runs YOLO detection and VLM captioning, and updates the global state.
    """
    global latest_frame, latest_caption, latest_boxes, is_running
    
    while is_running:
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        # Copy frame to avoid race condition and concurrent modification
        frame_to_process = latest_frame.copy()
        
        try:
            # 1. Object Detection
            # YOLO works natively with BGR numpy array from cv2
            detected_objs = detector.detect_objects(frame_to_process)
            
            # Convert BGR to RGB PIL Image for VLM
            frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 2. VLM Caption Generation (Single caption for faster live feed)
            # Use a slightly lower beam_size to speed up live processing
            caption = captioner.generate_caption(pil_image, num_beams=3)
            
            if not caption:
                latest_caption = "Error generating caption."
                latest_boxes = detected_objs if detected_objs else []
                time.sleep(0.1)
                continue
                
            if not detected_objs:
                # No objects detected to refine with
                latest_caption = caption
                latest_boxes = []
            else:
                # 3. Refinement / Optimization
                # Refine the single generated caption using YOLO detections
                refined_caption = optimizer.refine_caption(caption, detected_objs)
                latest_caption = refined_caption
                latest_boxes = detected_objs
                
        except Exception as e:
            print(f"Inference error: {e}")
            
        # Optional: brief sleep to unburden GPU if running hot
        time.sleep(0.1)

def main():
    global latest_frame, is_running
    
    print("Setting up Live Video Feed...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load heavy models in main thread before starting capture
        captioner = VLMCaptionGenerator(device=device)
        detector = YoloDetector(device=device)
        optimizer = CaptionOptimizer(alpha=0.6, beta=0.4)
    except Exception as e:
        print(f"Failed to intialize models: {e}")
        return

    # Start inference thread
    t = threading.Thread(target=inference_thread, args=(captioner, detector, optimizer))
    t.daemon = True
    t.start()

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        is_running = False
        return
        
    print("Webcam started. Press 'q' to quit.")
    
    # ── Fullscreen Window Setup ──
    window_name = "Assistive Vision - Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        # Update global frame for the inference thread (use original resolution)
        latest_frame = frame.copy()
        
        # ── Draw Overlays ──
        
        # Draw bounding boxes
        for obj in latest_boxes:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            conf = obj['confidence']
            cls = obj['class']
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f"{cls} {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1-10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
        # Draw caption
        # Put text on a dark semi-transparent rectangle for readability
        (text_w, text_h), baseline = cv2.getTextSize(latest_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (10, 10), (10 + text_w + 10, 10 + text_h + 10 + baseline), (0, 0, 0), -1)
        cv2.putText(frame, latest_caption, (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Show video (fullscreen window stretches frame to fit screen)
        cv2.imshow(window_name, frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Live feed stopped.")

if __name__ == "__main__":
    main()

import torch
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: ultralytics package not found. Please install it with: pip install ultralytics")
    YOLO = None

class YoloDetector:
    """
    A wrapper class for the YOLOv8 object detection model.
    By default, it uses the YOLOv8 nano model (yolov8n.pt).
    """
    def __init__(self, model_name="yolov8n.pt", device=None):
        if YOLO is None:
            raise ImportError("Cannot initialize YoloDetector because ultralytics is not installed.")
            
        self.model_name = model_name
        
        # Load the YOLO model. The ultralytics package will automatically download 
        # the weights if they are not present locally.
        print(f"Loading YOLO model: {model_name}...")
        self.model = YOLO(model_name)
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"YOLO detector successfully initialized on {self.device}.")

    def detect_objects(self, image_path, conf_threshold=0.25):
        """
        Runs object detection on the given image.
        
        Args:
            image_path (str or Path): Path to the image file.
            conf_threshold (float): Minimum confidence threshold for detection.
            
        Returns:
            list of dict: A list of detected objects, each with 'class', 'confidence', and 'bbox'.
        """
        image_path_str = str(image_path)
        
        # Run inference
        results = self.model(image_path_str, conf=conf_threshold, device=self.device, verbose=False)
        
        detected_objects = []
        
        # Parse the results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get the class index, name, confidence, and bounding box
                cls_idx = int(box.cls[0].item())
                cls_name = self.model.names[cls_idx]
                conf = float(box.conf[0].item())
                # x1, y1, x2, y2 coordinates
                bbox = box.xyxy[0].tolist() 
                
                detected_objects.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": bbox
                })
                
        return detected_objects

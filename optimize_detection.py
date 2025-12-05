import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time
import os

class CapsuleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.5):
        """
        Initialize the capsule detector with ONNX model
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Configure ONNX Runtime for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4  # Use 4 threads
        sess_options.intra_op_num_threads = 4  # Use 4 threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"Model input shape: {self.input_shape}")
        print(f"Using providers: {self.session.get_providers()}")
        print(f"CPU threads: {sess_options.inter_op_num_threads}")
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Optimized preprocessing with correct batch size
        """
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(self.input_width / orig_width, self.input_height / orig_height)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image (use INTER_LINEAR for speed)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Convert BGR to RGB and normalize in one step
        rgb_normalized = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        input_tensor = np.transpose(rgb_normalized, (2, 0, 1))
        
        # Create batch with the expected batch size (8 in your case)
        batch_size = self.input_shape[0]
        batched_tensor = np.tile(input_tensor, (batch_size, 1, 1, 1))
        
        return batched_tensor, (scale, scale), (pad_x, pad_y)
    
    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int], 
                             scale_factors: Tuple[float, float], padding: Tuple[int, int]) -> List[dict]:
        """
        Optimized postprocessing
        """
        # Get predictions from first (and only) batch item
        predictions = outputs[0][0].T  # Shape: (num_detections, num_classes + 4)
        
        # Vectorized filtering
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores > self.confidence_threshold
        
        if not np.any(valid_mask):
            return []
        
        # Filter predictions
        valid_predictions = predictions[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = np.argmax(valid_predictions[:, 4:], axis=1)
        
        # Process coordinates vectorized
        scale_x, scale_y = scale_factors
        pad_x, pad_y = padding
        
        # Extract and convert coordinates
        x_centers = (valid_predictions[:, 0] - pad_x) / scale_x
        y_centers = (valid_predictions[:, 1] - pad_y) / scale_y
        widths = valid_predictions[:, 2] / scale_x
        heights = valid_predictions[:, 3] / scale_y
        
        # Convert to corner coordinates
        x1s = (x_centers - widths / 2).astype(int)
        y1s = (y_centers - heights / 2).astype(int)
        x2s = (x_centers + widths / 2).astype(int)
        y2s = (y_centers + heights / 2).astype(int)
        
        # Clip to image bounds
        orig_height, orig_width = original_shape
        x1s = np.clip(x1s, 0, orig_width)
        y1s = np.clip(y1s, 0, orig_height)
        x2s = np.clip(x2s, 0, orig_width)
        y2s = np.clip(y2s, 0, orig_height)
        
        # Prepare boxes for NMS
        boxes = []
        for i in range(len(x1s)):
            boxes.append([x1s[i], y1s[i], x2s[i] - x1s[i], y2s[i] - y1s[i]])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, valid_scores.tolist(), 
                                  self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': (x1s[i], y1s[i], x2s[i], y2s[i]),
                    'confidence': float(valid_scores[i]),
                    'class_id': int(valid_class_ids[i])
                })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run detection on a single image
        """
        # Preprocess image
        input_tensor, scale_factors, padding = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess results
        detections = self.postprocess_detections(outputs, image.shape[:2], scale_factors, padding)
        
        return detections

def draw_detections(image: np.ndarray, detections: List[dict], class_names: List[str] = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    """
    if class_names is None:
        class_names = ['capsule']
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
            
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

def main():
    """
    Main function to run live detection with optimizations
    """
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_model.onnx')
    
    # Initialize detector
    detector = CapsuleDetector(model_path, confidence_threshold=0.5, nms_threshold=0.5)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Optimize camera settings for lower resolution = better FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Reduced from 640
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Reduced from 480
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
    
    print("Starting live detection. Press 'q' to quit.")
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    frame_skip_counter = 0
    detections = []  # Initialize detections
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process every 3rd frame for better performance
        frame_skip_counter += 1
        if frame_skip_counter % 3 == 0:
            detections = detector.detect(frame)
            frame_skip_counter = 0
        
        # Draw detections
        frame_with_detections = draw_detections(frame.copy(), detections, ['capsule'])
        
        # Calculate and display FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        # Display info on frame
        cv2.putText(frame_with_detections, f"FPS: {fps if 'fps' in locals() else 0:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame_with_detections, f"Capsules: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Capsule Detection', frame_with_detections)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
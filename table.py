import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time
import os
import math

class CapsuleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.5):
        """
        Initialize the capsule detector with ONNX model and FOV measurement
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # FOV measurement variables
        self.camera_height = None  # Will be set during calibration
        self.fov_width_cm = None
        self.fov_height_cm = None
        self.pixels_per_cm_x = None
        self.pixels_per_cm_y = None
        self.calibrated = False
        
        # Configure ONNX Runtime for CPU optimization
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 4
        sess_options.intra_op_num_threads = 4
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
    
    def calibrate_fov(self, frame_width: int, frame_height: int, camera_height_cm: float, known_object_width_cm: float = None):
        """
        Calibrate the field of view based on camera height
        
        Args:
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels  
            camera_height_cm: Height of camera above table in centimeters
            known_object_width_cm: Optional known object width for precise calibration
        """
        self.camera_height = camera_height_cm
        
        # Estimate FOV using typical webcam specifications
        # Most webcams have ~60-70 degree horizontal FOV
        horizontal_fov_degrees = 65  # Typical webcam FOV
        vertical_fov_degrees = horizontal_fov_degrees * (frame_height / frame_width)
        
        # Calculate table coverage using trigonometry
        # FOV width = 2 * height * tan(horizontal_fov / 2)
        horizontal_fov_radians = math.radians(horizontal_fov_degrees)
        vertical_fov_radians = math.radians(vertical_fov_degrees)
        
        self.fov_width_cm = 2 * camera_height_cm * math.tan(horizontal_fov_radians / 2)
        self.fov_height_cm = 2 * camera_height_cm * math.tan(vertical_fov_radians / 2)
        
        # Calculate pixels per centimeter
        self.pixels_per_cm_x = frame_width / self.fov_width_cm
        self.pixels_per_cm_y = frame_height / self.fov_height_cm
        
        self.calibrated = True
        
        print(f"\n📏 FOV Calibration Complete:")
        print(f"  Camera height: {camera_height_cm:.1f} cm")
        print(f"  Table coverage: {self.fov_width_cm:.1f} × {self.fov_height_cm:.1f} cm")
        print(f"  Resolution: {self.pixels_per_cm_x:.1f} × {self.pixels_per_cm_y:.1f} pixels/cm")
        print(f"  Total area: {(self.fov_width_cm * self.fov_height_cm):.0f} cm²\n")
    
    def pixel_to_cm(self, pixel_distance: float, axis: str = 'x') -> float:
        """Convert pixel distance to centimeters"""
        if not self.calibrated:
            return pixel_distance
        
        if axis == 'x':
            return pixel_distance / self.pixels_per_cm_x
        else:
            return pixel_distance / self.pixels_per_cm_y
    
    def get_capsule_size(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get capsule size in centimeters
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (width_cm, height_cm)
        """
        if not self.calibrated:
            return (0, 0)
        
        x1, y1, x2, y2 = bbox
        width_pixels = x2 - x1
        height_pixels = y2 - y1
        
        width_cm = self.pixel_to_cm(width_pixels, 'x')
        height_cm = self.pixel_to_cm(height_pixels, 'y')
        
        return width_cm, height_cm
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """Optimized preprocessing with correct batch size"""
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
        
        # Create batch with the expected batch size
        batch_size = self.input_shape[0]
        batched_tensor = np.tile(input_tensor, (batch_size, 1, 1, 1))
        
        return batched_tensor, (scale, scale), (pad_x, pad_y)
    
    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int], 
                             scale_factors: Tuple[float, float], padding: Tuple[int, int]) -> List[dict]:
        """Optimized postprocessing"""
        # Get predictions from first batch item
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
                bbox = (x1s[i], y1s[i], x2s[i], y2s[i])
                width_cm, height_cm = self.get_capsule_size(bbox)
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(valid_scores[i]),
                    'class_id': int(valid_class_ids[i]),
                    'size_cm': (width_cm, height_cm)
                })
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """Run detection on a single image"""
        # Preprocess image
        input_tensor, scale_factors, padding = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess results
        detections = self.postprocess_detections(outputs, image.shape[:2], scale_factors, padding)
        
        return detections

def draw_fov_grid(image: np.ndarray, detector: CapsuleDetector) -> np.ndarray:
    """Draw a grid showing the field of view measurements"""
    if not detector.calibrated:
        return image
    
    height, width = image.shape[:2]
    
    # Draw grid lines every 5 cm
    grid_spacing_cm = 5
    grid_spacing_x = int(grid_spacing_cm * detector.pixels_per_cm_x)
    grid_spacing_y = int(grid_spacing_cm * detector.pixels_per_cm_y)
    
    # Draw vertical lines
    for x in range(0, width, grid_spacing_x):
        cv2.line(image, (x, 0), (x, height), (128, 128, 128), 1)
        # Add distance labels
        if x > 0:
            distance_cm = detector.pixel_to_cm(x, 'x')
            cv2.putText(image, f"{distance_cm:.0f}cm", (x + 2, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Draw horizontal lines
    for y in range(0, height, grid_spacing_y):
        cv2.line(image, (0, y), (width, y), (128, 128, 128), 1)
        # Add distance labels
        if y > 0:
            distance_cm = detector.pixel_to_cm(y, 'y')
            cv2.putText(image, f"{distance_cm:.0f}cm", (2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return image

def draw_detections(image: np.ndarray, detections: List[dict], class_names: List[str] = None) -> np.ndarray:
    """Draw bounding boxes and labels with size measurements"""
    if class_names is None:
        class_names = ['capsule']
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        width_cm, height_cm = detection.get('size_cm', (0, 0))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create detailed label with size info
        if class_id < len(class_names):
            if width_cm > 0:
                label = f"{class_names[class_id]}: {confidence:.2f}"
                size_label = f"Size: {width_cm:.1f}×{height_cm:.1f}cm"
            else:
                label = f"{class_names[class_id]}: {confidence:.2f}"
                size_label = "Size: Not calibrated"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
            size_label = ""
        
        # Draw main label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 35), 
                     (x1 + max(label_size[0], 150), y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw size label if available
        if size_label:
            cv2.putText(image, size_label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def main():
    """Main function with FOV measurement capabilities"""
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
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get actual frame dimensions
    ret, test_frame = cap.read()
    if ret:
        frame_height, frame_width = test_frame.shape[:2]
        print(f"Camera resolution: {frame_width}×{frame_height}")
        
        # Calibrate FOV - ADJUST THIS VALUE TO YOUR CAMERA HEIGHT
        camera_height_cm = float(input("Enter camera height above table (cm): "))
        detector.calibrate_fov(frame_width, frame_height, camera_height_cm)
    
    print("\nStarting live detection with FOV measurement.")
    print("Controls:")
    print("  'g' - Toggle grid display")
    print("  'r' - Recalibrate camera height")
    print("  'q' - Quit")
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    frame_skip_counter = 0
    detections = []
    show_grid = True
    
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
        
        # Draw grid if enabled
        if show_grid:
            frame = draw_fov_grid(frame, detector)
        
        # Draw detections with size info
        frame_with_detections = draw_detections(frame.copy(), detections, ['capsule'])
        
        # Calculate and display FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
        
        # Display enhanced info
        info_y = 30
        cv2.putText(frame_with_detections, f"FPS: {fps if 'fps' in locals() else 0:.1f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if detector.calibrated:
            cv2.putText(frame_with_detections, f"FOV: {detector.fov_width_cm:.0f}×{detector.fov_height_cm:.0f}cm", 
                       (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame_with_detections, f"Height: {detector.camera_height:.0f}cm", 
                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(frame_with_detections, f"Capsules: {len(detections)}", 
                   (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Capsule Detection with FOV Measurement', frame_with_detections)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid display: {'ON' if show_grid else 'OFF'}")
        elif key == ord('r'):
            new_height = float(input("\nEnter new camera height (cm): "))
            detector.calibrate_fov(frame_width, frame_height, new_height)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
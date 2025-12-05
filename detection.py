import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time

class CapsuleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.5):
        """
        Initialize the capsule detector with ONNX model
        
        Args:
            model_path: Path to the ONNX model file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for removing duplicate detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Preprocess image for YOLO model with letterboxing
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Tuple of (preprocessed image, scale factors, padding)
        """
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(self.input_width / orig_width, self.input_height / orig_height)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        
        # Create batch with the expected batch size (8 in your case)
        batch_size = self.input_shape[0]
        batched_tensor = np.tile(input_tensor, (batch_size, 1, 1, 1))
        
        return batched_tensor, (scale, scale), (pad_x, pad_y)
    
    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int], 
                             scale_factors: Tuple[float, float], padding: Tuple[int, int]) -> List[dict]:
        """
        Process YOLOv8 model outputs to extract bounding boxes
        
        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (height, width)
            scale_factors: Scale factors used in preprocessing
            padding: Padding applied in preprocessing
            
        Returns:
            List of detection dictionaries
        """
        # YOLOv8 output format: (batch_size, num_classes + 4, num_detections)
        # Take only the first batch item since we duplicated the same image
        predictions = outputs[0][0]  # Remove batch dimension, take first item
        
        # Transpose to (num_detections, num_classes + 4)
        predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        for prediction in predictions:
            # Extract box coordinates (x_center, y_center, width, height)
            x_center, y_center, width, height = prediction[:4]
            
            # Extract class scores
            class_scores = prediction[4:]
            max_score = np.max(class_scores)
            
            if max_score > self.confidence_threshold:
                class_id = np.argmax(class_scores)
                
                # Convert from model coordinate space to original image space
                scale_x, scale_y = scale_factors
                pad_x, pad_y = padding
                
                # Adjust for padding and scaling
                x_center = (x_center - pad_x) / scale_x
                y_center = (y_center - pad_y) / scale_y
                width = width / scale_x
                height = height / scale_y
                
                # Convert to corner coordinates
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Clip to image bounds
                orig_height, orig_width = original_shape
                x1 = max(0, min(x1, orig_width))
                y1 = max(0, min(y1, orig_height))
                x2 = max(0, min(x2, orig_width))
                y2 = max(0, min(y2, orig_height))
                
                boxes.append([x1, y1, x2, y2])
                scores.append(max_score)
                class_ids.append(class_id)
        
        # Apply NMS
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)
            
            # Convert to format expected by cv2.dnn.NMSBoxes
            nms_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                nms_boxes.append([x1, y1, x2 - x1, y2 - y1])
            
            indices = cv2.dnn.NMSBoxes(nms_boxes, scores, self.confidence_threshold, self.nms_threshold)
            
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    detections.append({
                        'bbox': (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]),
                        'confidence': float(scores[i]),
                        'class_id': int(class_ids[i])
                    })
            
            return detections
        
        return []
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Run detection on a single image
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detections
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
    
    Args:
        image: Input image
        detections: List of detection dictionaries
        class_names: List of class names
        
    Returns:
        Image with drawn detections
    """
    if class_names is None:
        class_names = ['capsule']  # Default class name
    
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
    Main function to run live detection
    """
    # Initialize detector with your model - using full path to the model
    detector = CapsuleDetector('f:/capsule/pills-detection/best_model.onnx', confidence_threshold=0.5, nms_threshold=0.5)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting live detection. Press 'q' to quit.")
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw detections
        frame_with_detections = draw_detections(frame, detections, ['capsule'])
        
        # Calculate and display FPS
        fps_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps = fps_counter / elapsed_time
            fps_counter = 0
            start_time = time.time()
            
            # Display FPS on frame
            cv2.putText(frame_with_detections, f"FPS: {fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display detection count
        cv2.putText(frame_with_detections, f"Capsules: {len(detections)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
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
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import time
import os

class CapsuleDetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, nms_threshold: float = 0.5):
        """
        Initialize the capsule detector with ONNX model - GPU optimized
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Configure ONNX Runtime for GPU optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # Try GPU providers first, fallback to CPU
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kSameAsRequested',
            })
        ]
        
        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            print(f"✓ Successfully loaded model with providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"⚠ GPU not available, falling back to CPU: {e}")
            # Fallback to CPU only
            providers = ['CPUExecutionProvider']
            sess_options.inter_op_num_threads = 4
            sess_options.intra_op_num_threads = 4
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"Model input shape: {self.input_shape}")
        print(f"Active providers: {self.session.get_providers()}")
        
        # Check if GPU is being used
        self.using_gpu = 'CUDAExecutionProvider' in self.session.get_providers()
        if self.using_gpu:
            print("🚀 GPU acceleration enabled!")
        else:
            print("💻 Using CPU acceleration")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Ultra-optimized preprocessing - FIXED for batch size issue
        """
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(self.input_width / orig_width, self.input_height / orig_height)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Use GPU-optimized resize if available, otherwise fast CPU resize
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image with zeros
        padded = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (self.input_width - new_width) // 2
        pad_y = (self.input_height - new_height) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        # Optimized color conversion and normalization
        rgb_normalized = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # FIXED: Always use the required batch size for your model
        input_tensor = np.transpose(rgb_normalized, (2, 0, 1))
        batch_size = self.input_shape[0]  # This is 8 for your model
        batched_tensor = np.tile(input_tensor, (batch_size, 1, 1, 1))
        
        return batched_tensor, (scale, scale), (pad_x, pad_y)
    
    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int], 
                             scale_factors: Tuple[float, float], padding: Tuple[int, int]) -> List[dict]:
        """
        Ultra-fast vectorized postprocessing
        """
        # Handle both single batch (GPU) and multi-batch (CPU) outputs
        if self.using_gpu and len(outputs[0].shape) == 3:
            predictions = outputs[0][0].T  # GPU: single batch
        else:
            predictions = outputs[0][0].T  # CPU: take first of batch
        
        # Early exit if no predictions
        if predictions.shape[0] == 0:
            return []
        
        # Vectorized confidence filtering
        class_scores = predictions[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        valid_mask = max_scores > self.confidence_threshold
        
        if not np.any(valid_mask):
            return []
        
        # Filter valid predictions in one go
        valid_predictions = predictions[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = np.argmax(valid_predictions[:, 4:], axis=1)
        
        # Vectorized coordinate transformation
        scale_x, scale_y = scale_factors
        pad_x, pad_y = padding
        
        # Process all coordinates at once
        x_centers = (valid_predictions[:, 0] - pad_x) / scale_x
        y_centers = (valid_predictions[:, 1] - pad_y) / scale_y
        widths = valid_predictions[:, 2] / scale_x
        heights = valid_predictions[:, 3] / scale_y
        
        # Convert to corner coordinates (vectorized)
        half_widths = widths / 2
        half_heights = heights / 2
        x1s = np.clip(x_centers - half_widths, 0, original_shape[1]).astype(int)
        y1s = np.clip(y_centers - half_heights, 0, original_shape[0]).astype(int)
        x2s = np.clip(x_centers + half_widths, 0, original_shape[1]).astype(int)
        y2s = np.clip(y_centers + half_heights, 0, original_shape[0]).astype(int)
        
        # Fast NMS preparation
        boxes = np.column_stack([x1s, y1s, x2s - x1s, y2s - y1s])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), valid_scores.tolist(), 
                                  self.confidence_threshold, self.nms_threshold)
        
        # Build final detections
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
        GPU-accelerated detection
        """
        # Preprocess image
        input_tensor, scale_factors, padding = self.preprocess_image(image)
        
        # Run inference (automatically uses GPU if available)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess results
        detections = self.postprocess_detections(outputs, image.shape[:2], scale_factors, padding)
        
        return detections

def draw_detections(image: np.ndarray, detections: List[dict], class_names: List[str] = None) -> np.ndarray:
    """
    Optimized drawing function
    """
    if class_names is None:
        class_names = ['capsule']
    
    # Pre-compute common values
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Draw bounding box with thicker line for better visibility
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        
        # Create label
        if class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
        
        # Get label size
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_height - baseline - 5), 
                     (x1 + label_width, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - baseline - 2), 
                   font, font_scale, (0, 0, 0), thickness)
    
    return image

def main():
    """
    GPU-accelerated live detection with CPU optimization
    """
    # Get model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_model.onnx')
    
    # Initialize detector
    detector = CapsuleDetector(model_path, confidence_threshold=0.5, nms_threshold=0.45)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Optimize camera settings based on GPU availability
    if detector.using_gpu:
        # Higher resolution for GPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_skip = 1  # Process every frame
        print("🎥 Using high resolution (640x480) with GPU acceleration")
    else:
        # Much lower resolution for CPU to improve FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced further
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Reduced further
        frame_skip = 4  # Process every 4th frame for CPU
        print("🎥 Using lower resolution (320x240) with CPU - optimized for performance")
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Starting live detection. Press 'q' to quit.")
    
    # Performance tracking
    fps_counter = 0
    start_time = time.time()
    frame_skip_counter = 0
    detections = []
    
    # Warm up the model with correct size
    print("Warming up model...")
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(2):  # Reduced warm-up iterations
        _ = detector.detect(dummy_frame)
    print("Model warmed up!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Adaptive frame skipping - more aggressive for CPU
        frame_skip_counter += 1
        if frame_skip_counter % frame_skip == 0:
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
            print(f"📊 FPS: {fps:.1f} | Detections: {len(detections)} | Provider: {'GPU' if detector.using_gpu else 'CPU'}")
        
        # Display enhanced info on frame
        info_color = (0, 255, 255) if detector.using_gpu else (255, 0, 0)
        cv2.putText(frame_with_detections, f"FPS: {fps if 'fps' in locals() else 0:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
        cv2.putText(frame_with_detections, f"Capsules: {len(detections)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
        cv2.putText(frame_with_detections, f"{'GPU' if detector.using_gpu else 'CPU'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
        
        # Show frame
        cv2.imshow('Capsule Detection', frame_with_detections)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Detection stopped. Cleanup complete.")

if __name__ == "__main__":
    main()
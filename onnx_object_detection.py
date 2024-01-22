from typing import Tuple, Union
import numpy as np
import onnxruntime as ort
from typing import List
import cv2
from onnx_base import OnnxBase


class OnnxObjectDetection(OnnxBase):
    """ONNX Base class."""
    weight_path: str = None
    classnames: List[str] = None
    session: ort.InferenceSession = None

    input_size: Tuple[int, int] = None
    input_name: str = None
    output_name: str = None

    def __init__(self, weight_path: str, classnames: List[str] = None) -> None:
        """Initialize class.

        Args:
            weight_path: Location of the weight file
        """
        super().__init__(weight_path=weight_path)
        self.weight_path: str = weight_path
        self.classnames: List[str] = classnames if classnames else []

    def preprocess(self, frame):
        frame = cv2.resize(frame, (640, 640)) 
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = rgb_frame.astype(np.float32) / 255.0

        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=(0))
        
        return frame, image
    
    def predict(self, input_data: np.ndarray) -> Union[np.ndarray, None]:
        """OCR model predict code, independent of weight format.

        Args:
            input_data: Input data

        Returns:
            Resulting predictions
        """
        return self.predict_onnx(input_data)[0]
    
    def visualize_detections(self, org_frame, detections, output):
        for detection in detections:
            class_index, confidence, x1, y1, x2, y2 = detection
            class_name = self.classnames[int(class_index)]

            color = (0, 255, 0)  # Green color for the bounding boxes
            thickness = 2
            font_scale = 1

            # Draw bounding box
            cv2.rectangle(org_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # Display class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(org_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        output.write(org_frame)
        return org_frame

    def postprocess(self, results, confidence_threshold):
        # Extract outputs
        output = results[0]

        bounding_box = output[:4, :]
        class_probs = output[4:, :]

        class_indices = np.argmax(class_probs, axis=0)

        class_confidences = np.max(class_probs, axis=0)
        x, y, width, height = bounding_box
        detections = np.vstack((class_indices, class_confidences, x, y, width, height)).T

        detections = detections[detections[:, 1] > confidence_threshold]

        return detections

    
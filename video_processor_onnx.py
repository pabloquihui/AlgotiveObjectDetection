import cv2
import os
from object_detector import YoloDectector
from onnx_object_detection import OnnxObjectDetection
import time 
coco_classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear',
                   'hair drier', 'toothbrush']


class VideoProcessor:
    def __init__(self, class_subset):
        self.object_detector = OnnxObjectDetection(weight_path='models/yolov8n.onnx', classnames=class_subset)

    def process_videos(self, video_directory):
        # Get a list of video file paths in the specified directory
        video_paths = [os.path.join(video_directory, filename) for filename in os.listdir(video_directory) if filename.endswith(('.mp4', '.avi', '.mkv'))]

        # Process each video in the directory
        for video_path in video_paths:
            print(f"Processing video: {video_path}")
            self.process_video(video_path)

    def process_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        filename = video_path.split("/")[-1].split(".")[0]
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename + "_detected.mp4", fourcc, fps, (640, 640))
        # Process each frame
        total_time=0
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.time()
            # Preprocess the frame
            preprocessed_frame, preprocessed_image = self.object_detector.preprocess(frame)
            # Detect objects in the preprocessed frame
            detections = self.object_detector.predict(input_data=preprocessed_image)           
            # Save video with bounding boxes and classes
            post_detection = self.object_detector.postprocess(results=detections, confidence_threshold=0.5)
            # print(post_detection.shape)
            bb_img = self.object_detector.visualize_detections(org_frame= preprocessed_frame, 
                                                               detections=post_detection,
                                                               output=out, 
                                                               )
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
        # Release video capture object
        out.release()
        cap.release()
        # Calculate average inference time per frame and FPS
        avg_inference_time = total_time / frame_count
        avg_fps = 1 / avg_inference_time

        print(f"Video: {video_path}")
        print(f"Average Inference Time per Frame: {avg_inference_time:.4f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print()

# Example usage:
# Assuming you have a YOLODetector instance named 'yolo_detector' and class_subset ['car', 'motorcycle', 'person']
video_processor = VideoProcessor(class_subset=coco_classnames)
video_processor.process_videos('challenge_videos/test/')

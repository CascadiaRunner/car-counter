import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from ultralytics.nn.tasks import DetectionModel
from scipy.optimize import linear_sum_assignment

class CarDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        # Load model
        self.model = YOLO('yolov8n.pt')
        self.total_cars = 0  # Total historical count
        self.tracked_ids = set()  # Set to keep track of unique car IDs
        self.frame_count = 0
        # Initialize OpenCV tracker
        self.trackers = []  # List of (tracker, track_id, frames_without_detection, last_bbox)
        self.next_id = 0
        self.confidence_threshold = 0.45  # Lowered confidence threshold to catch more vehicles
        self.max_frames_without_detection = 45  # Increased to keep tracking longer
        self.iou_threshold = 0.5  # Lowered IoU threshold to be more lenient with matches
        self.detection_interval = 3  # Run YOLO detection more frequently
        self.last_detection_frame = 0
        self.edge_tolerance_frames = 5  # Number of frames to allow tracking near edge before dropping
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
        
    def is_box_near_edge(self, x, y, w, h, frame_width, frame_height, margin=10):  # Reduced margin
        """Check if a box is near the edge of the frame"""
        # Check if box is significantly outside the frame
        if (x < -w/4 or y < -h/4 or 
            x > frame_width + w/4 or y > frame_height + h/4):
            return True
            
        # Check if box is near the edge with reduced margin
        return (x < margin or 
                y < margin or 
                x + w > frame_width - margin or 
                y + h > frame_height - margin)
        
    def is_box_at_edge(self, x, y, w, h, frame_width, frame_height, margin=50):
        # Remove if any part of the box is within 'margin' pixels of the frame boundary
        return (
            x <= margin or y <= margin or x + w >= frame_width - margin or y + h >= frame_height - margin
        )
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        max_unmatched_frames = 10  # Allow trackers to persist this many frames without detection
        start_time = time.time()  # Track total processing time
        fps_values = []  # List to store FPS values
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_detections = []
            
            # Only run YOLO detection every few frames
            if self.frame_count - self.last_detection_frame >= self.detection_interval:
                # Detect all vehicle classes (2:car, 3:motorcycle, 5:bus, 7:truck, 8:boat, 9:traffic light, 11:stop sign)
                results = self.model(frame, classes=[2, 3, 5, 7, 8])
                self.last_detection_frame = self.frame_count
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        if conf > self.confidence_threshold:
                            x1 = max(0, min(int(x1), frame_width - 1))
                            y1 = max(0, min(int(y1), frame_height - 1))
                            x2 = max(0, min(int(x2), frame_width - 1))
                            y2 = max(0, min(int(y2), frame_height - 1))
                            
                            if x2 > x1 and y2 > y1:
                                current_detections.append((x1, y1, x2, y2))
            
            # Update trackers and get their predicted boxes
            tracker_bboxes = []
            active_trackers = []
            for tracker, track_id, frames_without_detection, last_bbox in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # Remove if box is at or beyond the edge (within 50px margin)
                    if self.is_box_at_edge(x, y, w, h, frame_width, frame_height, margin=50):
                        continue
                    # Remove if area is too small
                    if w * h < 10:
                        continue
                    if w <= 0 or h <= 0 or w > frame_width * 2 or h > frame_height * 2:
                        continue
                    if 15 < w < frame_width/1.5 and 15 < h < frame_height/1.5:
                        active_trackers.append((tracker, track_id, frames_without_detection, (x, y, x+w, y+h)))
                        tracker_bboxes.append((x, y, x+w, y+h))
                    else:
                        continue
                # If tracking failed, do not keep tracker
            self.trackers = active_trackers
            
            # Hungarian matching between detections and trackers
            unmatched_detections = set(range(len(current_detections)))
            unmatched_trackers = set(range(len(self.trackers)))
            matched_trackers = set()
            if current_detections and self.trackers:
                iou_matrix = np.zeros((len(self.trackers), len(current_detections)), dtype=np.float32)
                for t_idx, (_, _, _, t_bbox) in enumerate(self.trackers):
                    for d_idx, det in enumerate(current_detections):
                        if t_bbox is not None:
                            iou_matrix[t_idx, d_idx] = self.calculate_iou(t_bbox, det)
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU
                for t_idx, d_idx in zip(row_ind, col_ind):
                    if iou_matrix[t_idx, d_idx] > self.iou_threshold:
                        unmatched_detections.discard(d_idx)
                        unmatched_trackers.discard(t_idx)
                        # Detection-corrected tracking: update last_bbox to detection box and re-init tracker
                        tracker, track_id, _, _ = self.trackers[t_idx]
                        det_box = current_detections[d_idx]
                        x1, y1, x2, y2 = det_box
                        w, h = x2 - x1, y2 - y1
                        tracker.init(frame, (x1, y1, w, h))  # Re-initialize tracker with detection box
                        self.trackers[t_idx] = (tracker, track_id, 0, det_box)
                        matched_trackers.add(t_idx)
            # For unmatched trackers, increment frames_without_detection
            for t_idx in unmatched_trackers:
                tracker, track_id, frames_without_detection, last_bbox = self.trackers[t_idx]
                self.trackers[t_idx] = (tracker, track_id, frames_without_detection + 1, last_bbox)
            # Remove trackers that have been unmatched for too long
            self.trackers = [t for t in self.trackers if t[2] <= max_unmatched_frames]
            
            # Create new trackers for unmatched detections
            for d_idx in unmatched_detections:
                x1, y1, x2, y2 = current_detections[d_idx]
                w, h = x2 - x1, y2 - y1
                # Skip if detection is within 50px of any edge
                if (
                    x1 <= 50 or y1 <= 50 or x2 >= frame_width - 50 or y2 >= frame_height - 50
                ):
                    continue
                if w > 0 and h > 0:
                    # Check IoU with all existing trackers' last_bbox
                    overlaps = False
                    for _, _, _, last_bbox in self.trackers:
                        if last_bbox is not None:
                            iou = self.calculate_iou((x1, y1, x2, y2), last_bbox)
                            if iou > 0.1:
                                overlaps = True
                                break
                    if overlaps:
                        continue  # Skip creating a new tracker if overlap is too high
                    try:
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        self.trackers.append((tracker, self.next_id, 0, (x1, y1, x2, y2)))
                        self.tracked_ids.add(self.next_id)
                        self.total_cars += 1
                        self.next_id += 1
                    except cv2.error:
                        continue
            
            # Draw boxes for active trackers
            for tracker, track_id, _, last_bbox in self.trackers:
                if last_bbox is not None:
                    x1, y1, x2, y2 = last_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {track_id}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Calculate FPS
            if not hasattr(self, 'prev_time'):
                self.prev_time = time.time()
                self.fps = 0
            else:
                current_time = time.time()
                self.fps = 1 / (current_time - self.prev_time)
                self.prev_time = current_time
                fps_values.append(self.fps)  # Store FPS value
            
            # Display the frame with detections
            cv2.putText(frame, f'Total Unique Vehicles: {self.total_cars}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f'Currently Tracking: {len(self.trackers)}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Car Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate average FPS
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        return self.total_cars, avg_fps

def main():
    video_path = "videos/highway_light_tiny.mp4"
    
    detector = CarDetector(video_path)
    detector.process_video()

if __name__ == "__main__":
    main() 
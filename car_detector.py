import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
from ultralytics.nn.tasks import DetectionModel

class CarDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        # Load model
        self.model = YOLO('yolov8n.pt')
        self.total_cars = 0  # Total historical count
        self.tracked_ids = set()  # Set to keep track of unique car IDs
        self.frame_count = 0
        # Initialize OpenCV tracker
        self.trackers = []
        self.next_id = 0
        self.confidence_threshold = 0.75  # Increased confidence threshold
        self.max_frames_without_detection = 30  # Maximum frames to keep tracking without detection
        self.iou_threshold = 0.5  # Increased IoU threshold to be more strict
        
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
        
    def is_box_near_edge(self, x, y, w, h, frame_width, frame_height, margin=20):
        """Check if a box is near the edge of the frame"""
        return (x < margin or 
                y < margin or 
                x + w > frame_width - margin or 
                y + h > frame_height - margin)
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        # Get video dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Dictionary to track how long each tracker has been without a detection
        tracker_frames_without_detection = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLOv8 inference on the frame
            results = self.model(frame, classes=[2, 3, 5, 7])  # Only detect cars, trucks, buses, and motorcycles
            
            # Get current detections
            current_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    if conf > self.confidence_threshold:  # Only consider high confidence detections
                        # Ensure coordinates are within frame boundaries
                        x1 = max(0, min(int(x1), frame_width - 1))
                        y1 = max(0, min(int(y1), frame_height - 1))
                        x2 = max(0, min(int(x2), frame_width - 1))
                        y2 = max(0, min(int(y2), frame_height - 1))
                        
                        # Only add if the box has valid dimensions
                        if x2 > x1 and y2 > y1:
                            current_detections.append((x1, y1, x2, y2))
            
            # Update existing trackers
            active_trackers = []
            for tracker, track_id in self.trackers:
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    # Ensure coordinates are within frame boundaries
                    x = max(0, min(x, frame_width - 1))
                    y = max(0, min(y, frame_height - 1))
                    w = max(1, min(w, frame_width - x))
                    h = max(1, min(h, frame_height - y))
                    
                    # Check if the box is near the edge of the frame
                    if not self.is_box_near_edge(x, y, w, h, frame_width, frame_height):
                        # Draw bounding box and ID
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID: {track_id}', (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        active_trackers.append((tracker, track_id))
                        tracker_frames_without_detection[track_id] = 0
                    else:
                        # If near edge, increment the counter
                        tracker_frames_without_detection[track_id] = tracker_frames_without_detection.get(track_id, 0) + 1
                else:
                    # If tracking failed, increment the counter
                    tracker_frames_without_detection[track_id] = tracker_frames_without_detection.get(track_id, 0) + 1
            
            # Remove trackers that have been without detection for too long
            self.trackers = [(t, tid) for t, tid in active_trackers 
                           if tracker_frames_without_detection.get(tid, 0) < self.max_frames_without_detection]
            
            # Initialize new trackers for unmatched detections
            for det in current_detections:
                x1, y1, x2, y2 = det
                w, h = x2 - x1, y2 - y1
                
                # Check if this detection overlaps with existing trackers
                overlap = False
                for tracker, _ in self.trackers:
                    success, bbox = tracker.update(frame)
                    if success:
                        tx, ty, tw, th = [int(v) for v in bbox]
                        # Calculate IoU
                        iou = self.calculate_iou((x1, y1, x2, y2), (tx, ty, tx + tw, ty + th))
                        if iou > self.iou_threshold:  # More strict IoU threshold
                            overlap = True
                            break
                
                if not overlap and w > 0 and h > 0:
                    try:
                        # Create new tracker
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        self.trackers.append((tracker, self.next_id))
                        self.tracked_ids.add(self.next_id)
                        tracker_frames_without_detection[self.next_id] = 0
                        self.next_id += 1
                    except cv2.error:
                        # Skip this detection if tracker initialization fails
                        continue
            
            # Display the frame with detections
            cv2.putText(frame, f'Total Unique Cars: {len(self.tracked_ids)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Currently Tracking: {len(self.trackers)}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Car Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            self.frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        return len(self.tracked_ids)

def main():
    # Replace with your video file path
    video_path = "./27260-362770008_tiny.mp4"
    
    detector = CarDetector(video_path)
    total_cars = detector.process_video()
    print(f"Total unique cars detected: {total_cars}")

if __name__ == "__main__":
    main() 
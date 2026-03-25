"""
Single Camera Detector - Handles both vehicles and pedestrians
Built from scratch with Indian traffic constraints
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Types of objects we detect"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"


@dataclass
class Detection:
    """Data class for a single detection"""
    id: str
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    center: Tuple[int, int]
    area: int
    in_lane: bool = False
    is_stationary: bool = False
    stationary_time: float = 0
    first_seen: float = 0
    last_seen: float = 0
    type: DetectionType = DetectionType.VEHICLE


class UnifiedDetector:
    """
    Unified detector for vehicles and pedestrians using YOLOv8
    Optimized for Indian traffic conditions
    """
    
    # COCO class mappings
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    PEDESTRIAN_CLASSES = {
        0: 'pedestrian'
    }
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 lane_region: Optional[List[Tuple[int, int]]] = None,
                 crossing_region: Optional[List[Tuple[int, int]]] = None,
                 conf_threshold: float = 0.4):
        """
        Initialize detector with configurable regions
        
        Args:
            model_path: Path to YOLO model
            lane_region: Polygon for free-left lane
            crossing_region: Polygon for pedestrian crossing
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.lane_region = lane_region
        self.crossing_region = crossing_region
        self.conf_threshold = conf_threshold
        
        # Tracking state
        self.tracker = {}  # id -> Detection
        self.frame_count = 0
        self.last_detections = []
        
        # Performance metrics
        self.fps = 0
        self.last_frame_time = time.time()
        
        logger.info(f"Detector initialized | Lane: {lane_region is not None}, Crossing: {crossing_region is not None}")
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Detect vehicles and pedestrians in frame
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (detections list, annotated frame)
        """
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1 / (current_time - self.last_frame_time)) if self.fps else 0
        self.last_frame_time = current_time
        
        # Run YOLO detection
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Classify detection type
                if class_id in self.VEHICLE_CLASSES:
                    detection_type = DetectionType.VEHICLE
                    class_name = self.VEHICLE_CLASSES[class_id]
                elif class_id in self.PEDESTRIAN_CLASSES:
                    detection_type = DetectionType.PEDESTRIAN
                else:
                    continue  # Skip other objects
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                # Check if in lane or crossing
                in_lane = self._in_region(center, self.lane_region) if self.lane_region else False
                at_crossing = self._in_region(center, self.crossing_region) if self.crossing_region else False
                
                detection = Detection(
                    id=f"d{self.frame_count}_{len(detections)}",
                    class_name=class_name,
                    class_id=class_id,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    center=center,
                    area=area,
                    in_lane=in_lane,
                    type=detection_type,
                    first_seen=current_time,
                    last_seen=current_time
                )
                detections.append(detection)
        
        # Update tracking for stationary detection
        self._update_tracking(detections, current_time)
        
        # Annotate frame
        annotated = self._annotate(frame, detections)
        
        # Add FPS to frame
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return detections, annotated
    
    def _in_region(self, point: Tuple[int, int], region: List[Tuple[int, int]]) -> bool:
        """Check if point is inside a polygon region"""
        if not region:
            return False
        return cv2.pointPolygonTest(np.array(region, dtype=np.int32), point, False) >= 0
    
    def _update_tracking(self, detections: List[Detection], current_time: float) -> None:
        """Update tracking for stationary vehicle detection"""
        # Simple IoU-based tracking
        for det in detections:
            best_match = None
            best_iou = 0
            
            for track_id, track_det in self.tracker.items():
                iou = self._calculate_iou(det.bbox, track_det.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = track_id
            
            if best_match:
                # Update existing track
                tracked = self.tracker[best_match]
                tracked.last_seen = current_time
                tracked.bbox = det.bbox
                tracked.center = det.center
                tracked.in_lane = det.in_lane
                
                # Check if stationary (minimal movement)
                movement = np.sqrt((tracked.center[0] - det.center[0])**2 +
                                  (tracked.center[1] - det.center[1])**2)
                if movement < 10:
                    tracked.is_stationary = True
                    tracked.stationary_time = current_time - tracked.first_seen
                else:
                    tracked.is_stationary = False
                
                det.id = best_match
                det.first_seen = tracked.first_seen
                det.stationary_time = tracked.stationary_time
            else:
                # New track
                self.tracker[det.id] = det
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Annotate frame with detection results"""
        annotated = frame.copy()
        
        # Draw lane region if defined
        if self.lane_region:
            cv2.polylines(annotated, [np.array(self.lane_region, dtype=np.int32)],
                         True, (0, 255, 255), 2)
            cv2.putText(annotated, "FREE LEFT LANE", (self.lane_region[0][0], self.lane_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw crossing region if defined
        if self.crossing_region:
            cv2.polylines(annotated, [np.array(self.crossing_region, dtype=np.int32)],
                         True, (255, 255, 0), 2)
            cv2.putText(annotated, "PEDESTRIAN CROSSING", (self.crossing_region[0][0], self.crossing_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on type and location
            if det.type == DetectionType.PEDESTRIAN:
                color = (255, 0, 0) if det.in_lane else (255, 255, 0)
            else:  # VEHICLE
                color = (0, 0, 255) if det.in_lane else (0, 255, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, det.center, 5, color, -1)
            
            # Label
            label = f"{det.class_name} ({det.confidence:.2f})"
            if det.in_lane and det.type == DetectionType.VEHICLE:
                if det.stationary_time > 3:
                    label += " ⚠️ BLOCKING"
                else:
                    label += " ⚠️ IN LANE"
            elif det.at_crossing and det.type == DetectionType.PEDESTRIAN:
                label += " 🚶 AT CROSSING"
            
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return annotated
    
    def get_blocking_vehicles(self, min_duration: float = 3.0) -> List[Detection]:
        """Get vehicles blocking free-left lane for more than min_duration"""
        blocking = []
        for det in self.tracker.values():
            if det.type == DetectionType.VEHICLE and det.in_lane:
                if det.stationary_time >= min_duration:
                    blocking.append(det)
        return blocking
    
    def get_pedestrians_at_crossing(self) -> List[Detection]:
        """Get pedestrians at crossing"""
        pedestrians = []
        for det in self.tracker.values():
            if det.type == DetectionType.PEDESTRIAN and det.at_crossing:
                pedestrians.append(det)
        return pedestrians
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_frames': self.frame_count,
            'fps': self.fps,
            'tracked_vehicles': len([d for d in self.tracker.values() if d.type == DetectionType.VEHICLE]),
            'tracked_pedestrians': len([d for d in self.tracker.values() if d.type == DetectionType.PEDESTRIAN]),
            'blocking_vehicles': len(self.get_blocking_vehicles()),
            'pedestrians_waiting': len(self.get_pedestrians_at_crossing())
        }
"""
Unified Detector - Fixed Version with Proper Stats Dictionary
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
    """Detection types"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    TRUCK = "truck"
    BUS = "bus"


@dataclass
class Detection:
    """Data class for a single detection"""
    id: str
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]
    center: Tuple[int, int]
    area: int
    in_lane: bool = False
    at_crossing: bool = False
    is_stationary: bool = False
    stationary_time: float = 0
    first_seen: float = 0
    last_seen: float = 0
    type: DetectionType = DetectionType.VEHICLE
    aspect_ratio: float = 0.0


class UnifiedDetector:
    """
    Simplified detector - Basic vehicle and pedestrian detection
    """
    
    VEHICLE_CLASSES = {
        2: {'name': 'car', 'type': DetectionType.VEHICLE},
        3: {'name': 'motorcycle', 'type': DetectionType.MOTORCYCLE},
        5: {'name': 'bus', 'type': DetectionType.BUS},
        7: {'name': 'truck', 'type': DetectionType.TRUCK},
        1: {'name': 'bicycle', 'type': DetectionType.BICYCLE}
    }
    
    PEDESTRIAN_CLASSES = {
        0: {'name': 'pedestrian', 'type': DetectionType.PEDESTRIAN}
    }
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 lane_region: Optional[List[Tuple[int, int]]] = None,
                 auto_detect_crossing: bool = True,
                 conf_threshold: float = 0.4,
                 enhance_visibility: bool = True,
                 debug_mode: bool = False):
        """
        Initialize detector
        """
        self.model = YOLO(model_path)
        self.lane_region = lane_region
        self.conf_threshold = conf_threshold
        self.enhance_visibility = enhance_visibility
        self.debug_mode = debug_mode
        
        # Manual crossing region
        self.manual_crossing_region = None
        self.crossing_polygon = None
        
        # Auto zebra crossing detection
        self.auto_detect_crossing = auto_detect_crossing
        self.zebra_detector = None
        self.crossing_tracker = None
        
        if auto_detect_crossing:
            from .zebra_detector import EnhancedZebraDetector, ZebraCrossingTracker
            self.zebra_detector = EnhancedZebraDetector(
                min_stripes=3,
                confidence_threshold=0.5,
                debug_mode=debug_mode
            )
            self.crossing_tracker = ZebraCrossingTracker(stability_frames=3)
        
        # Pedestrian tracker
        self.pedestrian_tracker = None
        self.pedestrian_tracker_enabled = False
        
        # Tracking
        self.tracker = {}
        self.frame_count = 0
        self.fps = 0
        self.last_frame_time = time.time()
        
        # Enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info(f"Detector initialized")
    
    def enable_pedestrian_tracking(self, 
                                   crossing_region: List[Tuple[int, int]],
                                   waiting_region: Optional[List[Tuple[int, int]]] = None):
        """Enable pedestrian tracking"""
        from .pedestrian_tracker import PedestrianTracker
        self.pedestrian_tracker = PedestrianTracker(
            crossing_region=crossing_region,
            waiting_region=waiting_region
        )
        self.pedestrian_tracker_enabled = True
        logger.info(f"Pedestrian tracking enabled")
    
    def set_manual_crossing(self, region: List[Tuple[int, int]]):
        """Set manual crossing region"""
        self.manual_crossing_region = region
        self.crossing_polygon = region
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply image enhancement"""
        if not self.enhance_visibility:
            return frame
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        gamma = 1.3
        look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, look_up_table)
        
        return enhanced
    
    def _classify_detection(self, class_id: int, bbox: List[int]) -> Tuple[DetectionType, str, int]:
        """Classify detection"""
        if class_id == 0:
            return DetectionType.PEDESTRIAN, "pedestrian", class_id
        elif class_id in self.VEHICLE_CLASSES:
            info = self.VEHICLE_CLASSES[class_id]
            return info['type'], info['name'], class_id
        elif class_id == 1:
            return DetectionType.BICYCLE, "bicycle", class_id
        else:
            return DetectionType.VEHICLE, "unknown", class_id
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Detect vehicles and pedestrians"""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        self.fps = 0.9 * self.fps + 0.1 * (1 / (current_time - self.last_frame_time)) if self.fps else 0
        self.last_frame_time = current_time
        
        # Apply enhancement
        processed_frame = self._enhance_frame(frame)
        
        # Determine crossing region
        if self.manual_crossing_region is not None:
            self.crossing_polygon = self.manual_crossing_region
        elif self.auto_detect_crossing and self.zebra_detector:
            detected = self.zebra_detector.detect(processed_frame)
            if self.crossing_tracker:
                tracked = self.crossing_tracker.update(detected)
                if tracked:
                    self.crossing_polygon = tracked.polygon
        
        # YOLO detection
        results = self.model(processed_frame, verbose=False, conf=self.conf_threshold)
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detection_type, class_name, mapped_id = self._classify_detection(class_id, [x1, y1, x2, y2])
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                in_lane = self._in_region(center, self.lane_region) if self.lane_region else False
                at_crossing = self._in_region(center, self.crossing_polygon) if self.crossing_polygon else False
                
                detection = Detection(
                    id=f"d{self.frame_count}_{len(detections)}",
                    class_name=class_name,
                    class_id=mapped_id,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                    center=center,
                    area=area,
                    in_lane=in_lane,
                    at_crossing=at_crossing,
                    type=detection_type,
                    first_seen=current_time,
                    last_seen=current_time
                )
                detections.append(detection)
        
        # Update tracking
        self._update_tracking(detections, current_time)
        
        # Update pedestrian tracker
        if self.pedestrian_tracker_enabled and self.pedestrian_tracker:
            tracked_peds, ped_stats = self.pedestrian_tracker.update(detections)
        
        # Annotate frame
        annotated = self._annotate(processed_frame, detections)
        
        # Add overlays
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.auto_detect_crossing:
            if self.crossing_polygon:
                cv2.putText(annotated, "✓ ZEBRA CROSSING DETECTED", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(annotated, "SCANNING FOR CROSSING...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        return detections, annotated
    
    def _in_region(self, point: Tuple[int, int], region: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon"""
        if not region:
            return False
        return cv2.pointPolygonTest(np.array(region, dtype=np.int32), point, False) >= 0
    
    def _update_tracking(self, detections: List[Detection], current_time: float) -> None:
        """Update tracking"""
        for det in detections:
            if det.type == DetectionType.PEDESTRIAN:
                continue
                
            best_match = None
            best_iou = 0
            
            for track_id, track_det in self.tracker.items():
                iou = self._calculate_iou(det.bbox, track_det.bbox)
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_match = track_id
            
            if best_match:
                tracked = self.tracker[best_match]
                tracked.last_seen = current_time
                tracked.bbox = det.bbox
                tracked.center = det.center
                tracked.in_lane = det.in_lane
                tracked.at_crossing = det.at_crossing
                tracked.class_name = det.class_name
                tracked.type = det.type
                
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
                self.tracker[det.id] = det
        
        # Remove stale tracks
        stale_ids = []
        for track_id, track_det in self.tracker.items():
            if current_time - track_det.last_seen > 3.0:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracker[track_id]
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union"""
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
        """Annotate frame"""
        annotated = frame.copy()
        
        # Draw lane region
        if self.lane_region:
            pts = np.array(self.lane_region, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
            cv2.putText(annotated, "FREE LEFT LANE", (self.lane_region[0][0], self.lane_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw crossing region
        if self.crossing_polygon:
            color = (0, 255, 0) if self.manual_crossing_region else (0, 255, 255)
            pts = np.array(self.crossing_polygon, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, color, 3)
            label = "MANUAL CROSSING" if self.manual_crossing_region else "AUTO CROSSING"
            cv2.putText(annotated, label, (self.crossing_polygon[0][0], self.crossing_polygon[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            if det.type == DetectionType.PEDESTRIAN:
                color = (0, 0, 255) if det.at_crossing else (255, 0, 0)
            elif det.type == DetectionType.MOTORCYCLE:
                color = (0, 165, 255) if det.in_lane else (255, 255, 0)
            elif det.type in [DetectionType.BUS, DetectionType.TRUCK]:
                color = (255, 0, 0) if det.in_lane else (0, 255, 0)
            else:
                color = (0, 0, 255) if det.in_lane and det.is_stationary else (0, 255, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, det.center, 5, color, -1)
            
            label = f"{det.class_name} ({det.confidence:.2f})"
            if det.type == DetectionType.MOTORCYCLE:
                label = f"🏍️ {label}"
            elif det.type == DetectionType.PEDESTRIAN:
                label = f"🚶 {label}"
            
            if det.type != DetectionType.PEDESTRIAN and det.in_lane and det.is_stationary:
                label += f" ⚠️ {det.stationary_time:.0f}s"
            elif det.type == DetectionType.PEDESTRIAN and det.at_crossing:
                label += " 🚶 AT CROSSING"
            
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
    
    def get_blocking_vehicles(self, min_duration: float = 3.0) -> List[Detection]:
        """Get blocking vehicles"""
        blocking = []
        for det in self.tracker.values():
            if det.type != DetectionType.PEDESTRIAN and det.in_lane and det.is_stationary:
                if det.stationary_time >= min_duration:
                    blocking.append(det)
        return blocking
    
    def get_pedestrians_at_crossing(self) -> List[Detection]:
        """Get pedestrians at crossing"""
        return [d for d in self.tracker.values() if d.type == DetectionType.PEDESTRIAN and d.at_crossing]
    
    def get_crossing_region(self) -> Optional[List[Tuple[int, int]]]:
        """Get crossing region"""
        return self.crossing_polygon
    
    def get_stats(self) -> Dict:
        """Get statistics - ALL KEYS DEFINED"""
        # Count tracked vehicles by type
        tracked_vehicles = 0
        tracked_motorcycles = 0
        tracked_pedestrians = 0
        
        for det in self.tracker.values():
            if det.type == DetectionType.MOTORCYCLE:
                tracked_motorcycles += 1
                tracked_vehicles += 1
            elif det.type == DetectionType.VEHICLE:
                tracked_vehicles += 1
            elif det.type == DetectionType.BUS:
                tracked_vehicles += 1
            elif det.type == DetectionType.TRUCK:
                tracked_vehicles += 1
            elif det.type == DetectionType.PEDESTRIAN:
                tracked_pedestrians += 1
        
        blocking_vehicles = len(self.get_blocking_vehicles())
        pedestrians_waiting = len(self.get_pedestrians_at_crossing())
        
        return {
            'total_frames': self.frame_count,
            'fps': round(self.fps, 1),
            'tracked_vehicles': tracked_vehicles,
            'tracked_motorcycles': tracked_motorcycles,
            'tracked_pedestrians': tracked_pedestrians,
            'blocking_vehicles': blocking_vehicles,
            'pedestrians_waiting': pedestrians_waiting,
            'crossing_detected': self.crossing_polygon is not None
        }
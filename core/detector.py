"""
Unified Detector - COMPLETE with Weighted Occupancy, Spillover Detection, and Enhanced Tracking
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class DetectionType(Enum):
    """Detection types with weights for occupancy calculation"""
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    TRUCK = "truck"
    BUS = "bus"
    AUTO = "auto"


# Vehicle weight factors for accurate lane occupancy
VEHICLE_WEIGHTS = {
    DetectionType.MOTORCYCLE: 0.5,   # Motorcycle counts as 0.5 vehicle
    DetectionType.BICYCLE: 0.3,      # Bicycle counts as 0.3 vehicle
    DetectionType.VEHICLE: 1.0,      # Car counts as 1 vehicle
    DetectionType.BUS: 2.5,          # Bus counts as 2.5 vehicles
    DetectionType.TRUCK: 2.0,        # Truck counts as 2 vehicles
    DetectionType.AUTO: 0.8          # Auto-rickshaw counts as 0.8 vehicle
}


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
    weight: float = 1.0  # Added for weighted occupancy


class UnifiedDetector:
    """
    Unified detector with weighted occupancy, spillover detection, and enhanced tracking
    """
    
    VEHICLE_CLASSES = {
        2: {'name': 'car', 'type': DetectionType.VEHICLE, 'weight': 1.0},
        3: {'name': 'motorcycle', 'type': DetectionType.MOTORCYCLE, 'weight': 0.5},
        5: {'name': 'bus', 'type': DetectionType.BUS, 'weight': 2.5},
        7: {'name': 'truck', 'type': DetectionType.TRUCK, 'weight': 2.0},
        1: {'name': 'bicycle', 'type': DetectionType.BICYCLE, 'weight': 0.3}
    }
    
    PEDESTRIAN_CLASSES = {
        0: {'name': 'pedestrian', 'type': DetectionType.PEDESTRIAN, 'weight': 0}
    }
    
    MIN_VEHICLE_AREA = 800
    MAX_PEDESTRIAN_AREA = 1500
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt',
                 lane_region: Optional[List[Tuple[int, int]]] = None,
                 auto_detect_crossing: bool = True,
                 conf_threshold: float = 0.4,
                 enhance_visibility: bool = True,
                 debug_mode: bool = False):
        """
        Initialize detector with all features
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
        
        # Sliding window for detection history
        self.detection_history = deque(maxlen=30)
        self.occupancy_history = deque(maxlen=30)
        
        # Enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        logger.info(f"Detector initialized | Auto zebra: {auto_detect_crossing}")
    
    def enable_pedestrian_tracking(self, 
                                   crossing_region: List[Tuple[int, int]],
                                   waiting_region: Optional[List[Tuple[int, int]]] = None,
                                   max_disappear_frames: int = 10):
        """Enable dedicated pedestrian tracking"""
        from .pedestrian_tracker import PedestrianTracker
        self.pedestrian_tracker = PedestrianTracker(
            crossing_region=crossing_region,
            waiting_region=waiting_region,
            max_disappear_frames=max_disappear_frames
        )
        self.pedestrian_tracker_enabled = True
        logger.info(f"Pedestrian tracking enabled")
    
    def set_manual_crossing(self, region: List[Tuple[int, int]]):
        """Set manual crossing region"""
        self.manual_crossing_region = region
        self.crossing_polygon = region
        logger.info(f"Manual crossing region set: {region}")
    
    def get_weighted_occupancy(self, lane_vehicles: List[Detection]) -> float:
        """
        Calculate weighted lane occupancy considering vehicle types
        
        Returns:
            Weighted occupancy score (higher = more blocking impact)
        """
        if not lane_vehicles:
            return 0.0
        
        total_weight = 0.0
        for vehicle in lane_vehicles:
            weight = vehicle.weight
            total_weight += weight
            
            # Additional weight for stationary vehicles
            if vehicle.is_stationary:
                total_weight += weight * 0.5
            
            # Occlusion penalty for large vehicles (they block view)
            if vehicle.type in [DetectionType.BUS, DetectionType.TRUCK]:
                total_weight += 0.3
        
        # Update history
        self.occupancy_history.append(total_weight)
        
        return total_weight
    
    def get_effective_blocking_count(self, lane_vehicles: List[Detection]) -> int:
        """Get effective blocking count with weighted vehicles"""
        weighted = self.get_weighted_occupancy(lane_vehicles)
        return max(1, int(weighted)) if weighted > 0 else 0
    
    def detect_spillover(self, lane_vehicles: List[Detection]) -> bool:
        """
        Detect if queue is spilling over into straight lane
        
        Returns:
            True if spillover detected
        """
        if not lane_vehicles or not self.lane_region:
            return False
        
        # Get lane bottom boundary
        lane_bottom = self.lane_region[3][1] if len(self.lane_region) > 3 else 0
        
        spillover_count = 0
        for vehicle in lane_vehicles:
            # Check if vehicle is near or beyond lane boundary
            if vehicle.center[1] > lane_bottom - 30:
                spillover_count += 1
        
        # If multiple vehicles are at lane exit, spillover likely
        return spillover_count >= 2
    
    def get_queue_length(self, lane_vehicles: List[Detection]) -> int:
        """
        Estimate queue length based on vehicle positions
        
        Returns:
            Estimated queue length in pixels
        """
        if not lane_vehicles:
            return 0
        
        # Sort vehicles by position
        sorted_vehicles = sorted(lane_vehicles, key=lambda v: v.center[1])
        
        if len(sorted_vehicles) < 2:
            return 20  # Single vehicle queue
        
        # Estimate length from first to last vehicle
        first_y = sorted_vehicles[0].center[1]
        last_y = sorted_vehicles[-1].center[1]
        
        return last_y - first_y
    
    def get_occlusion_factor(self, lane_vehicles: List[Detection]) -> float:
        """
        Calculate occlusion factor (how much large vehicles block view)
        
        Returns:
            Occlusion factor between 0 and 1
        """
        if not lane_vehicles:
            return 0.0
        
        # Check for large vehicles that could block view
        large_vehicles = [v for v in lane_vehicles if v.type in [DetectionType.BUS, DetectionType.TRUCK]]
        
        if not large_vehicles:
            return 0.0
        
        # Calculate occlusion based on large vehicle positions
        occlusion = 0.0
        for lv in large_vehicles:
            # Larger vehicle = more occlusion
            size_factor = min(1.0, lv.area / 5000)
            # Vehicles closer to camera cause more occlusion
            position_factor = 1.0 - (lv.center[1] / 720) if lv.center[1] < 720 else 0
            
            occlusion = max(occlusion, size_factor * position_factor)
        
        return min(1.0, occlusion)
    
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
    
    def _classify_detection(self, class_id: int, bbox: List[int], confidence: float) -> Tuple[DetectionType, str, int, float]:
        """Enhanced classification with weights"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # PEDESTRIAN DETECTION - CLASS 0 ONLY
        if class_id == 0:
            return DetectionType.PEDESTRIAN, "pedestrian", class_id, 0.0
        
        # VEHICLE DETECTION
        elif class_id in self.VEHICLE_CLASSES:
            info = self.VEHICLE_CLASSES[class_id]
            return info['type'], info['name'], class_id, info['weight']
        
        # BICYCLE detection
        elif class_id == 1:
            return DetectionType.BICYCLE, "bicycle", class_id, 0.3
        
        # Unknown
        else:
            if self.debug_mode:
                logger.warning(f"Unknown class {class_id} detected")
            return DetectionType.VEHICLE, "unknown", class_id, 1.0
    
    def _validate_detection(self, detection: Detection) -> bool:
        """Validate detection based on area"""
        if detection.type == DetectionType.PEDESTRIAN:
            if detection.area > self.MAX_PEDESTRIAN_AREA:
                if self.debug_mode:
                    logger.warning(f"Large pedestrian detected (area: {detection.area})")
            return True
        
        if detection.type in [DetectionType.VEHICLE, DetectionType.MOTORCYCLE, 
                              DetectionType.BUS, DetectionType.TRUCK, DetectionType.AUTO]:
            if detection.area < self.MIN_VEHICLE_AREA and detection.confidence < 0.5:
                if self.debug_mode:
                    logger.debug(f"Filtering small vehicle (area: {detection.area})")
                return False
        
        return True
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Detect objects with weighted occupancy and spillover detection"""
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
            else:
                if detected:
                    self.crossing_polygon = detected.polygon
        
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
                
                detection_type, class_name, mapped_id, weight = self._classify_detection(class_id, [x1, y1, x2, y2], confidence)
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                
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
                    last_seen=current_time,
                    aspect_ratio=aspect_ratio,
                    weight=weight
                )
                
                if self._validate_detection(detection):
                    detections.append(detection)
        
        # Update vehicle tracking
        self._update_tracking(detections, current_time)
        
        # Update pedestrian tracker
        if self.pedestrian_tracker_enabled and self.pedestrian_tracker:
            tracked_peds, ped_stats = self.pedestrian_tracker.update(detections)
        
        # Update detection history
        self.detection_history.append(detections)
        
        # Annotate frame
        annotated = self._annotate(processed_frame, detections)
        
        # Add overlays
        self._add_overlays(annotated)
        
        return detections, annotated
    
    def _in_region(self, point: Tuple[int, int], region: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon"""
        if not region:
            return False
        return cv2.pointPolygonTest(np.array(region, dtype=np.int32), point, False) >= 0
    
    def _update_tracking(self, detections: List[Detection], current_time: float) -> None:
        """Update vehicle tracking"""
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
                tracked.weight = det.weight
                
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
        """Annotate frame with regions and detections"""
        annotated = frame.copy()
        
        # Draw lane region
        if self.lane_region:
            pts = np.array(self.lane_region, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
            cv2.putText(annotated, "FREE LEFT LANE", (self.lane_region[0][0], self.lane_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw crossing region
        if self.crossing_polygon:
            if self.manual_crossing_region is not None:
                color = (0, 255, 0)
                label = "MANUAL CROSSING"
            else:
                color = (0, 255, 255)
                label = "AUTO CROSSING"
            
            pts = np.array(self.crossing_polygon, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, color, 3)
            cv2.putText(annotated, label, (self.crossing_polygon[0][0], self.crossing_polygon[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw detections with color coding
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color coding based on type
            if det.type == DetectionType.PEDESTRIAN:
                color = (0, 0, 255)  # Red
                icon = "🚶"
            elif det.type == DetectionType.MOTORCYCLE:
                color = (0, 165, 255)  # Orange
                icon = "🏍️"
            elif det.type in [DetectionType.BUS, DetectionType.TRUCK]:
                color = (255, 0, 0)  # Blue
                icon = "🚛"
            else:
                color = (0, 255, 0)  # Green
                icon = "🚗"
            
            # Highlight blocking vehicles
            if det.type != DetectionType.PEDESTRIAN and det.in_lane and det.is_stationary:
                color = (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated, "BLOCKING!", (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            cv2.circle(annotated, det.center, 5, color, -1)
            
            # Label with weight indicator
            label = f"{icon} {det.class_name} ({det.confidence:.2f})"
            if det.weight > 1.0:
                label += f" [W:{det.weight:.1f}]"
            
            if det.type == DetectionType.PEDESTRIAN and det.at_crossing:
                label += " 🚶 AT CROSSING"
            elif det.type != DetectionType.PEDESTRIAN and det.in_lane and det.is_stationary:
                label += f" ⚠️ {det.stationary_time:.0f}s"
            elif det.type != DetectionType.PEDESTRIAN and det.in_lane:
                label += " ⚠️ IN LANE"
            
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated
    
    def _add_overlays(self, annotated: np.ndarray) -> None:
        """Add information overlays to frame"""
        # FPS
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mode status
        if self.manual_crossing_region is not None:
            cv2.putText(annotated, "MANUAL CROSSING MODE", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif self.auto_detect_crossing:
            if self.crossing_polygon:
                cv2.putText(annotated, "AUTO CROSSING DETECTED", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(annotated, "SCANNING FOR CROSSING...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Pedestrian tracking status
        if self.pedestrian_tracker_enabled:
            cv2.putText(annotated, "PEDESTRIAN TRACKING ACTIVE", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def get_blocking_vehicles(self, min_duration: float = 3.0) -> List[Detection]:
        """Get vehicles blocking free-left lane"""
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
    
    def get_pedestrian_stats(self) -> Dict:
        """Get pedestrian tracker statistics"""
        if self.pedestrian_tracker_enabled and self.pedestrian_tracker:
            return self.pedestrian_tracker.get_stats()
        return {}
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        ped_stats = self.get_pedestrian_stats()
        lane_vehicles = [d for d in self.tracker.values() if d.type != DetectionType.PEDESTRIAN and d.in_lane]
        
        return {
            'total_frames': self.frame_count,
            'fps': self.fps,
            'tracked_vehicles': len([d for d in self.tracker.values() if d.type != DetectionType.PEDESTRIAN]),
            'tracked_motorcycles': len([d for d in self.tracker.values() if d.type == DetectionType.MOTORCYCLE]),
            'tracked_pedestrians': len([d for d in self.tracker.values() if d.type == DetectionType.PEDESTRIAN]),
            'blocking_vehicles': len(self.get_blocking_vehicles()),
            'weighted_occupancy': self.get_weighted_occupancy(lane_vehicles),
            'spillover_detected': self.detect_spillover(lane_vehicles),
            'queue_length': self.get_queue_length(lane_vehicles),
            'occlusion_factor': self.get_occlusion_factor(lane_vehicles),
            'pedestrian_tracker_active': self.pedestrian_tracker_enabled,
            'pedestrian_waiting': ped_stats.get('waiting_count', 0),
            'pedestrian_crossing': ped_stats.get('crossing_count', 0),
            'total_pedestrians_crossed': ped_stats.get('total_crossed', 0)
        }
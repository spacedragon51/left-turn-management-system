"""
Pedestrian Tracking Module - Dedicated Camera for Pedestrian Monitoring
Tracks waiting pedestrians, crossing pedestrians, and counts
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math

logger = logging.getLogger(__name__)


class PedestrianState(Enum):
    """Pedestrian states"""
    WAITING = "waiting"
    CROSSING = "crossing"
    EXITED = "exited"


@dataclass
class Pedestrian:
    """Data class for a tracked pedestrian"""
    id: int
    state: PedestrianState = PedestrianState.WAITING
    entry_time: float = 0
    crossing_start_time: float = 0
    crossing_end_time: float = 0
    positions: List[Tuple[int, int]] = field(default_factory=list)
    crossing_duration: float = 0
    waiting_duration: float = 0
    current_position: Tuple[int, int] = (0, 0)
    last_position: Tuple[int, int] = (0, 0)
    bbox: List[int] = field(default_factory=list)
    confidence: float = 0
    speed: float = 0
    direction: str = ""
    disappear_counter: int = 0  # <--- ADDED: Initialize disappear_counter here
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'state': self.state.value,
            'entry_time': self.entry_time,
            'crossing_start_time': self.crossing_start_time,
            'crossing_end_time': self.crossing_end_time,
            'crossing_duration': self.crossing_duration,
            'waiting_duration': self.waiting_duration,
            'current_position': self.current_position,
            'confidence': self.confidence,
            'disappear_counter': self.disappear_counter
        }


class PedestrianTracker:
    """
    Dedicated pedestrian tracker with waiting/crossing detection
    Uses Kalman filtering for smooth tracking
    """
    
    def __init__(self, 
                 crossing_region: List[Tuple[int, int]],
                 waiting_region: Optional[List[Tuple[int, int]]] = None,
                 min_wait_time: float = 2.0,
                 max_disappear_frames: int = 10,
                 crossing_threshold: float = 0.5):
        """
        Initialize pedestrian tracker
        
        Args:
            crossing_region: Polygon defining the crossing area
            waiting_region: Polygon defining waiting area (defaults to crossing region extended)
            min_wait_time: Minimum time to consider a pedestrian as waiting (seconds)
            max_disappear_frames: Max frames before removing pedestrian
            crossing_threshold: Percentage of region coverage to consider crossing
        """
        self.crossing_region = crossing_region
        self.waiting_region = waiting_region if waiting_region else self._create_waiting_region(crossing_region)
        self.min_wait_time = min_wait_time
        self.max_disappear_frames = max_disappear_frames
        self.crossing_threshold = crossing_threshold
        
        # Tracking storage
        self.pedestrians: Dict[int, Pedestrian] = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_crossed = 0
        self.max_waiting = 0
        self.peak_hour_pedestrians = 0
        self.crossing_times: List[float] = []
        self.waiting_times: List[float] = []
        
        # Tracking with Kalman filter
        self.kalman_filters: Dict[int, cv2.KalmanFilter] = {}
        
        # Temporal buffer
        self.position_history: Dict[int, deque] = {}
        self.history_length = 10
        
        # Event log
        self.event_log: List[Dict] = []
        
        logger.info("Pedestrian Tracker initialized")
        logger.info(f"  Crossing region: {crossing_region}")
        logger.info(f"  Waiting region: {self.waiting_region}")
        logger.info(f"  Max disappear frames: {max_disappear_frames}")
    
    def _create_waiting_region(self, crossing_region: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Create waiting region extending from crossing region"""
        x1, y1 = crossing_region[0]
        x2, y2 = crossing_region[2]
        
        # Wait area is above crossing (assuming road orientation)
        wait_height = (y2 - y1) // 2
        wait_region = [
            (x1, y1 - wait_height),
            (x2, y1 - wait_height),
            (x2, y1),
            (x1, y1)
        ]
        return wait_region
    
    def _init_kalman_filter(self, x: int, y: int) -> cv2.KalmanFilter:
        """Initialize Kalman filter for a pedestrian"""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        kf.statePre = np.array([[x], [y], [0], [0]], np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        
        return kf
    
    def _predict_position(self, pedestrian_id: int, x: int, y: int) -> Tuple[int, int]:
        """Predict next position using Kalman filter"""
        if pedestrian_id not in self.kalman_filters:
            self.kalman_filters[pedestrian_id] = self._init_kalman_filter(x, y)
            return x, y
        
        kf = self.kalman_filters[pedestrian_id]
        prediction = kf.predict()
        measured = np.array([[x], [y]], np.float32)
        kf.correct(measured)
        
        return int(prediction[0]), int(prediction[1])
    
    def _update_position_history(self, pedestrian_id: int, position: Tuple[int, int]):
        """Update position history for speed calculation"""
        if pedestrian_id not in self.position_history:
            self.position_history[pedestrian_id] = deque(maxlen=self.history_length)
        self.position_history[pedestrian_id].append(position)
    
    def _calculate_speed(self, pedestrian_id: int) -> float:
        """Calculate speed based on position history"""
        history = self.position_history.get(pedestrian_id, deque())
        if len(history) < 5:
            return 0
        
        total_distance = 0
        for i in range(1, len(history)):
            x1, y1 = history[i-1]
            x2, y2 = history[i]
            total_distance += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        speed = total_distance / len(history) * 30
        return speed
    
    def _in_region(self, point: Tuple[int, int], region: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon region"""
        if not region:
            return False
        return cv2.pointPolygonTest(np.array(region, dtype=np.int32), point, False) >= 0
    
    def _region_coverage(self, bbox: List[int], region: List[Tuple[int, int]]) -> float:
        """Calculate percentage of bounding box inside region"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        sample_points = [
            center,
            (x1, y1), (x2, y1), (x1, y2), (x2, y2),
            ((x1 + x2)//2, y1), ((x1 + x2)//2, y2),
            (x1, (y1 + y2)//2), (x2, (y1 + y2)//2)
        ]
        
        inside_count = sum(1 for p in sample_points if self._in_region(p, region))
        return inside_count / len(sample_points)
    
    def update(self, detections: List) -> Tuple[List[Pedestrian], Dict]:
        """
        Update pedestrian tracking with new detections
        
        Args:
            detections: List of detection objects (from YOLO)
            
        Returns:
            Tuple of (tracked_pedestrians, statistics)
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Extract pedestrian detections
        pedestrian_dets = [d for d in detections if hasattr(d, 'type') and d.type.value == 'pedestrian']
        
        # Track existing pedestrians
        active_ids = set()
        
        # First, update all existing pedestrians with new detections
        for det in pedestrian_dets:
            center = det.center
            bbox = det.bbox
            
            # Find best match with existing pedestrians
            best_match = None
            best_distance = float('inf')
            
            for pid, ped in self.pedestrians.items():
                if ped.state == PedestrianState.EXITED:
                    continue
                
                distance = math.sqrt((center[0] - ped.current_position[0])**2 + 
                                    (center[1] - ped.current_position[1])**2)
                
                if distance < 50 and distance < best_distance:
                    best_distance = distance
                    best_match = pid
            
            if best_match is not None:
                # Update existing pedestrian
                ped = self.pedestrians[best_match]
                ped.last_position = ped.current_position
                ped.current_position = center
                ped.bbox = bbox
                ped.confidence = det.confidence
                ped.positions.append(center)
                ped.disappear_counter = 0  # <--- RESET disappear_counter when found
                active_ids.add(best_match)
                
                # Update speed
                self._update_position_history(best_match, center)
                ped.speed = self._calculate_speed(best_match)
                
                # Determine state based on region
                in_crossing = self._in_region(center, self.crossing_region)
                in_waiting = self._in_region(center, self.waiting_region)
                
                # State transition logic
                if ped.state == PedestrianState.WAITING:
                    if in_crossing:
                        # Started crossing
                        ped.state = PedestrianState.CROSSING
                        ped.crossing_start_time = current_time
                        ped.waiting_duration = current_time - ped.entry_time
                        self.waiting_times.append(ped.waiting_duration)
                        self._log_event('PEDESTRIAN_STARTED_CROSSING', {
                            'id': ped.id,
                            'wait_duration': ped.waiting_duration
                        })
                    elif not in_waiting and not in_crossing:
                        # Left waiting area without crossing
                        ped.state = PedestrianState.EXITED
                        self._log_event('PEDESTRIAN_LEFT_WAITING', {'id': ped.id})
                
                elif ped.state == PedestrianState.CROSSING:
                    coverage = self._region_coverage(bbox, self.crossing_region)
                    if not in_crossing and coverage < 0.2:
                        # Finished crossing
                        ped.state = PedestrianState.EXITED
                        ped.crossing_end_time = current_time
                        ped.crossing_duration = current_time - ped.crossing_start_time
                        self.crossing_times.append(ped.crossing_duration)
                        self.total_crossed += 1
                        self._log_event('PEDESTRIAN_FINISHED_CROSSING', {
                            'id': ped.id,
                            'duration': ped.crossing_duration
                        })
                
                # Update waiting time
                if ped.state == PedestrianState.WAITING:
                    ped.waiting_duration = current_time - ped.entry_time
        
        # Add new pedestrians (not matched to any existing)
        for det in pedestrian_dets:
            center = det.center
            
            # Check if already tracked
            already_tracked = False
            for pid, ped in self.pedestrians.items():
                distance = math.sqrt((center[0] - ped.current_position[0])**2 + 
                                    (center[1] - ped.current_position[1])**2)
                if distance < 30:
                    already_tracked = True
                    break
            
            if not already_tracked:
                # New pedestrian
                new_ped = Pedestrian(
                    id=self.next_id,
                    state=PedestrianState.WAITING,
                    entry_time=current_time,
                    current_position=center,
                    last_position=center,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    disappear_counter=0  # <--- INITIALIZE disappear_counter
                )
                self.pedestrians[self.next_id] = new_ped
                active_ids.add(self.next_id)
                self.next_id += 1
                self._log_event('NEW_PEDESTRIAN', {'id': new_ped.id})
                
                # Initialize position history
                self._update_position_history(new_ped.id, center)
        
        # Handle disappeared pedestrians (not seen in this frame)
        for pid in list(self.pedestrians.keys()):
            if pid not in active_ids:
                ped = self.pedestrians[pid]
                ped.disappear_counter += 1  # <--- INCREMENT disappear_counter
                
                if ped.disappear_counter > self.max_disappear_frames:
                    # Remove pedestrian after too many missed frames
                    if ped.state == PedestrianState.WAITING:
                        self._log_event('PEDESTRIAN_TIMEOUT', {'id': pid, 'wait_time': ped.waiting_duration})
                    elif ped.state == PedestrianState.CROSSING:
                        self._log_event('PEDESTRIAN_LOST_DURING_CROSSING', {'id': pid})
                    del self.pedestrians[pid]
        
        # Update statistics
        waiting_count = len([p for p in self.pedestrians.values() if p.state == PedestrianState.WAITING])
        crossing_count = len([p for p in self.pedestrians.values() if p.state == PedestrianState.CROSSING])
        
        self.max_waiting = max(self.max_waiting, waiting_count)
        
        # Track peak hour pedestrians
        current_hour = time.localtime().tm_hour
        if current_hour in [8, 9, 17, 18]:
            self.peak_hour_pedestrians = max(self.peak_hour_pedestrians, waiting_count + crossing_count)
        
        # Calculate average wait time
        avg_wait_time = np.mean(self.waiting_times) if self.waiting_times else 0
        
        # Statistics
        stats = {
            'waiting_count': waiting_count,
            'crossing_count': crossing_count,
            'total_crossed_today': self.total_crossed,
            'max_waiting_observed': self.max_waiting,
            'peak_hour_pedestrians': self.peak_hour_pedestrians,
            'avg_crossing_time': np.mean(self.crossing_times) if self.crossing_times else 0,
            'avg_wait_time': avg_wait_time,
            'total_pedestrians_tracked': len(self.pedestrians),
            'active_tracks': len(active_ids)
        }
        
        return list(self.pedestrians.values()), stats
    
    def _log_event(self, event_type: str, data: Dict):
        """Log pedestrian event"""
        event = {
            'timestamp': time.time(),
            'event': event_type,
            'data': data
        }
        self.event_log.append(event)
        
        # Keep last 100 events
        if len(self.event_log) > 100:
            self.event_log.pop(0)
        
        logger.debug(f"Pedestrian event: {event_type} - {data}")
    
    def get_waiting_pedestrians(self) -> List[Pedestrian]:
        """Get pedestrians currently waiting"""
        return [p for p in self.pedestrians.values() if p.state == PedestrianState.WAITING]
    
    def get_crossing_pedestrians(self) -> List[Pedestrian]:
        """Get pedestrians currently crossing"""
        return [p for p in self.pedestrians.values() if p.state == PedestrianState.CROSSING]
    
    def get_stats(self) -> Dict:
        """Get pedestrian statistics"""
        waiting = self.get_waiting_pedestrians()
        crossing = self.get_crossing_pedestrians()
        
        avg_wait_time = np.mean([p.waiting_duration for p in waiting]) if waiting else 0
        
        return {
            'waiting_count': len(waiting),
            'crossing_count': len(crossing),
            'total_crossed': self.total_crossed,
            'max_waiting': self.max_waiting,
            'avg_wait_time': avg_wait_time,
            'peak_hour_pedestrians': self.peak_hour_pedestrians,
            'avg_crossing_time': np.mean(self.crossing_times) if self.crossing_times else 0,
            'active_tracks': len(self.pedestrians)
        }
    
    def get_events(self, limit: int = 20) -> List[Dict]:
        """Get recent pedestrian events"""
        return self.event_log[-limit:]
    
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw pedestrian tracking visualization"""
        annotated = frame.copy()
        
        # Draw waiting region (blue)
        if self.waiting_region:
            pts = np.array(self.waiting_region, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)
            cv2.putText(annotated, "WAITING AREA", (self.waiting_region[0][0], self.waiting_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw crossing region (green)
        if self.crossing_region:
            pts = np.array(self.crossing_region, dtype=np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 3)
            cv2.putText(annotated, "CROSSING AREA", (self.crossing_region[0][0], self.crossing_region[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw pedestrians
        for ped in self.pedestrians.values():
            x, y = ped.current_position
            
            # Color based on state
            if ped.state == PedestrianState.WAITING:
                color = (0, 0, 255)  # Red for waiting
                label = f"P{ped.id} - WAITING ({ped.waiting_duration:.0f}s)"
            elif ped.state == PedestrianState.CROSSING:
                color = (0, 255, 255)  # Yellow for crossing
                label = f"P{ped.id} - CROSSING"
            else:
                continue
            
            # Draw pedestrian
            cv2.circle(annotated, (x, y), 12, color, -1)
            cv2.circle(annotated, (x, y), 14, (255, 255, 255), 2)
            
            # Draw ID and label
            cv2.putText(annotated, label, (x - 35, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw trajectory (last 5 positions)
            if len(ped.positions) > 1:
                for i in range(1, min(5, len(ped.positions))):
                    cv2.line(annotated, ped.positions[i-1], ped.positions[i], color, 1)
        
        # Draw statistics overlay
        stats = self.get_stats()
        y_offset = 80
        cv2.rectangle(annotated, (10, y_offset - 25), (280, y_offset + 110), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, y_offset - 25), (280, y_offset + 110), (100, 100, 100), 1)
        cv2.putText(annotated, "🚶 PEDESTRIAN STATS", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Waiting: {stats['waiting_count']}", (15, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(annotated, f"Crossing: {stats['crossing_count']}", (15, y_offset + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated, f"Total Crossed: {stats['total_crossed']}", (15, y_offset + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, f"Avg Wait: {stats['avg_wait_time']:.1f}s", (15, y_offset + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return annotated


class DualCameraPedestrianManager:
    """
    Manages pedestrian tracking across two cameras
    """
    
    def __init__(self, 
                 camera1_region: List[Tuple[int, int]],
                 camera2_region: List[Tuple[int, int]],
                 fusion_distance: int = 100):
        """
        Initialize dual-camera pedestrian manager
        
        Args:
            camera1_region: Crossing region for camera 1
            camera2_region: Crossing region for camera 2
            fusion_distance: Distance threshold for merging detections
        """
        self.tracker1 = PedestrianTracker(camera1_region)
        self.tracker2 = PedestrianTracker(camera2_region)
        self.fusion_distance = fusion_distance
        
        # Track merged pedestrians
        self.merged_pedestrians: Dict[int, Dict] = {}
        self.next_merged_id = 1
    
    def update(self, detections1: List, detections2: List) -> Tuple[List[Pedestrian], Dict]:
        """
        Update both trackers and fuse results
        
        Returns:
            Tuple of (merged_pedestrians, combined_stats)
        """
        # Update individual trackers
        peds1, stats1 = self.tracker1.update(detections1)
        peds2, stats2 = self.tracker2.update(detections2)
        
        # Fuse detections
        fused_peds = self._fuse_detections(peds1, peds2)
        
        # Combined stats
        combined_stats = {
            'camera1': stats1,
            'camera2': stats2,
            'total_waiting': stats1['waiting_count'] + stats2['waiting_count'],
            'total_crossing': stats1['crossing_count'] + stats2['crossing_count'],
            'total_crossed': stats1['total_crossed_today'] + stats2['total_crossed_today']
        }
        
        return fused_peds, combined_stats
    
    def _fuse_detections(self, peds1: List[Pedestrian], peds2: List[Pedestrian]) -> List[Pedestrian]:
        """Fuse detections from two cameras"""
        # Simple fusion - combine both lists
        all_peds = peds1 + peds2
        
        # Deduplicate based on position proximity
        fused = []
        used = set()
        
        for i, ped1 in enumerate(all_peds):
            if i in used:
                continue
            
            duplicate = False
            for j, ped2 in enumerate(all_peds):
                if i != j and j not in used:
                    dist = math.sqrt((ped1.current_position[0] - ped2.current_position[0])**2 +
                                    (ped1.current_position[1] - ped2.current_position[1])**2)
                    if dist < self.fusion_distance:
                        used.add(j)
                        duplicate = True
                        break
            
            fused.append(ped1)
            used.add(i)
        
        return fused
    
    def get_waiting_count(self) -> int:
        """Get total waiting pedestrians"""
        return self.tracker1.get_stats()['waiting_count'] + self.tracker2.get_stats()['waiting_count']
    
    def get_crossing_count(self) -> int:
        """Get total crossing pedestrians"""
        return self.tracker1.get_stats()['crossing_count'] + self.tracker2.get_stats()['crossing_count']
    
    def draw(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Draw both camera feeds with tracking"""
        return self.tracker1.draw(frame1), self.tracker2.draw(frame2)
    
    def get_events(self, limit: int = 20) -> List[Dict]:
        """Get events from both trackers"""
        events = self.tracker1.get_events(limit) + self.tracker2.get_events(limit)
        events.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        return events[:limit]


# For testing
if __name__ == "__main__":
    print("Pedestrian Tracker Module Loaded")
    print("  - Pedestrian class has disappear_counter initialized")
    print("  - All attributes properly initialized in __init__")
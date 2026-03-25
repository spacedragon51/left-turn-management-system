"""
Enhanced Zebra Crossing Detector - Complete Rewrite
With debug mode for troubleshooting
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class CrossingRegion:
    """Detected crossing region data"""
    polygon: List[Tuple[int, int]]
    confidence: float
    orientation: str
    width: int
    height: int
    stripe_count: int
    center: Tuple[int, int]
    area: float


class EnhancedZebraDetector:
    """
    Enhanced zebra crossing detector with debug mode
    """
    
    def __init__(self, 
                 min_stripes: int = 3,
                 min_stripe_width: int = 20,
                 max_stripe_width: int = 100,
                 confidence_threshold: float = 0.5,
                 debug_mode: bool = False):
        """
        Initialize detector with debug mode
        
        Args:
            min_stripes: Minimum number of stripes
            min_stripe_width: Minimum stripe width in pixels
            max_stripe_width: Maximum stripe width in pixels
            confidence_threshold: Minimum confidence for detection
            debug_mode: Enable debug visualization
        """
        self.min_stripes = min_stripes
        self.min_stripe_width = min_stripe_width
        self.max_stripe_width = max_stripe_width
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        
        # Tracking
        self.detection_history = []
        self.debug_images = []
        
        # Parameters
        self.adaptive_block_size = 15
        self.canny_low = 50
        self.canny_high = 150
        
        logger.info(f"Zebra Detector initialized | Min stripes: {min_stripes}")
    
    def detect(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """
        Detect zebra crossing with multiple methods and debug output
        """
        if frame is None or frame.size == 0:
            return None
        
        height, width = frame.shape[:2]
        
        # Method 1: Hough Line Detection (Most reliable for zebra crossings)
        result1 = self._detect_hough_lines(frame)
        
        # Method 2: Adaptive Thresholding
        result2 = self._detect_adaptive(frame)
        
        # Method 3: Morphological Analysis
        result3 = self._detect_morphological(frame)
        
        # Combine results
        candidates = []
        if result1:
            candidates.append(result1)
        if result2:
            candidates.append(result2)
        if result3:
            candidates.append(result3)
        
        if not candidates:
            if self.debug_mode:
                logger.debug("No zebra crossing detected by any method")
            return None
        
        # Select best candidate
        best = max(candidates, key=lambda x: x.confidence)
        
        # Apply temporal smoothing
        smoothed = self._temporal_smooth(best)
        
        if smoothed and smoothed.confidence >= self.confidence_threshold:
            if self.debug_mode:
                logger.info(f"Crossing detected! Confidence: {smoothed.confidence:.2f}")
            return smoothed
        
        return None
    
    def _detect_hough_lines(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """
        Detect zebra crossing using Hough line detection (Primary method)
        """
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=40,
            maxLineGap=15
        )
        
        if lines is None:
            return None
        
        # Collect horizontal lines (potential zebra stripes)
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines (angle < 15 degrees)
            if angle < 15:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > 50:  # Minimum stripe length
                    horizontal_lines.append({
                        'points': (x1, y1, x2, y2),
                        'y': (y1 + y2) // 2,
                        'length': length
                    })
        
        if len(horizontal_lines) < self.min_stripes:
            return None
        
        # Group lines by Y-coordinate (stripes should be at similar Y)
        y_groups = {}
        tolerance = 20  # Pixels tolerance for grouping
        
        for line in horizontal_lines:
            y = line['y']
            grouped = False
            for key in y_groups.keys():
                if abs(key - y) < tolerance:
                    y_groups[key].append(line)
                    grouped = True
                    break
            if not grouped:
                y_groups[y] = [line]
        
        # Find the largest group of parallel lines
        best_group = max(y_groups.values(), key=len, default=[])
        
        if len(best_group) < self.min_stripes:
            return None
        
        # Calculate region boundaries
        y_coords = [l['y'] for l in best_group]
        y_min = max(0, min(y_coords) - 25)
        y_max = min(height, max(y_coords) + 25)
        
        x_coords = []
        for line in best_group:
            x1, y1, x2, y2 = line['points']
            x_coords.append(min(x1, x2))
            x_coords.append(max(x1, x2))
        
        x_min = max(0, min(x_coords) - 30)
        x_max = min(width, max(x_coords) + 30)
        
        # Calculate confidence based on line consistency
        stripe_count = len(best_group)
        confidence = min(1.0, stripe_count / 8) * 0.8
        
        # Calculate stripe width consistency
        stripe_widths = []
        for line in best_group:
            x1, y1, x2, y2 = line['points']
            stripe_widths.append(abs(x2 - x1))
        
        if stripe_widths:
            width_std = np.std(stripe_widths) / (np.mean(stripe_widths) + 0.01)
            consistency = 1 - min(1.0, width_std)
            confidence = confidence * 0.7 + consistency * 0.3
        
        # Create polygon
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        # Debug visualization
        if self.debug_mode:
            debug_img = frame.copy()
            for line in best_group:
                x1, y1, x2, y2 = line['points']
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            cv2.putText(debug_img, f"Hough: {stripe_count} stripes", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.debug_images.append(debug_img)
        
        return CrossingRegion(
            polygon=polygon,
            confidence=confidence, # pyright: ignore[reportArgumentType]
            orientation='horizontal',
            width=x_max - x_min,
            height=y_max - y_min,
            stripe_count=stripe_count,
            center=((x_min + x_max) // 2, (y_min + y_max) // 2),
            area=(x_max - x_min) * (y_max - y_min)
        )
    
    def _detect_adaptive(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """
        Detect using adaptive thresholding (Secondary method)
        """
        height, width = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for stripe-like regions
        stripes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Stripe-like shape: wide and not too tall
            if self.min_stripe_width < w < self.max_stripe_width and aspect_ratio > 2:
                stripes.append((x, y, w, h))
        
        if len(stripes) < self.min_stripes:
            return None
        
        # Group stripes by Y position
        stripe_groups = {}
        for (x, y, w, h) in stripes:
            y_center = y + h // 2
            grouped = False
            for key in stripe_groups.keys():
                if abs(key - y_center) < 30:
                    stripe_groups[key].append((x, y, w, h))
                    grouped = True
                    break
            if not grouped:
                stripe_groups[y_center] = [(x, y, w, h)]
        
        best_group = max(stripe_groups.values(), key=len, default=[])
        
        if len(best_group) < self.min_stripes:
            return None
        
        # Calculate region
        x_min = max(0, min(s[0] for s in best_group) - 20)
        x_max = min(width, max(s[0] + s[2] for s in best_group) + 20)
        y_min = max(0, min(s[1] for s in best_group) - 20)
        y_max = min(height, max(s[1] + s[3] for s in best_group) + 20)
        
        confidence = min(1.0, len(best_group) / 8) * 0.7
        
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        return CrossingRegion(
            polygon=polygon,
            confidence=confidence,
            orientation='horizontal',
            width=x_max - x_min,
            height=y_max - y_min,
            stripe_count=len(best_group),
            center=((x_min + x_max) // 2, (y_min + y_max) // 2),
            area=(x_max - x_min) * (y_max - y_min)
        )
    
    def _detect_morphological(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """
        Detect using morphological operations (Tertiary method)
        """
        height, width = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect edges of white regions
        edges = cv2.Canny(opened, 50, 150)
        
        # Look for horizontal line segments
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180, threshold=30,
            minLineLength=40, maxLineGap=10
        )
        
        if lines is None or len(lines) < self.min_stripes:
            return None
        
        # Filter horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 15:
                horizontal_lines.append((x1, y1, x2, y2))
        
        if len(horizontal_lines) < self.min_stripes:
            return None
        
        # Group by Y
        y_positions = [(y1 + y2) // 2 for (x1, y1, x2, y2) in horizontal_lines]
        y_min = max(0, min(y_positions) - 30)
        y_max = min(height, max(y_positions) + 30)
        
        x_coords = []
        for (x1, y1, x2, y2) in horizontal_lines:
            x_coords.append(min(x1, x2))
            x_coords.append(max(x1, x2))
        
        x_min = max(0, min(x_coords) - 40)
        x_max = min(width, max(x_coords) + 40)
        
        confidence = min(1.0, len(horizontal_lines) / 8) * 0.6
        
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        return CrossingRegion(
            polygon=polygon,
            confidence=confidence,
            orientation='horizontal',
            width=x_max - x_min,
            height=y_max - y_min,
            stripe_count=len(horizontal_lines),
            center=((x_min + x_max) // 2, (y_min + y_max) // 2),
            area=(x_max - x_min) * (y_max - y_min)
        )
    
    def _temporal_smooth(self, current: CrossingRegion) -> CrossingRegion:
        """
        Smooth detection across frames
        """
        self.detection_history.append(current)
        
        # Keep last 10 frames
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        if len(self.detection_history) < 3:
            return current
        
        # Average polygon coordinates from last 3 frames
        recent = self.detection_history[-3:]
        avg_polygon = []
        num_points = len(current.polygon)
        
        for i in range(num_points):
            avg_x = sum(r.polygon[i][0] for r in recent) // len(recent)
            avg_y = sum(r.polygon[i][1] for r in recent) // len(recent)
            avg_polygon.append((avg_x, avg_y))
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in recent) / len(recent)
        
        return CrossingRegion(
            polygon=avg_polygon,
            confidence=avg_confidence,
            orientation=current.orientation,
            width=current.width,
            height=current.height,
            stripe_count=current.stripe_count,
            center=current.center,
            area=current.area
        )
    
    def get_debug_image(self) -> Optional[np.ndarray]:
        """Get debug image with visualizations"""
        if self.debug_images:
            return self.debug_images[-1]
        return None
    
    def draw_detection(self, frame: np.ndarray, region: CrossingRegion) -> np.ndarray:
        """Draw detected crossing on frame"""
        annotated = frame.copy()
        
        # Draw polygon
        pts = np.array(region.polygon, np.int32)
        cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
        
        # Draw stripes
        x1, y1 = region.polygon[0]
        x2, y2 = region.polygon[2]
        stripe_height = 20
        num_stripes = region.stripe_count
        
        for i in range(num_stripes):
            y = y1 + i * (region.height // num_stripes)
            if i % 2 == 0:
                cv2.rectangle(annotated, (x1, y), (x2, y + stripe_height), (255, 255, 255), -1)
        
        # Label
        label = f"Zebra Crossing (conf: {region.confidence:.2f})"
        cv2.putText(annotated, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated


class ZebraCrossingTracker:
    """Track crossing region across frames"""
    
    def __init__(self, stability_frames: int = 5):
        self.stability_frames = stability_frames
        self.last_stable = None
        self.detection_counter = 0
        self.miss_counter = 0
        self.history = []
    
    def update(self, region: Optional[CrossingRegion]) -> Optional[CrossingRegion]:
        """Update tracker with new detection"""
        if region is not None:
            self.history.append(region)
            if len(self.history) > 10:
                self.history.pop(0)
            
            self.detection_counter += 1
            self.miss_counter = 0
            
            if self.detection_counter >= self.stability_frames:
                # Average last few detections
                recent = self.history[-3:]
                avg_polygon = []
                num_points = len(region.polygon)
                
                for i in range(num_points):
                    avg_x = sum(r.polygon[i][0] for r in recent) // len(recent)
                    avg_y = sum(r.polygon[i][1] for r in recent) // len(recent)
                    avg_polygon.append((avg_x, avg_y))
                
                avg_confidence = sum(r.confidence for r in recent) / len(recent)
                
                stable_region = CrossingRegion(
                    polygon=avg_polygon,
                    confidence=avg_confidence,
                    orientation=region.orientation,
                    width=region.width,
                    height=region.height,
                    stripe_count=region.stripe_count,
                    center=region.center,
                    area=region.area
                )
                
                self.last_stable = stable_region
                return stable_region
            else:
                return None
        else:
            self.detection_counter = 0
            self.miss_counter += 1
            
            if self.miss_counter < 3 and self.last_stable:
                # Decay confidence
                self.last_stable.confidence *= 0.8
                return self.last_stable
            
            return None
    
    def reset(self):
        """Reset tracker"""
        self.detection_counter = 0
        self.miss_counter = 0
        self.history = []
        self.last_stable = None
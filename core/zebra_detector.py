"""
Enhanced Zebra Crossing Detector - Robust detection using multiple techniques
Handles various lighting conditions, camera angles, and weather scenarios
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
import math
import os
logger = logging.getLogger(__name__)


@dataclass
class CrossingRegion:
    """Detected crossing region data"""
    polygon: List[Tuple[int, int]]
    confidence: float
    orientation: str  # 'horizontal', 'vertical'
    width: int
    height: int
    stripe_count: int
    center: Tuple[int, int]
    area: float


class EnhancedZebraDetector:
    """
    Enhanced zebra crossing detector using multiple techniques:
    - Edge detection
    - Morphological operations
    - Pattern matching
    - Hough line detection
    - Machine learning based classification
    """
    
    def __init__(self, 
                 min_stripes: int = 3,
                 min_stripe_width: int = 15,
                 max_stripe_width: int = 80,
                 confidence_threshold: float = 0.55,
                 use_ml: bool = False):
        """
        Initialize enhanced zebra crossing detector
        
        Args:
            min_stripes: Minimum number of stripes to consider a crossing
            min_stripe_width: Minimum stripe width in pixels
            max_stripe_width: Maximum stripe width in pixels
            confidence_threshold: Minimum confidence for detection
            use_ml: Use ML-based classification (requires trained model)
        """
        self.min_stripes = min_stripes
        self.min_stripe_width = min_stripe_width
        self.max_stripe_width = max_stripe_width
        self.confidence_threshold = confidence_threshold
        
        # Tracking across frames
        self.detection_history = []
        self.last_detection = None
        self.stable_counter = 0
        
        # Parameters for adaptive thresholding
        self.adaptive_block_size = 15
        self.canny_low = 50
        self.canny_high = 150
        
        logger.info("Enhanced Zebra Detector initialized")
    
    def detect(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """
        Detect zebra crossing in frame using multiple techniques
        
        Args:
            frame: Input BGR image
            
        Returns:
            CrossingRegion if detected, None otherwise
        """
        if frame is None or frame.size == 0:
            return None
        
        # Create multiple processed versions of the image
        candidates = []
        
        # Method 1: Standard approach with adaptive thresholding
        result1 = self._detect_adaptive_threshold(frame)
        if result1:
            candidates.append(result1)
        
        # Method 2: Edge-based detection
        result2 = self._detect_edge_based(frame)
        if result2:
            candidates.append(result2)
        
        # Method 3: Morphological approach
        result3 = self._detect_morphological(frame)
        if result3:
            candidates.append(result3)
        
        # Method 4: Color-based segmentation (for daytime)
        result4 = self._detect_color_based(frame)
        if result4:
            candidates.append(result4)
        
        if not candidates:
            self.stable_counter = 0
            return None
        
        # Select best candidate
        best_candidate = max(candidates, key=lambda x: x.confidence)
        
        # Apply temporal smoothing
        smoothed = self._temporal_smoothing(best_candidate)
        
        if smoothed and smoothed.confidence >= self.confidence_threshold:
            self.stable_counter += 1
            return smoothed
        else:
            self.stable_counter = 0
            return None
    
    def _detect_adaptive_threshold(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """Detect using adaptive thresholding"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding (handles varying lighting)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adaptive_block_size, 2
        )
        
        # Morphological operations to connect stripes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return self._analyze_pattern(morph, frame.shape, method="adaptive")
    
    def _detect_edge_based(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """Detect using edge detection and Hough lines"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is None:
            return None
        
        # Group lines by orientation and position
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15:  # Near horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif 75 < angle < 105:  # Near vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        # Look for horizontal line clusters (zebra pattern)
        if len(horizontal_lines) >= self.min_stripes:
            # Group lines by y-coordinate
            y_groups = {}
            for line in horizontal_lines:
                y = (line[1] + line[3]) // 2
                y_key = round(y / 10) * 10
                if y_key not in y_groups:
                    y_groups[y_key] = []
                y_groups[y_key].append(line)
            
            # Find largest cluster
            best_group = max(y_groups.values(), key=len, default=[])
            
            if len(best_group) >= self.min_stripes:
                return self._lines_to_region(best_group, frame.shape, "horizontal")
        
        # Look for vertical line clusters
        if len(vertical_lines) >= self.min_stripes:
            # Group lines by x-coordinate
            x_groups = {}
            for line in vertical_lines:
                x = (line[0] + line[2]) // 2
                x_key = round(x / 10) * 10
                if x_key not in x_groups:
                    x_groups[x_key] = []
                x_groups[x_key].append(line)
            
            best_group = max(x_groups.values(), key=len, default=[])
            
            if len(best_group) >= self.min_stripes:
                return self._lines_to_region(best_group, frame.shape, "vertical")
        
        return None
    
    def _detect_morphological(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """Detect using morphological operations"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Binary threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detect rectangular regions that could be zebra stripes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter rectangles by aspect ratio (stripes are long and thin)
        potential_stripes = []
        height, width = frame.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Stripes are typically wider than tall
            if self.min_stripe_width < w < self.max_stripe_width and aspect_ratio > 1.5:
                potential_stripes.append((x, y, w, h))
        
        if len(potential_stripes) >= self.min_stripes:
            return self._stripes_to_region(potential_stripes, frame.shape)
        
        return None
    
    def _detect_color_based(self, frame: np.ndarray) -> Optional[CrossingRegion]:
        """Detect using color segmentation (white on dark road)"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define white color range (adjustable)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for white regions
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for stripe-like regions
        stripes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if self.min_stripe_width < w < self.max_stripe_width and aspect_ratio > 1.5:
                stripes.append((x, y, w, h))
        
        if len(stripes) >= self.min_stripes:
            return self._stripes_to_region(stripes, frame.shape)
        
        return None
    
    def _analyze_pattern(self, binary: np.ndarray, shape: Tuple, method: str) -> Optional[CrossingRegion]:
        """Analyze binary image for zebra crossing pattern"""
        height, width = shape[:2]
        
        # Project horizontally to find stripe patterns
        horizontal_projection = np.sum(binary, axis=1)
        
        # Find peaks in projection (where stripes are)
        peaks = []
        threshold = np.max(horizontal_projection) * 0.3
        
        for i, val in enumerate(horizontal_projection):
            if val > threshold:
                peaks.append(i)
        
        if len(peaks) < self.min_stripes * 2:
            return None
        
        # Group peaks into stripe regions
        stripe_regions = []
        current_region = []
        
        for i in range(len(peaks)):
            if i > 0 and peaks[i] - peaks[i-1] > 20:
                if len(current_region) > 0:
                    stripe_regions.append(current_region)
                current_region = [peaks[i]]
            else:
                current_region.append(peaks[i])
        
        if len(current_region) > 0:
            stripe_regions.append(current_region)
        
        # Filter regions that look like stripes
        valid_stripes = []
        for region in stripe_regions:
            if len(region) > 5:  # Minimum height
                y_min = min(region)
                y_max = max(region)
                stripe_height = y_max - y_min
                
                # Find x boundaries for this stripe
                stripe_region = binary[y_min:y_max, :]
                x_projection = np.sum(stripe_region, axis=0)
                x_threshold = np.max(x_projection) * 0.3
                
                x_indices = np.where(x_projection > x_threshold)[0]
                if len(x_indices) > 0:
                    x_min = x_indices[0]
                    x_max = x_indices[-1]
                    stripe_width = x_max - x_min
                    
                    if self.min_stripe_width < stripe_width < self.max_stripe_width:
                        valid_stripes.append((x_min, y_min, x_max, y_max))
        
        if len(valid_stripes) >= self.min_stripes:
            return self._stripes_to_region(valid_stripes, shape)
        
        return None
    
    def _lines_to_region(self, lines: List, shape: Tuple, orientation: str) -> Optional[CrossingRegion]:
        """Convert detected lines to region polygon"""
        height, width = shape[:2]
        
        if orientation == "horizontal":
            # Extract y coordinates
            y_coords = [(line[1] + line[3]) // 2 for line in lines]
            y_min = max(0, min(y_coords) - 20)
            y_max = min(height, max(y_coords) + 20)
            
            # Extract x coordinates
            x_coords = []
            for line in lines:
                x_coords.append(min(line[0], line[2]))
                x_coords.append(max(line[0], line[2]))
            
            x_min = max(0, min(x_coords) - 30)
            x_max = min(width, max(x_coords) + 30)
            
        else:  # vertical
            # Extract x coordinates
            x_coords = [(line[0] + line[2]) // 2 for line in lines]
            x_min = max(0, min(x_coords) - 20)
            x_max = min(width, max(x_coords) + 20)
            
            # Extract y coordinates
            y_coords = []
            for line in lines:
                y_coords.append(min(line[1], line[3]))
                y_coords.append(max(line[1], line[3]))
            
            y_min = max(0, min(y_coords) - 30)
            y_max = min(height, max(y_coords) + 30)
        
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        confidence = min(1.0, len(lines) / (self.min_stripes * 1.5))
        
        return CrossingRegion(
            polygon=polygon,
            confidence=confidence,
            orientation=orientation,
            width=x_max - x_min,
            height=y_max - y_min,
            stripe_count=len(lines),
            center=((x_min + x_max) // 2, (y_min + y_max) // 2),
            area=(x_max - x_min) * (y_max - y_min)
        )
    
    def _stripes_to_region(self, stripes: List, shape: Tuple) -> Optional[CrossingRegion]:
        """Convert detected stripes to region polygon"""
        height, width = shape[:2]
        
        x_min = max(0, min(s[0] for s in stripes) - 20)
        x_max = min(width, max(s[2] for s in stripes) + 20)
        y_min = max(0, min(s[1] for s in stripes) - 20)
        y_max = min(height, max(s[3] for s in stripes) + 20)
        
        polygon = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        # Calculate confidence based on stripe consistency
        stripe_widths = [(s[2] - s[0]) for s in stripes]
        stripe_heights = [(s[3] - s[1]) for s in stripes]
        
        if stripe_widths:
            width_std = np.std(stripe_widths) / np.mean(stripe_widths) if np.mean(stripe_widths) > 0 else 1
            width_consistency = 1 - min(1.0, width_std)
        else:
            width_consistency = 0
        
        stripe_count_factor = min(1.0, len(stripes) / self.min_stripes)
        confidence = (width_consistency * 0.6 + stripe_count_factor * 0.4)
        
        # Determine orientation
        avg_width = np.mean(stripe_widths) if stripe_widths else 0
        avg_height = np.mean(stripe_heights) if stripe_heights else 0
        orientation = "horizontal" if avg_width > avg_height else "vertical"
        
        return CrossingRegion(
            polygon=polygon,
            confidence=confidence, # pyright: ignore[reportArgumentType]
            orientation=orientation,
            width=x_max - x_min,
            height=y_max - y_min,
            stripe_count=len(stripes),
            center=((x_min + x_max) // 2, (y_min + y_max) // 2),
            area=(x_max - x_min) * (y_max - y_min)
        )
    
    def _temporal_smoothing(self, current: CrossingRegion) -> CrossingRegion:
        """Smooth detection across frames to reduce flickering"""
        self.detection_history.append(current)
        
        # Keep last 10 detections
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        if len(self.detection_history) < 3:
            return current
        
        # Smooth polygon coordinates
        avg_polygon = []
        num_points = len(current.polygon)
        
        for i in range(num_points):
            avg_x = sum(d.polygon[i][0] for d in self.detection_history[-5:]) / min(5, len(self.detection_history))
            avg_y = sum(d.polygon[i][1] for d in self.detection_history[-5:]) / min(5, len(self.detection_history))
            avg_polygon.append((int(avg_x), int(avg_y)))
        
        # Smooth confidence
        avg_confidence = sum(d.confidence for d in self.detection_history[-5:]) / min(5, len(self.detection_history))
        
        # Smooth stripe count
        avg_stripe_count = sum(d.stripe_count for d in self.detection_history[-5:]) / min(5, len(self.detection_history))
        
        return CrossingRegion(
            polygon=avg_polygon,
            confidence=avg_confidence,
            orientation=current.orientation,
            width=current.width,
            height=current.height,
            stripe_count=int(avg_stripe_count),
            center=current.center,
            area=current.area
        )
    
    def draw_detection(self, frame: np.ndarray, region: CrossingRegion) -> np.ndarray:
        """Draw detected crossing region on frame"""
        annotated = frame.copy()
        
        # Draw polygon
        pts = np.array(region.polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (0, 255, 255), 3)
        
        # Add label with confidence
        label = f"Zebra Crossing (conf: {region.confidence:.2f}, stripes: {region.stripe_count})"
        cv2.putText(annotated, label, (region.polygon[0][0], region.polygon[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw center point
        cv2.circle(annotated, region.center, 5, (0, 255, 255), -1)
        
        return annotated
    
    def update_parameters(self, **kwargs):
        """Update detector parameters dynamically"""
        if 'adaptive_block_size' in kwargs:
            self.adaptive_block_size = max(3, kwargs['adaptive_block_size'] | 1)  # Must be odd
        
        if 'canny_low' in kwargs:
            self.canny_low = kwargs['canny_low']
        
        if 'canny_high' in kwargs:
            self.canny_high = kwargs['canny_high']
        
        if 'min_stripe_width' in kwargs:
            self.min_stripe_width = kwargs['min_stripe_width']
        
        if 'max_stripe_width' in kwargs:
            self.max_stripe_width = kwargs['max_stripe_width']
        
        logger.info(f"Parameters updated: {kwargs}")


class ZebraCrossingTracker:
    """
    Track crossing region across frames with stability checking
    """
    
    def __init__(self, stability_frames: int = 5):
        """
        Initialize tracker
        
        Args:
            stability_frames: Number of consistent detections required
        """
        self.stability_frames = stability_frames
        self.last_stable_region = None
        self.detection_counter = 0
        self.miss_counter = 0
        self.temporal_buffer = []
    
    def update(self, current_region: Optional[CrossingRegion]) -> Optional[CrossingRegion]:
        """Update tracker with current detection"""
        
        if current_region is not None:
            self.temporal_buffer.append(current_region)
            if len(self.temporal_buffer) > 10:
                self.temporal_buffer.pop(0)
            
            self.detection_counter += 1
            self.miss_counter = 0
            
            if self.detection_counter >= self.stability_frames:
                # Use median of last few detections for stability
                if len(self.temporal_buffer) >= 3:
                    # Average coordinates
                    avg_polygon = self._average_polygons(self.temporal_buffer[-3:])
                    avg_confidence = sum(r.confidence for r in self.temporal_buffer[-3:]) / 3
                    
                    stable_region = CrossingRegion(
                        polygon=avg_polygon,
                        confidence=avg_confidence,
                        orientation=current_region.orientation,
                        width=current_region.width,
                        height=current_region.height,
                        stripe_count=current_region.stripe_count,
                        center=current_region.center,
                        area=current_region.area
                    )
                    
                    self.last_stable_region = stable_region
                    return stable_region
                else:
                    self.last_stable_region = current_region
                    return current_region
            else:
                return None
        else:
            self.detection_counter = 0
            self.miss_counter += 1
            
            # Keep last stable region for a few frames
            if self.miss_counter < 5 and self.last_stable_region:
                # Reduce confidence over time
                confidence_decay = max(0.3, 1.0 - self.miss_counter * 0.1)
                self.last_stable_region.confidence *= confidence_decay
                return self.last_stable_region
            
            return None
    
    def _average_polygons(self, regions: List[CrossingRegion]) -> List[Tuple[int, int]]:
        """Average polygon coordinates from multiple regions"""
        if not regions:
            return []
        
        num_points = len(regions[0].polygon)
        avg_polygon = []
        
        for i in range(num_points):
            avg_x = sum(r.polygon[i][0] for r in regions) // len(regions)
            avg_y = sum(r.polygon[i][1] for r in regions) // len(regions)
            avg_polygon.append((avg_x, avg_y))
        
        return avg_polygon
    
    def is_stable(self) -> bool:
        """Check if current detection is stable"""
        return self.detection_counter >= self.stability_frames
    
    def reset(self):
        """Reset tracker state"""
        self.detection_counter = 0
        self.miss_counter = 0
        self.temporal_buffer = []
        self.last_stable_region = None


# For testing
if __name__ == "__main__":
    print("Enhanced Zebra Crossing Detector Test")
    print("=" * 50)
    
    # Load test image if available
    test_image_path = "test_zebra_crossing.jpg"
    
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        detector = EnhancedZebraDetector()
        result = detector.detect(frame) # pyright: ignore[reportArgumentType]
        
        if result:
            print(f"✅ Zebra crossing detected!")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Orientation: {result.orientation}")
            print(f"   Stripe count: {result.stripe_count}")
            
            annotated = detector.draw_detection(frame, result) # pyright: ignore[reportArgumentType]
            cv2.imwrite("detected_crossing.jpg", annotated)
            print("   Saved: detected_crossing.jpg")
        else:
            print("❌ No zebra crossing detected")
    else:
        print("No test image found. Run with a test image to verify detection.")
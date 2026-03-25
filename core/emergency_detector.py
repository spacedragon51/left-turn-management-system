"""
Emergency Vehicle Detector - Detects ambulances, fire trucks, police vehicles
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EmergencyVehicleDetector:
    """
    Detects emergency vehicles by visual and audio cues
    """
    
    # Emergency vehicle color ranges (HSV)
    EMERGENCY_COLORS = {
        'red': ([0, 50, 50], [10, 255, 255]),      # Red lights
        'blue': ([100, 50, 50], [130, 255, 255]),   # Blue lights
        'white': ([0, 0, 200], [180, 30, 255])      # White strobes
    }
    
    # Emergency vehicle types
    EMERGENCY_TYPES = {
        'ambulance': ['ambulance', 'medical', 'emergency'],
        'fire_truck': ['fire', 'firetruck', 'fire engine'],
        'police': ['police', 'police car', 'cop'],
        'generic': ['emergency', 'emergency vehicle']
    }
    
    def __init__(self, audio_enabled: bool = False):
        """
        Initialize emergency vehicle detector
        
        Args:
            audio_enabled: Enable audio-based detection (requires microphone)
        """
        self.audio_enabled = audio_enabled
        
        # Detection state
        self.emergency_detected = False
        self.detection_time = 0
        self.detection_type = None
        self.confidence = 0.0
        
        # For temporal consistency
        self.detection_history = []
        self.confidence_threshold = 0.6
        self.min_detections = 3
        
        # Audio detection (simplified - would use real audio processing)
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        
        if audio_enabled:
            self._start_audio_thread()
        
        logger.info("Emergency Vehicle Detector initialized")
    
    def _start_audio_thread(self):
        """Start audio detection thread"""
        self.audio_running = True
        self.audio_thread = threading.Thread(target=self._audio_detection_loop, daemon=True)
        self.audio_thread.start()
    
    def _audio_detection_loop(self):
        """Audio detection loop (simplified)"""
        # In real implementation, use libraries like pyaudio, librosa
        # For now, simulate with simple pattern detection
        while self.audio_running:
            # Simulate siren detection (for demo purposes)
            # In production, use actual audio processing
            time.sleep(0.5)
            # Placeholder for real audio processing
    
    def detect_visual(self, frame: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """
        Detect emergency vehicles by visual appearance
        
        Returns:
            Tuple of (detected, vehicle_type, confidence)
        """
        if frame is None:
            return False, None, 0.0
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        detected = False
        detected_type = None
        max_confidence = 0.0
        
        # Check for emergency colors
        for color_name, (lower, upper) in self.EMERGENCY_COLORS.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Calculate color coverage
            coverage = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
            
            if coverage > 0.01:  # At least 1% of frame
                confidence = min(1.0, coverage * 10)
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_type = color_name
                    detected = True
        
        # Also check for flashing lights (strobing effect)
        # This would require temporal analysis across frames
        
        return detected, detected_type, max_confidence
    
    def detect_audio(self) -> Tuple[bool, Optional[str], float]:
        """
        Detect siren sounds (simplified)
        
        Returns:
            Tuple of (detected, vehicle_type, confidence)
        """
        if not self.audio_enabled:
            return False, None, 0.0
        
        # In real implementation, analyze audio buffer for siren patterns
        # For now, return False (manual override for demo)
        return False, None, 0.0
    
    def update(self, frame: Optional[np.ndarray] = None) -> Dict:
        """
        Update emergency detection status
        
        Returns:
            Dictionary with detection status
        """
        current_time = time.time()
        
        # Visual detection
        visual_detected, visual_type, visual_conf = self.detect_visual(frame) if frame is not None else (False, None, 0.0)
        
        # Audio detection
        audio_detected, audio_type, audio_conf = self.detect_audio()
        
        # Combine detections
        emergency = visual_detected or audio_detected
        detected_type = visual_type if visual_detected else audio_type
        confidence = max(visual_conf, audio_conf)
        
        # Update detection history
        self.detection_history.append({
            'timestamp': current_time,
            'detected': emergency,
            'type': detected_type,
            'confidence': confidence
        })
        
        # Keep last 30 frames
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)
        
        # Check for consistent detection
        recent_detections = [d for d in self.detection_history[-10:] if d['detected']]
        consistent = len(recent_detections) >= self.min_detections
        
        if consistent and not self.emergency_detected:
            # New emergency detected
            self.emergency_detected = True
            self.detection_time = current_time
            self.detection_type = detected_type
            self.confidence = confidence
            logger.info(f"🚨 EMERGENCY VEHICLE DETECTED: {detected_type or 'unknown'} (conf: {confidence:.2f})")
        
        elif not consistent and self.emergency_detected:
            # Emergency cleared
            if current_time - self.detection_time > 10:  # Wait 10 seconds
                self.emergency_detected = False
                self.detection_type = None
                self.confidence = 0.0
                logger.info("Emergency mode cleared")
        
        return {
            'emergency_detected': self.emergency_detected,
            'emergency_type': self.detection_type,
            'confidence': self.confidence,
            'duration': current_time - self.detection_time if self.emergency_detected else 0,
            'visual_detected': visual_detected,
            'audio_detected': audio_detected
        }
    
    def manual_override(self, enable: bool = True, vehicle_type: str = "Manual") -> Dict:
        """
        Manual override for emergency mode
        
        Args:
            enable: Enable or disable emergency mode
            vehicle_type: Type of emergency vehicle
            
        Returns:
            Updated detection status
        """
        if enable and not self.emergency_detected:
            self.emergency_detected = True
            self.detection_time = time.time()
            self.detection_type = vehicle_type
            self.confidence = 1.0
            logger.info(f"🚨 MANUAL EMERGENCY OVERRIDE: {vehicle_type}")
        
        elif not enable and self.emergency_detected:
            self.emergency_detected = False
            self.detection_type = None
            self.confidence = 0.0
            logger.info("Emergency override cleared")
        
        return self.get_status()
    
    def get_status(self) -> Dict:
        """Get current detection status"""
        current_time = time.time()
        return {
            'emergency_detected': self.emergency_detected,
            'emergency_type': self.detection_type,
            'confidence': self.confidence,
            'duration': current_time - self.detection_time if self.emergency_detected else 0
        }
    
    def reset(self):
        """Reset detector state"""
        self.emergency_detected = False
        self.detection_time = 0
        self.detection_type = None
        self.confidence = 0.0
        self.detection_history = []
        logger.info("Emergency detector reset")
    
    def shutdown(self):
        """Shutdown detector"""
        self.audio_running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
        logger.info("Emergency detector shutdown")
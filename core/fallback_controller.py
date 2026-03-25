"""
Fallback Controller - Handles system failures gracefully
"""

import time
import logging
from enum import Enum
from typing import Optional, Dict, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class FallbackMode(Enum):
    """Fallback operation modes"""
    NORMAL = "normal"
    FIXED_TIME = "fixed_time"
    ALL_RED = "all_red"
    FLASHING_YELLOW = "flashing_yellow"
    EMERGENCY = "emergency"


class FallbackController:
    """
    Manages fallback operations when primary system fails
    """
    
    def __init__(self, fallback_cycle: int = 60, fixed_cycle_duration: int = 30):
        """
        Initialize fallback controller
        
        Args:
            fallback_cycle: How often to check system health (seconds)
            fixed_cycle_duration: Duration of fixed timing cycle (seconds)
        """
        self.mode = FallbackMode.NORMAL
        self.fallback_cycle = fallback_cycle
        self.fixed_cycle_duration = fixed_cycle_duration
        
        # Health monitoring
        self.camera_healthy = True
        self.processor_healthy = True
        self.network_healthy = True
        self.last_heartbeat = time.time()
        self.consecutive_failures = 0
        self.max_failures = 5
        self.failure_log: List[Dict] = []
        
        # Fixed timing schedule (phase, duration in seconds)
        self.fixed_schedule: List[Tuple[str, int]] = [
            ('free_left', 15),      # 15 seconds free left
            ('protected_left', 10),  # 10 seconds protected left
            ('all_red', 3)           # 3 seconds all red
        ]
        
        # For emergency vehicles during fallback
        self.emergency_override = False
        self.emergency_start = 0
        
        # Performance metrics
        self.performance_history = deque(maxlen=100)
        self.fallback_activations = 0
        self.last_activation_time = 0
        
        logger.info("Fallback Controller initialized")
    
    def heartbeat(self) -> None:
        """Called periodically to indicate system is alive"""
        self.last_heartbeat = time.time()
        self.consecutive_failures = 0
        
        if self.mode != FallbackMode.NORMAL:
            # Check if we can return to normal
            if self.consecutive_failures == 0:
                self._attempt_recovery()
    
    def report_failure(self, component: str, error: str) -> None:
        """
        Report a failure in the system
        
        Args:
            component: Component that failed (camera, processor, network)
            error: Error description
        """
        self.consecutive_failures += 1
        self.failure_log.append({
            'timestamp': time.time(),
            'component': component,
            'error': error,
            'consecutive_failures': self.consecutive_failures
        })
        
        logger.warning(f"Failure reported - {component}: {error} (failures: {self.consecutive_failures})")
        
        # Keep last 100 failures
        if len(self.failure_log) > 100:
            self.failure_log.pop(0)
        
        if self.consecutive_failures >= self.max_failures and self.mode == FallbackMode.NORMAL:
            self._activate_fallback()
    
    def _activate_fallback(self) -> None:
        """Activate fallback mode"""
        self.mode = FallbackMode.FIXED_TIME
        self.fallback_activations += 1
        self.last_activation_time = time.time()
        self.schedule_start = time.time()
        self.schedule_index = 0
        
        logger.warning(f"🔄 Fallback mode activated - Fixed time schedule")
        self._log_event('FALLBACK_ACTIVATED', {'mode': self.mode.value})
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover to normal mode"""
        if self.mode != FallbackMode.NORMAL:
            # Check if we've been in fallback long enough
            if time.time() - self.last_activation_time > 30:
                self.mode = FallbackMode.NORMAL
                logger.info("✅ System recovered - Returning to normal mode")
                self._log_event('FALLBACK_DEACTIVATED', {})
    
    def check_health(self, detection_available: bool, frame_rate: float) -> bool:
        """
        Check system health
        
        Args:
            detection_available: Whether detections are available
            frame_rate: Current processing frame rate
            
        Returns:
            True if system is healthy, False if fallback needed
        """
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'detection_available': detection_available,
            'frame_rate': frame_rate,
            'mode': self.mode.value
        })
        
        # Check detection availability
        if not detection_available:
            self.report_failure('detection', 'No detections available')
            return False
        
        # Check frame rate
        if frame_rate < 5:  # Too slow
            self.report_failure('processor', f'Low frame rate: {frame_rate:.1f} fps')
            return False
        
        # Check heartbeat
        if time.time() - self.last_heartbeat > 10:
            self.report_failure('heartbeat', 'No heartbeat received')
            return False
        
        # Reset on success
        self.consecutive_failures = max(0, self.consecutive_failures - 1)
        return True
    
    def get_signal_phase(self, emergency: bool = False) -> Tuple[str, bool]:
        """
        Get current signal phase based on fallback mode
        
        Args:
            emergency: Emergency vehicle detected
            
        Returns:
            Tuple of (phase_string, is_fallback_active)
        """
        # Emergency has highest priority
        if emergency:
            return "emergency_stop", True
        
        if self.mode == FallbackMode.NORMAL:
            return None, False # pyright: ignore[reportReturnType]
        
        elif self.mode == FallbackMode.FIXED_TIME:
            # Fixed time schedule
            elapsed = time.time() - self.schedule_start
            
            for phase, duration in self.fixed_schedule:
                if elapsed < duration:
                    return phase, True
                elapsed -= duration
            
            # Cycle completed, reset
            self.schedule_start = time.time()
            return self.fixed_schedule[0][0], True
        
        elif self.mode == FallbackMode.ALL_RED:
            return "all_red", True
        
        elif self.mode == FallbackMode.FLASHING_YELLOW:
            # Flashing yellow pattern (1 second on, 1 second off)
            if int(time.time() * 1) % 2 == 0:
                return "flashing_yellow", True
            else:
                return "all_red", True
        
        return "free_left", True
    
    def manual_emergency(self) -> None:
        """Manual emergency override during fallback"""
        self.emergency_override = True
        self.emergency_start = time.time()
        self._log_event('MANUAL_EMERGENCY_FALLBACK', {})
        logger.info("🚨 Manual emergency override during fallback")
    
    def clear_emergency(self) -> None:
        """Clear manual emergency override"""
        self.emergency_override = False
        self._log_event('EMERGENCY_CLEARED', {})
        logger.info("Emergency override cleared")
    
    def reset(self) -> None:
        """Reset to normal mode"""
        self.mode = FallbackMode.NORMAL
        self.consecutive_failures = 0
        self.emergency_override = False
        logger.info("Fallback controller reset to normal mode")
        self._log_event('FALLBACK_RESET', {})
    
    def get_status(self) -> Dict:
        """Get current fallback status"""
        current_time = time.time()
        
        return {
            'mode': self.mode.value,
            'is_active': self.mode != FallbackMode.NORMAL,
            'consecutive_failures': self.consecutive_failures,
            'fallback_activations': self.fallback_activations,
            'last_activation': self.last_activation_time,
            'time_in_fallback': current_time - self.last_activation_time if self.mode != FallbackMode.NORMAL else 0,
            'emergency_override': self.emergency_override,
            'health_metrics': {
                'camera': self.camera_healthy,
                'processor': self.processor_healthy,
                'network': self.network_healthy,
                'heartbeat_age': current_time - self.last_heartbeat
            }
        }
    
    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log fallback event"""
        self.failure_log.append({
            'timestamp': time.time(),
            'event': event_type,
            'data': data
        })
        if len(self.failure_log) > 200:
            self.failure_log.pop(0)
    
    def get_failure_log(self, limit: int = 50) -> List[Dict]:
        """Get recent failure log entries"""
        return self.failure_log[-limit:]
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.performance_history:
            return {'avg_frame_rate': 0, 'stability': 0}
        
        recent = list(self.performance_history)[-50:]
        frame_rates = [p['frame_rate'] for p in recent if p['frame_rate'] > 0]
        
        return {
            'avg_frame_rate': sum(frame_rates) / len(frame_rates) if frame_rates else 0,
            'min_frame_rate': min(frame_rates) if frame_rates else 0,
            'max_frame_rate': max(frame_rates) if frame_rates else 0,
            'samples': len(recent),
            'fallback_time_percentage': len([p for p in recent if p['mode'] != 'normal']) / len(recent) * 100
        }
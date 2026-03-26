"""
Signal Controller - Simplified Version
Removes complex constraints like weighted occupancy, dynamic timing, emergency priority
"""

import time
import logging
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SignalPhase(Enum):
    """Signal phase states"""
    FREE_LEFT = "🟡 FREE LEFT (Yield to oncoming)"
    PROTECTED_LEFT = "🟢 PROTECTED LEFT (Green Arrow)"
    PEDESTRIAN_CROSSING = "🚶 PEDESTRIAN CROSSING (All Red)"


class EnhancedSignalController:
    """
    Simplified signal controller - Basic vehicle and pedestrian detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize controller with basic thresholds"""
        
        self.config = {
            'violation_threshold': 3,          # Number of vehicles to trigger
            'blocking_duration_threshold': 5,   # Seconds before considering blocking
            'min_protected_duration': 10,       # Minimum protected left duration
            'max_free_left_duration': 30,       # Max free-left before forced protection
            'cooldown_after_protected': 15,     # Cooldown after returning to free
            'pedestrian_wait_threshold': 5,     # Seconds before pedestrian mode
            'pedestrian_crossing_duration': 8   # Pedestrian crossing duration
        }
        
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self.blocking_vehicles: Dict[str, Dict] = {}
        self.cooldown_until = 0
        self.manual_override = False
        
        # Pedestrian tracking
        self.pedestrian_mode_triggered = False
        self.pedestrian_mode_start = 0
        self.pedestrians_waiting_count = 0
        self.pedestrians_waiting_start = 0
        
        # Statistics
        self.protected_triggers = 0
        self.pedestrian_triggers = 0
        self.event_log: List[Dict] = []
        
        logger.info("Signal Controller initialized")
    
    def update_detections(self, lane_vehicles: List, pedestrians: List) -> SignalPhase:
        """Update with real-time detections"""
        if self.manual_override:
            return self.current_phase
        
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.cooldown_until and self.current_phase != SignalPhase.PEDESTRIAN_CROSSING:
            return SignalPhase.FREE_LEFT
        
        # Update blocking vehicles
        self._update_blocking_tracking(lane_vehicles, current_time)
        
        # Count blocking vehicles
        persistent_blocks = sum(1 for v in self.blocking_vehicles.values() 
                               if v.get('duration', 0) > self.config['blocking_duration_threshold'])
        total_blocking = len(self.blocking_vehicles)
        
        # ========== PEDESTRIAN MODE ==========
        if pedestrians and not self.pedestrian_mode_triggered:
            if self.pedestrians_waiting_count == 0:
                self.pedestrians_waiting_start = current_time
            self.pedestrians_waiting_count = len(pedestrians)
            
            waiting_time = current_time - self.pedestrians_waiting_start
            
            # Trigger pedestrian mode if 2+ pedestrians or waiting time exceeds threshold
            if self.pedestrians_waiting_count >= 2 or waiting_time > self.config['pedestrian_wait_threshold']:
                self.pedestrian_mode_triggered = True
                self.pedestrian_mode_start = current_time
                self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
                self.phase_start_time = current_time
                self.pedestrian_triggers += 1
                self._log_event('PEDESTRIAN_MODE', {
                    'count': self.pedestrians_waiting_count,
                    'waiting_time': waiting_time
                })
                logger.info(f"Pedestrian mode activated - {self.pedestrians_waiting_count} waiting")
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # Existing pedestrian mode
        if self.pedestrian_mode_triggered:
            if current_time - self.pedestrian_mode_start > self.config['pedestrian_crossing_duration']:
                self.pedestrian_mode_triggered = False
                self.pedestrians_waiting_count = 0
                self.current_phase = SignalPhase.FREE_LEFT
                self.phase_start_time = current_time
                self._log_event('PEDESTRIAN_MODE_END', {})
                logger.info("Pedestrian mode ended")
            else:
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # ========== VEHICLE DECISION ==========
        if self.current_phase == SignalPhase.FREE_LEFT:
            # Conditions to trigger protected left
            should_protect = (
                total_blocking >= self.config['violation_threshold'] or
                persistent_blocks >= 2 or
                (current_time - self.phase_start_time) > self.config['max_free_left_duration']
            )
            
            if should_protect:
                self.current_phase = SignalPhase.PROTECTED_LEFT
                self.phase_start_time = current_time
                self.protected_triggers += 1
                self._log_event('PROTECTED_TRIGGERED', {
                    'total_blocking': total_blocking,
                    'persistent': persistent_blocks
                })
                logger.info(f"Protected left triggered - {total_blocking} blocking vehicles")
        
        elif self.current_phase == SignalPhase.PROTECTED_LEFT:
            phase_duration = current_time - self.phase_start_time
            
            if phase_duration >= self.config['min_protected_duration']:
                if total_blocking == 0:
                    self.current_phase = SignalPhase.FREE_LEFT
                    self.phase_start_time = current_time
                    self.cooldown_until = current_time + self.config['cooldown_after_protected']
                    self._log_event('RETURNED_TO_FREE', {'duration': phase_duration})
                    logger.info(f"Returned to free left - {phase_duration:.1f}s")
        
        return self.current_phase
    
    def _update_blocking_tracking(self, lane_vehicles: List, current_time: float) -> None:
        """Update tracking of blocking vehicles"""
        current_ids = {v.id for v in lane_vehicles if hasattr(v, 'id')}
        
        for vid in current_ids:
            if vid not in self.blocking_vehicles:
                for v in lane_vehicles:
                    if v.id == vid:
                        self.blocking_vehicles[vid] = {
                            'first_seen': current_time,
                            'duration': 0,
                            'vehicle': v
                        }
                        break
            else:
                self.blocking_vehicles[vid]['duration'] = current_time - self.blocking_vehicles[vid]['first_seen']
        
        for vid in list(self.blocking_vehicles.keys()):
            if vid not in current_ids:
                del self.blocking_vehicles[vid]
    
    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log event"""
        self.event_log.append({
            'timestamp': time.time(),
            'event': event_type,
            'data': data
        })
        if len(self.event_log) > 500:
            self.event_log = self.event_log[-500:]
    
    def get_state(self) -> Dict:
        """Get current signal state"""
        current_time = time.time()
        
        return {
            'phase': self.current_phase.value,
            'phase_duration': current_time - self.phase_start_time,
            'blocking_vehicles': len(self.blocking_vehicles),
            'cooldown_remaining': max(0, self.cooldown_until - current_time),
            'pedestrian_mode_active': self.pedestrian_mode_triggered,
            'pedestrians_waiting': self.pedestrians_waiting_count,
            'protected_triggers': self.protected_triggers,
            'pedestrian_triggers': self.pedestrian_triggers
        }
    
    def manual_protect(self) -> None:
        """Manual override to trigger protected left"""
        self.manual_override = True
        self.current_phase = SignalPhase.PROTECTED_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PROTECT', {})
        logger.info("Manual override - Protected left activated")
    
    def manual_pedestrian(self) -> None:
        """Manual pedestrian crossing override"""
        self.manual_override = True
        self.pedestrian_mode_triggered = True
        self.pedestrian_mode_start = time.time()
        self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PEDESTRIAN', {})
        logger.info("Manual override - Pedestrian crossing activated")
    
    def manual_reset(self) -> None:
        """Reset to automatic mode"""
        self.manual_override = False
        self.pedestrian_mode_triggered = False
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_RESET', {})
        logger.info("Manual reset - Automatic mode restored")
    
    def integrate_dataset(self, insights: Dict) -> None:
        """Integrate dataset insights to adjust thresholds"""
        risk = insights.get('risk_assessment', {})
        risk_level = risk.get('level', 'MEDIUM')
        
        if risk_level == 'HIGH':
            self.config['violation_threshold'] = 2
            self.config['blocking_duration_threshold'] = 3
            self.config['max_free_left_duration'] = 20
            logger.info("HIGH RISK - Aggressive thresholds applied")
        elif risk_level == 'MEDIUM':
            self.config['violation_threshold'] = 3
            self.config['blocking_duration_threshold'] = 5
            self.config['max_free_left_duration'] = 30
            logger.info("MEDIUM RISK - Standard thresholds applied")
        else:
            self.config['violation_threshold'] = 4
            self.config['blocking_duration_threshold'] = 8
            self.config['max_free_left_duration'] = 45
            logger.info("LOW RISK - Relaxed thresholds applied")
        
        self._log_event('DATASET_INTEGRATED', {'risk_level': risk_level})
    
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return self.event_log[-limit:]
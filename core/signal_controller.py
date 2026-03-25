"""
Enhanced Signal Controller with Pedestrian Priority
Built from scratch with Indian traffic constraints
"""

import time
import json
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalPhase(Enum):
    """Signal phase states"""
    FREE_LEFT = "🟡 FREE LEFT (Yield to oncoming)"
    PROTECTED_LEFT = "🟢 PROTECTED LEFT (Green Arrow)"
    PEDESTRIAN_CROSSING = "🚶 PEDESTRIAN CROSSING (All Red)"
    EMERGENCY_STOP = "🔴 EMERGENCY STOP (All Red)"
    MAIN_GREEN = "🟢 MAIN GREEN (Through traffic)"


class ConflictPredictor:
    """Predicts potential conflicts between vehicles and pedestrians"""
    
    def __init__(self):
        self.conflict_history = []
    
    def predict(self, turning_vehicles: List, pedestrians: List) -> Dict:
        """
        Predict conflict between turning vehicles and pedestrians
        
        Returns:
            Dictionary with conflict prediction
        """
        if not turning_vehicles or not pedestrians:
            return {'has_conflict': False, 'conflicts': [], 'risk': 'LOW'}
        
        conflicts = []
        for vehicle in turning_vehicles:
            for pedestrian in pedestrians:
                # Simple distance-based conflict detection
                v_center = vehicle.center
                p_center = pedestrian.center
                distance = ((v_center[0] - p_center[0])**2 + (v_center[1] - p_center[1])**2) ** 0.5
                
                if distance < 100:  # Within 100 pixels
                    conflicts.append({
                        'vehicle_id': vehicle.id,
                        'pedestrian_id': pedestrian.id,
                        'distance': distance,
                        'severity': 'HIGH' if distance < 50 else 'MEDIUM'
                    })
        
        risk = 'HIGH' if any(c['severity'] == 'HIGH' for c in conflicts) else 'MEDIUM' if conflicts else 'LOW'
        
        return {
            'has_conflict': len(conflicts) > 0,
            'conflicts': conflicts,
            'risk': risk
        }


class EnhancedSignalController:
    """
    Enhanced signal controller with pedestrian priority and multi-camera input
    Built from scratch for Indian traffic conditions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize controller with configuration"""
        self.config = config or {
            'violation_threshold': 5,           # Number of blocking vehicles to trigger
            'blocking_duration_threshold': 10,   # Seconds before persistent blocking
            'min_protected_duration': 15,        # Minimum protected left duration
            'max_free_left_duration': 120,       # Max free-left before forced protection
            'cooldown_after_protected': 60,      # Seconds before another intervention
            'pedestrian_wait_threshold': 10,     # Seconds before pedestrian mode
            'pedestrian_crossing_duration': 15,  # Seconds for pedestrian crossing
            'dynamic_thresholds': True
        }
        
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self.blocking_vehicles: Dict[str, Dict] = {}
        self.cooldown_until = 0
        self.manual_override = False
        self.dataset_insights = None
        self.peak_hours = [8, 9, 17, 18]  # Default peak hours
        self.event_log: List[Dict] = []
        
        # Conflict predictor
        self.conflict_predictor = ConflictPredictor()
        
        # Pedestrian mode tracking
        self.pedestrian_mode_triggered = False
        self.pedestrian_mode_start = 0
        self.pedestrians_waiting_count = 0
        
        logger.info("Enhanced Signal Controller initialized")
    
    def integrate_dataset(self, insights: Dict) -> None:
        """Integrate dataset insights to adjust thresholds"""
        risk = insights.get('risk_assessment', {})
        risk_level = risk.get('level', 'MEDIUM')
        violations = insights.get('violation_analysis', {})
        
        # Adjust thresholds based on risk level
        if risk_level == 'HIGH':
            self.config['violation_threshold'] = 3
            self.config['blocking_duration_threshold'] = 5
            self.config['max_free_left_duration'] = 60
            self.config['cooldown_after_protected'] = 45
        elif risk_level == 'MEDIUM':
            self.config['violation_threshold'] = 5
            self.config['blocking_duration_threshold'] = 8
            self.config['max_free_left_duration'] = 90
        else:  # LOW
            self.config['violation_threshold'] = 7
            self.config['blocking_duration_threshold'] = 12
            self.config['max_free_left_duration'] = 150
        
        # Extract peak hours
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            self.peak_hours = [int(h) for h in peak_hours.keys()]
        
        self.dataset_insights = insights
        self._log_event('DATASET_INTEGRATED', {'risk_level': risk_level})
        
        logger.info(f"Dataset integrated - Risk: {risk_level}, Threshold: {self.config['violation_threshold']}")
    
    def update_detections(self, 
                          lane_vehicles: List,
                          pedestrians: List,
                          oncoming_vehicles: List = None) -> SignalPhase: # pyright: ignore[reportArgumentType]
        """
        Update with real-time detections and determine next phase
        
        Args:
            lane_vehicles: Vehicles detected in free-left lane
            pedestrians: Pedestrians detected
            oncoming_vehicles: Oncoming traffic (optional)
            
        Returns:
            Current signal phase
        """
        if self.manual_override:
            return self.current_phase
        
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.cooldown_until and self.current_phase != SignalPhase.PEDESTRIAN_CROSSING:
            return SignalPhase.FREE_LEFT
        
        # Update blocking vehicles tracking
        self._update_blocking_tracking(lane_vehicles, current_time)
        
        # Count persistent blocks
        persistent_blocks = sum(1 for v in self.blocking_vehicles.values() 
                               if v.get('duration', 0) > self.config['blocking_duration_threshold'])
        
        # Check for pedestrian conflict (HIGHEST PRIORITY)
        conflict = self.conflict_predictor.predict(lane_vehicles, pedestrians)
        
        if conflict['has_conflict'] and conflict['risk'] == 'HIGH':
            self.current_phase = SignalPhase.EMERGENCY_STOP
            self.phase_start_time = current_time
            self._log_event('EMERGENCY_STOP', {'conflicts': len(conflict['conflicts'])})
            logger.warning(f"EMERGENCY STOP triggered! {len(conflict['conflicts'])} conflicts detected")
            return SignalPhase.EMERGENCY_STOP
        
        # Check for pedestrian crossing request
        if pedestrians and not self.pedestrian_mode_triggered:
            self.pedestrians_waiting_count = len(pedestrians)
            
            # Check if pedestrians have been waiting
            if self.pedestrians_waiting_count >= 2 or self._pedestrians_waiting_time() > self.config['pedestrian_wait_threshold']:
                self.pedestrian_mode_triggered = True
                self.pedestrian_mode_start = current_time
                self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
                self.phase_start_time = current_time
                self._log_event('PEDESTRIAN_MODE', {'count': self.pedestrians_waiting_count})
                logger.info(f"Pedestrian mode activated - {self.pedestrians_waiting_count} waiting")
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # Check if pedestrian mode should end
        if self.pedestrian_mode_triggered:
            if current_time - self.pedestrian_mode_start > self.config['pedestrian_crossing_duration']:
                self.pedestrian_mode_triggered = False
                self.pedestrians_waiting_count = 0
                self.current_phase = SignalPhase.FREE_LEFT
                self.phase_start_time = current_time
                self._log_event('PEDESTRIAN_MODE_END', {})
            else:
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # Vehicle-based decision logic
        if self.current_phase == SignalPhase.FREE_LEFT:
            # Conditions to trigger protected left
            should_protect = (
                len(lane_vehicles) >= self.config['violation_threshold'] or
                persistent_blocks >= 2 or
                (current_time - self.phase_start_time) > self.config['max_free_left_duration']
            )
            
            # More sensitive during peak hours
            if self.is_peak_hour() and len(lane_vehicles) >= max(1, self.config['violation_threshold'] - 1):
                should_protect = True
            
            # Oncoming traffic density check
            if oncoming_vehicles and len(oncoming_vehicles) > 10:
                should_protect = True
            
            if should_protect:
                self.current_phase = SignalPhase.PROTECTED_LEFT
                self.phase_start_time = current_time
                self._log_event('PROTECTED_TRIGGERED', {
                    'violations': len(lane_vehicles),
                    'persistent': persistent_blocks
                })
                logger.info(f"Protected left triggered - {len(lane_vehicles)} violations")
        
        elif self.current_phase == SignalPhase.PROTECTED_LEFT:
            # Check if protected phase should end
            phase_duration = current_time - self.phase_start_time
            
            if phase_duration >= self.config['min_protected_duration']:
                if len(lane_vehicles) == 0 and len(self.blocking_vehicles) == 0:
                    self.current_phase = SignalPhase.FREE_LEFT
                    self.phase_start_time = current_time
                    self.cooldown_until = current_time + self.config['cooldown_after_protected']
                    self._log_event('RETURNED_TO_FREE', {'duration': phase_duration})
                    logger.info("Returned to free-left mode")
        
        return self.current_phase
    
    def _update_blocking_tracking(self, lane_vehicles: List, current_time: float) -> None:
        """Update tracking of blocking vehicles"""
        current_ids = {v.id for v in lane_vehicles}
        
        # Update existing
        for vid in current_ids:
            if vid not in self.blocking_vehicles:
                # Find vehicle to get first_seen
                for v in lane_vehicles:
                    if v.id == vid:
                        self.blocking_vehicles[vid] = {
                            'first_seen': current_time,
                            'duration': 0,
                            'vehicle': v
                        }
                        break
            else:
                # Update duration
                self.blocking_vehicles[vid]['duration'] = current_time - self.blocking_vehicles[vid]['first_seen']
        
        # Remove resolved
        for vid in list(self.blocking_vehicles.keys()):
            if vid not in current_ids:
                del self.blocking_vehicles[vid]
    
    def _pedestrians_waiting_time(self) -> float:
        """Get how long pedestrians have been waiting"""
        if not self.pedestrians_waiting_count:
            return 0
        # Simplified - would need actual tracking
        return 5  # Placeholder
    
    def is_peak_hour(self) -> bool:
        """Check if current time is within peak hours"""
        current_hour = datetime.now().hour
        return current_hour in self.peak_hours
    
    def get_state(self) -> Dict:
        """Get current signal state"""
        current_time = time.time()
        return {
            'phase': self.current_phase.value,
            'phase_duration': current_time - self.phase_start_time,
            'violations_detected': len(self.blocking_vehicles),
            'blocking_vehicles': len(self.blocking_vehicles),
            'cooldown_remaining': max(0, self.cooldown_until - current_time),
            'is_peak_hour': self.is_peak_hour(),
            'risk_level': self.dataset_insights.get('risk_assessment', {}).get('level', 'MEDIUM') if self.dataset_insights else 'UNKNOWN',
            'pedestrian_mode_active': self.pedestrian_mode_triggered,
            'pedestrians_waiting': self.pedestrians_waiting_count
        }
    
    def manual_protect(self) -> None:
        """Manual override to trigger protected left"""
        self.manual_override = True
        self.current_phase = SignalPhase.PROTECTED_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PROTECT', {})
        logger.info("Manual override - Protected left activated")
    
    def manual_pedestrian(self) -> None:
        """Manual override to trigger pedestrian crossing"""
        self.manual_override = True
        self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PEDESTRIAN', {})
        logger.info("Manual override - Pedestrian crossing activated")
    
    def manual_reset(self) -> None:
        """Manual reset to automatic mode"""
        self.manual_override = False
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_RESET', {})
        logger.info("Manual reset - Automatic mode restored")
    
    def _log_event(self, event_type: str, data: Dict) -> None:
        """Log event for audit trail"""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'data': data
        })
        
        # Keep last 1000 events
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return self.event_log[-limit:]
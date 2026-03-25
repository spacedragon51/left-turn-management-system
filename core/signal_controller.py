"""
Enhanced Signal Controller - FIXED Dynamic Signaling
Properly triggers protected left when conditions are met
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


class ConflictPredictor:
    """Predicts potential conflicts between vehicles and pedestrians"""
    
    def __init__(self, distance_threshold: int = 80):
        self.distance_threshold = distance_threshold
    
    def predict(self, turning_vehicles: List, pedestrians: List) -> Dict:
        """Predict conflict between turning vehicles and pedestrians"""
        if not turning_vehicles or not pedestrians:
            return {'has_conflict': False, 'conflicts': [], 'risk': 'LOW'}
        
        conflicts = []
        for vehicle in turning_vehicles:
            for pedestrian in pedestrians:
                if hasattr(vehicle, 'center') and hasattr(pedestrian, 'center'):
                    v_center = vehicle.center
                    p_center = pedestrian.center
                    distance = ((v_center[0] - p_center[0])**2 + (v_center[1] - p_center[1])**2) ** 0.5
                    
                    if distance < self.distance_threshold:
                        conflicts.append({
                            'vehicle_id': vehicle.id,
                            'pedestrian_id': pedestrian.id,
                            'distance': distance,
                            'severity': 'HIGH' if distance < 40 else 'MEDIUM'
                        })
        
        risk = 'HIGH' if any(c['severity'] == 'HIGH' for c in conflicts) else 'MEDIUM' if conflicts else 'LOW'
        
        return {
            'has_conflict': len(conflicts) > 0,
            'conflicts': conflicts,
            'risk': risk
        }


class EnhancedSignalController:
    """
    Enhanced signal controller with PROPER dynamic signaling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize controller with optimized thresholds"""
        
        # OPTIMIZED CONFIGURATION for dynamic signaling
        self.config = {
            # Vehicle blocking thresholds
            'violation_threshold': 2,              # TRIGGER after 2 vehicles blocking (was 5)
            'blocking_duration_threshold': 5,       # TRIGGER after 5 seconds (was 10)
            'min_protected_duration': 10,           # Minimum protected left duration
            'max_free_left_duration': 30,           # Force protected left after 30 seconds (was 120)
            'cooldown_after_protected': 15,          # Cooldown after returning to free (was 60)
            
            # Pedestrian thresholds
            'pedestrian_wait_threshold': 5,          # Trigger after 5 seconds waiting
            'pedestrian_crossing_duration': 10,      # Pedestrian crossing duration
            
            # Dynamic adjustment
            'dynamic_thresholds': True,
            'peak_hour_multiplier': 0.7              # More sensitive during peak hours
        }
        
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self.blocking_vehicles: Dict[str, Dict] = {}
        self.cooldown_until = 0
        self.manual_override = False
        self.dataset_insights = None
        self.peak_hours = [8, 9, 17, 18]  # 8-9 AM, 5-6 PM
        self.event_log: List[Dict] = []
        
        # Conflict predictor
        self.conflict_predictor = ConflictPredictor()
        
        # Pedestrian mode tracking
        self.pedestrian_mode_triggered = False
        self.pedestrian_mode_start = 0
        self.pedestrians_waiting_count = 0
        self.pedestrians_waiting_start = 0
        
        # Statistics
        self.protected_triggers = 0
        self.pedestrian_triggers = 0
        
        logger.info("Enhanced Signal Controller initialized with OPTIMIZED thresholds")
        logger.info(f"  Violation threshold: {self.config['violation_threshold']}")
        logger.info(f"  Blocking duration: {self.config['blocking_duration_threshold']}s")
        logger.info(f"  Max free-left: {self.config['max_free_left_duration']}s")
    
    def integrate_dataset(self, insights: Dict) -> None:
        """Integrate dataset insights to adjust thresholds"""
        risk = insights.get('risk_assessment', {})
        risk_level = risk.get('level', 'MEDIUM')
        
        # Adjust thresholds based on risk level (more aggressive)
        if risk_level == 'HIGH':
            self.config['violation_threshold'] = 1     # Trigger on 1 vehicle
            self.config['blocking_duration_threshold'] = 3
            self.config['max_free_left_duration'] = 20
            self.config['cooldown_after_protected'] = 10
            logger.info("HIGH RISK - Aggressive thresholds applied")
            
        elif risk_level == 'MEDIUM':
            self.config['violation_threshold'] = 2
            self.config['blocking_duration_threshold'] = 5
            self.config['max_free_left_duration'] = 30
            self.config['cooldown_after_protected'] = 15
            logger.info("MEDIUM RISK - Standard thresholds applied")
            
        else:  # LOW
            self.config['violation_threshold'] = 3
            self.config['blocking_duration_threshold'] = 8
            self.config['max_free_left_duration'] = 45
            self.config['cooldown_after_protected'] = 20
            logger.info("LOW RISK - Relaxed thresholds applied")
        
        # Extract peak hours
        peak_hours = insights.get('violation_analysis', {}).get('peak_hours', {})
        if peak_hours:
            self.peak_hours = [int(h) for h in peak_hours.keys()]
        
        self.dataset_insights = insights
        self._log_event('DATASET_INTEGRATED', {'risk_level': risk_level})
    
    def is_peak_hour(self) -> bool:
        """Check if current time is within peak hours"""
        current_hour = datetime.now().hour
        return current_hour in self.peak_hours
    
    def get_current_threshold(self) -> int:
        """Get current violation threshold (adjusted for peak hours)"""
        threshold = self.config['violation_threshold']
        if self.is_peak_hour() and self.config['dynamic_thresholds']:
            # More sensitive during peak hours
            adjusted = max(1, int(threshold * self.config['peak_hour_multiplier']))
            return adjusted
        return threshold
    
    def update_detections(self, lane_vehicles: List, pedestrians: List, oncoming_vehicles: List = None) -> SignalPhase: # pyright: ignore[reportArgumentType]
        """
        Update with real-time detections and determine next phase
        """
        if self.manual_override:
            return self.current_phase
        
        current_time = time.time()
        
        # Check cooldown - prevent rapid switching
        if current_time < self.cooldown_until and self.current_phase != SignalPhase.PEDESTRIAN_CROSSING:
            # Still in cooldown, remain in FREE LEFT
            return SignalPhase.FREE_LEFT
        
        # Update blocking vehicles tracking
        self._update_blocking_tracking(lane_vehicles, current_time)
        
        # Calculate statistics
        active_blocking = len([v for v in self.blocking_vehicles.values() 
                               if v.get('duration', 0) < self.config['blocking_duration_threshold']])
        persistent_blocks = len([v for v in self.blocking_vehicles.values() 
                                 if v.get('duration', 0) >= self.config['blocking_duration_threshold']])
        total_blocking = len(self.blocking_vehicles)
        
        # LOGGING for debugging
        if total_blocking > 0:
            logger.debug(f"Blocking: {total_blocking} total, {persistent_blocks} persistent, "
                        f"threshold: {self.get_current_threshold()}")
        
        # ========== PRIORITY 1: EMERGENCY CONFLICT ==========
        conflict = self.conflict_predictor.predict(lane_vehicles, pedestrians)
        if conflict['has_conflict'] and conflict['risk'] == 'HIGH':
            self.current_phase = SignalPhase.EMERGENCY_STOP
            self.phase_start_time = current_time
            self._log_event('EMERGENCY_STOP', {'conflicts': len(conflict['conflicts'])})
            logger.warning(f"🚨 EMERGENCY STOP! {len(conflict['conflicts'])} conflicts")
            return SignalPhase.EMERGENCY_STOP
        
        # ========== PRIORITY 2: PEDESTRIAN MODE ==========
        if pedestrians and not self.pedestrian_mode_triggered:
            # Track pedestrian waiting time
            if self.pedestrians_waiting_count == 0:
                self.pedestrians_waiting_start = current_time
            self.pedestrians_waiting_count = len(pedestrians)
            
            waiting_time = current_time - self.pedestrians_waiting_start
            
            # Trigger pedestrian mode if:
            # - 2+ pedestrians waiting OR
            # - Pedestrians waiting > threshold OR
            # - Vehicles are not blocking (to be efficient)
            if (self.pedestrians_waiting_count >= 2 or 
                waiting_time > self.config['pedestrian_wait_threshold']):
                
                self.pedestrian_mode_triggered = True
                self.pedestrian_mode_start = current_time
                self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
                self.phase_start_time = current_time
                self.pedestrian_triggers += 1
                self._log_event('PEDESTRIAN_MODE', {
                    'count': self.pedestrians_waiting_count,
                    'waiting_time': waiting_time
                })
                logger.info(f"🚶 PEDESTRIAN MODE activated - {self.pedestrians_waiting_count} waiting")
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # ========== PRIORITY 3: EXISTING PEDESTRIAN MODE ==========
        if self.pedestrian_mode_triggered:
            if current_time - self.pedestrian_mode_start > self.config['pedestrian_crossing_duration']:
                # End pedestrian mode
                self.pedestrian_mode_triggered = False
                self.pedestrians_waiting_count = 0
                self.current_phase = SignalPhase.FREE_LEFT
                self.phase_start_time = current_time
                self._log_event('PEDESTRIAN_MODE_END', {})
                logger.info("Pedestrian mode ended")
            else:
                return SignalPhase.PEDESTRIAN_CROSSING
        
        # ========== PRIORITY 4: VEHICLE-BASED DECISION ==========
        current_threshold = self.get_current_threshold()
        
        # Conditions to trigger PROTECTED LEFT
        should_protect = False
        trigger_reason = ""
        
        # Condition 1: Too many vehicles blocking
        if total_blocking >= current_threshold:
            should_protect = True
            trigger_reason = f"{total_blocking} vehicles blocking (threshold: {current_threshold})"
        
        # Condition 2: Persistent blocking (long duration)
        elif persistent_blocks >= 1:  # Even 1 persistent block triggers
            should_protect = True
            trigger_reason = f"{persistent_blocks} vehicle(s) blocking for >{self.config['blocking_duration_threshold']}s"
        
        # Condition 3: Free-left duration exceeded
        elif (current_time - self.phase_start_time) > self.config['max_free_left_duration']:
            should_protect = True
            trigger_reason = f"Free-left duration exceeded ({self.config['max_free_left_duration']}s)"
        
        # Condition 4: Peak hour with any blocking
        elif self.is_peak_hour() and total_blocking >= 1:
            should_protect = True
            trigger_reason = f"Peak hour with {total_blocking} blocking vehicle(s)"
        
        # Execute protection if needed
        if self.current_phase == SignalPhase.FREE_LEFT:
            if should_protect:
                self.current_phase = SignalPhase.PROTECTED_LEFT
                self.phase_start_time = current_time
                self.protected_triggers += 1
                self._log_event('PROTECTED_TRIGGERED', {
                    'total_blocking': total_blocking,
                    'persistent': persistent_blocks,
                    'reason': trigger_reason,
                    'threshold': current_threshold
                })
                logger.info(f"🟢 PROTECTED LEFT TRIGGERED - {trigger_reason}")
                
        # Handle existing PROTECTED phase
        elif self.current_phase == SignalPhase.PROTECTED_LEFT:
            phase_duration = current_time - self.phase_start_time
            
            # Minimum protected duration
            if phase_duration >= self.config['min_protected_duration']:
                # Check if lane is clear
                if total_blocking == 0:
                    # Return to FREE LEFT with cooldown
                    self.current_phase = SignalPhase.FREE_LEFT
                    self.phase_start_time = current_time
                    self.cooldown_until = current_time + self.config['cooldown_after_protected']
                    self._log_event('RETURNED_TO_FREE', {
                        'duration': phase_duration,
                        'vehicles_cleared': self.protected_triggers
                    })
                    logger.info(f"🟡 RETURNED TO FREE LEFT - {phase_duration:.1f}s protected, {self.protected_triggers} triggers total")
                else:
                    # Still blocking, continue protection
                    if self.debug_mode:
                        logger.debug(f"Still blocking: {total_blocking} vehicles, continuing protection")
        
        return self.current_phase
    
    def _update_blocking_tracking(self, lane_vehicles: List, current_time: float) -> None:
        """Update tracking of blocking vehicles"""
        current_ids = {v.id for v in lane_vehicles}
        
        # Update existing vehicles
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
        
        # Remove resolved vehicles
        for vid in list(self.blocking_vehicles.keys()):
            if vid not in current_ids:
                del self.blocking_vehicles[vid]
    
    def get_state(self) -> Dict:
        """Get current signal state"""
        current_time = time.time()
        
        # Calculate pedestrian waiting time
        pedestrian_waiting_time = 0
        if self.pedestrians_waiting_count > 0 and not self.pedestrian_mode_triggered:
            pedestrian_waiting_time = current_time - self.pedestrians_waiting_start if hasattr(self, 'pedestrians_waiting_start') else 0
        
        return {
            'phase': self.current_phase.value,
            'phase_duration': current_time - self.phase_start_time,
            'violations_detected': len(self.blocking_vehicles),
            'blocking_vehicles': len(self.blocking_vehicles),
            'persistent_blocks': len([v for v in self.blocking_vehicles.values() 
                                      if v.get('duration', 0) >= self.config['blocking_duration_threshold']]),
            'cooldown_remaining': max(0, self.cooldown_until - current_time),
            'is_peak_hour': self.is_peak_hour(),
            'current_threshold': self.get_current_threshold(),
            'risk_level': self.dataset_insights.get('risk_assessment', {}).get('level', 'MEDIUM') if self.dataset_insights else 'UNKNOWN',
            'pedestrian_mode_active': self.pedestrian_mode_triggered,
            'pedestrians_waiting': self.pedestrians_waiting_count,
            'pedestrian_waiting_time': pedestrian_waiting_time,
            'protected_triggers': self.protected_triggers,
            'pedestrian_triggers': self.pedestrian_triggers
        }
    
    def manual_protect(self) -> None:
        """Manual override to trigger protected left"""
        self.manual_override = True
        self.current_phase = SignalPhase.PROTECTED_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PROTECT', {})
        logger.info("🔒 Manual override - Protected left activated")
    
    def manual_pedestrian(self) -> None:
        """Manual override to trigger pedestrian crossing"""
        self.manual_override = True
        self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PEDESTRIAN', {})
        logger.info("🚶 Manual override - Pedestrian crossing activated")
    
    def manual_reset(self) -> None:
        """Manual reset to automatic mode"""
        self.manual_override = False
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_RESET', {})
        logger.info("🔄 Manual reset - Automatic mode restored")
    
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
    
    @property
    def debug_mode(self):
        """Debug mode property"""
        return False
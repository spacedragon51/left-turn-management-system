"""
Enhanced Signal Controller - COMPLETE with Weighted Occupancy, Dynamic Timing, Emergency Priority, and Fallback
"""

import time
import logging
from collections import deque
from typing import List, Dict, Optional, Any
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class SignalPhase(Enum):
    """Signal phase states"""
    FREE_LEFT = "🟡 FREE LEFT (Yield to oncoming)"
    PROTECTED_LEFT = "🟢 PROTECTED LEFT (Green Arrow)"
    PEDESTRIAN_CROSSING = "🚶 PEDESTRIAN CROSSING (All Red)"
    EMERGENCY_STOP = "🔴 EMERGENCY STOP (All Red)"
    ALL_RED = "🔴 ALL RED"
    FLASHING_YELLOW = "⚠️ FLASHING YELLOW (Proceed with caution)"


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
    Complete signal controller with all constraints implemented:
    - Weighted occupancy detection
    - Dynamic green timing with sliding window
    - Emergency vehicle priority
    - Spillover management
    - Pedestrian safety
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize controller with optimized thresholds"""
        
        # Complete configuration with all parameters
        self.config = {
            # Weighted occupancy thresholds
            'weighted_threshold': 2.5,        # Trigger protected left when weighted occupancy exceeds this
            'blocking_duration_threshold': 5,  # Seconds before considering persistent
            
            # Green timing
            'min_protected_duration': 10,      # Minimum protected left duration
            'max_protected_duration': 20,      # Maximum protected left duration
            'max_free_left_duration': 30,      # Max free-left before forced protection
            'cooldown_after_protected': 15,    # Cooldown after returning to free
            'extension_threshold': 3,          # Extend green if queue > this
            
            # Pedestrian
            'pedestrian_wait_threshold': 5,    # Seconds before pedestrian mode
            'pedestrian_crossing_duration': 10, # Pedestrian crossing duration
            
            # Spillover
            'spillover_threshold': 2,          # Vehicles at lane exit to trigger
            
            # Dynamic adjustment
            'dynamic_thresholds': True,
            'peak_hour_multiplier': 0.7,       # More sensitive during peak
            'oncoming_traffic_factor': 0.8     # Reduce threshold when oncoming is heavy
        }
        
        # State variables
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self.blocking_vehicles: Dict[str, Dict] = {}
        self.cooldown_until = 0
        self.manual_override = False
        self.peak_hours = [8, 9, 17, 18]  # 8-9 AM, 5-6 PM
        
        # Sliding windows for dynamic timing
        self.vehicle_history = deque(maxlen=30)
        self.weighted_history = deque(maxlen=30)
        self.arrival_rate = 0.0
        self.departure_rate = 0.0
        
        # Pedestrian tracking
        self.pedestrian_mode_triggered = False
        self.pedestrian_mode_start = 0
        self.pedestrians_waiting_count = 0
        self.pedestrians_waiting_start = 0
        
        # Emergency mode
        self.emergency_mode = False
        self.emergency_start = 0
        self.emergency_vehicle_type = None
        
        # Statistics
        self.protected_triggers = 0
        self.pedestrian_triggers = 0
        self.emergency_triggers = 0
        self.spillover_triggers = 0
        self.event_log: List[Dict] = []
        
        # Conflict predictor
        self.conflict_predictor = ConflictPredictor()
        
        logger.info("Enhanced Signal Controller initialized with complete constraints")
        logger.info(f"  Weighted threshold: {self.config['weighted_threshold']}")
        logger.info(f"  Min protected: {self.config['min_protected_duration']}s")
        logger.info(f"  Max free-left: {self.config['max_free_left_duration']}s")
    
    def update_detections(self, lane_vehicles: List, pedestrians: List, 
                          detector=None, oncoming_count: int = 0,
                          weighted_occupancy: float = 0,
                          spillover: bool = False,
                          emergency: bool = False,
                          emergency_type: str = None) -> SignalPhase: # pyright: ignore[reportArgumentType]
        """
        Complete update with all constraints
        
        Args:
            lane_vehicles: Vehicles in free-left lane
            pedestrians: Pedestrians detected
            detector: Detector instance for additional methods
            oncoming_count: Number of oncoming vehicles
            weighted_occupancy: Pre-calculated weighted occupancy
            spillover: Whether spillover detected
            emergency: Emergency vehicle detected
            emergency_type: Type of emergency vehicle
            
        Returns:
            Current signal phase
        """
        if self.manual_override:
            return self.current_phase
        
        current_time = time.time()
        
        # ========== PRIORITY 1: EMERGENCY VEHICLES ==========
        if emergency:
            if not self.emergency_mode:
                self.emergency_mode = True
                self.emergency_start = current_time
                self.emergency_vehicle_type = emergency_type
                self.emergency_triggers += 1
                self._log_event('EMERGENCY_MODE', {'type': emergency_type})
                logger.warning(f"🚨 EMERGENCY VEHICLE DETECTED - {emergency_type or 'Unknown'} - All red activated")
            
            self.current_phase = SignalPhase.EMERGENCY_STOP
            self.phase_start_time = current_time
            return SignalPhase.EMERGENCY_STOP
        
        # Reset emergency after duration (10 seconds)
        if self.emergency_mode and current_time - self.emergency_start > 10:
            self.emergency_mode = False
            self.emergency_vehicle_type = None
            self._log_event('EMERGENCY_MODE_END', {})
            logger.info("Emergency mode ended")
        
        # ========== PRIORITY 2: PEDESTRIAN SAFETY ==========
        if pedestrians and not self.pedestrian_mode_triggered:
            if self.pedestrians_waiting_count == 0:
                self.pedestrians_waiting_start = current_time
            self.pedestrians_waiting_count = len(pedestrians)
            
            waiting_time = current_time - self.pedestrians_waiting_start
            
            # Trigger pedestrian mode if:
            # - 2+ pedestrians waiting OR
            # - Waiting time exceeds threshold
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
                logger.info(f"🚶 Pedestrian mode activated - {self.pedestrians_waiting_count} waiting")
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
        
        # ========== PRIORITY 3: CONFLICT DETECTION ==========
        conflict = self.conflict_predictor.predict(lane_vehicles, pedestrians)
        if conflict['has_conflict'] and conflict['risk'] == 'HIGH':
            self.current_phase = SignalPhase.EMERGENCY_STOP
            self.phase_start_time = current_time
            self._log_event('CONFLICT_DETECTED', {'conflicts': len(conflict['conflicts'])})
            logger.warning(f"⚠️ CONFLICT DETECTED - Emergency stop")
            return SignalPhase.EMERGENCY_STOP
        
        # ========== PRIORITY 4: UPDATE HISTORICAL DATA ==========
        # Update sliding windows
        self.vehicle_history.append(len(lane_vehicles))
        self.weighted_history.append(weighted_occupancy)
        
        # Calculate arrival rate
        if len(self.vehicle_history) > 5:
            recent = list(self.vehicle_history)[-10:]
            self.arrival_rate = sum(recent) / 10
        
        # ========== PRIORITY 5: DYNAMIC TIMING ADJUSTMENT ==========
        self._update_dynamic_timing(len(lane_vehicles), oncoming_count, weighted_occupancy)
        
        # ========== PRIORITY 6: SPILLOVER MANAGEMENT ==========
        if spillover and not self.pedestrian_mode_triggered:
            self.spillover_triggers += 1
            self._log_event('SPILLOVER_DETECTED', {'queue': len(lane_vehicles)})
            logger.info(f"⚠️ Spillover detected - {len(lane_vehicles)} vehicles in queue")
        
        # ========== DECISION LOGIC ==========
        current_threshold = self._get_current_threshold(oncoming_count)
        
        should_protect = False
        trigger_reason = ""
        
        # Condition 1: Weighted occupancy threshold
        if weighted_occupancy >= self.config['weighted_threshold']:
            should_protect = True
            trigger_reason = f"Weighted occupancy: {weighted_occupancy:.1f} >= {self.config['weighted_threshold']}"
        
        # Condition 2: Spillover detected
        elif spillover:
            should_protect = True
            trigger_reason = f"Spillover detected - {len(lane_vehicles)} vehicles"
        
        # Condition 3: Persistent blocking
        elif self._has_persistent_blocking():
            should_protect = True
            trigger_reason = "Persistent blocking detected"
        
        # Condition 4: Timeout
        elif current_time - self.phase_start_time > self.config['max_free_left_duration']:
            should_protect = True
            trigger_reason = f"Timeout ({self.config['max_free_left_duration']}s)"
        
        # Condition 5: Traffic surge detected
        elif self._has_traffic_surge():
            should_protect = True
            trigger_reason = "Traffic surge detected"
        
        # Condition 6: Peak hour with any occupancy
        elif self._is_peak_hour() and weighted_occupancy > 0.5:
            should_protect = True
            trigger_reason = f"Peak hour with occupancy {weighted_occupancy:.1f}"
        
        # Execute protection
        if self.current_phase == SignalPhase.FREE_LEFT:
            if should_protect:
                self.current_phase = SignalPhase.PROTECTED_LEFT
                self.phase_start_time = current_time
                self.protected_triggers += 1
                self._log_event('PROTECTED_TRIGGERED', {
                    'weighted_occupancy': weighted_occupancy,
                    'reason': trigger_reason,
                    'spillover': spillover,
                    'queue_length': len(lane_vehicles)
                })
                logger.info(f"🟢 PROTECTED LEFT - {trigger_reason}")
        
        # Handle existing protected phase
        elif self.current_phase == SignalPhase.PROTECTED_LEFT:
            phase_duration = current_time - self.phase_start_time
            
            # Check if should extend green for burst traffic
            extend = self._should_extend_green(lane_vehicles)
            
            if extend:
                self._log_event('GREEN_EXTENDED', {'duration': phase_duration})
                # Extend by resetting timer (up to max)
                if phase_duration < self.config['max_protected_duration']:
                    self.phase_start_time = current_time
            
            # Minimum duration check
            if phase_duration >= self.config['min_protected_duration']:
                # Check if lane is clear (weighted)
                if weighted_occupancy < 0.5:
                    self.current_phase = SignalPhase.FREE_LEFT
                    self.phase_start_time = current_time
                    self.cooldown_until = current_time + self.config['cooldown_after_protected']
                    self._log_event('RETURNED_TO_FREE', {'duration': phase_duration})
                    logger.info(f"🟡 Returned to free left - {phase_duration:.1f}s")
        
        return self.current_phase
    
    def _update_dynamic_timing(self, queue_length: int, oncoming_count: int, weighted_occupancy: float) -> None:
        """Update timing based on current conditions"""
        # Dynamic minimum green based on queue length
        if queue_length > 3:
            self.config['min_protected_duration'] = min(20, 10 + queue_length)
        else:
            self.config['min_protected_duration'] = 10
        
        # Dynamic maximum based on oncoming traffic
        if oncoming_count > 15:
            self.config['max_free_left_duration'] = 20
        elif oncoming_count > 10:
            self.config['max_free_left_duration'] = 30
        else:
            self.config['max_free_left_duration'] = 45
        
        # Dynamic weighted threshold based on conditions
        if self._is_peak_hour():
            self.config['weighted_threshold'] = 2.0
        elif oncoming_count > 10:
            self.config['weighted_threshold'] = 2.5
        else:
            self.config['weighted_threshold'] = 3.0
    
    def _has_persistent_blocking(self) -> bool:
        """Check if any vehicle has been blocking for too long"""
        current_time = time.time()
        for v in self.blocking_vehicles.values():
            if current_time - v.get('first_seen', current_time) > self.config['blocking_duration_threshold']:
                return True
        return False
    
    def _has_traffic_surge(self) -> bool:
        """Detect sudden surge in vehicles using sliding window"""
        if len(self.vehicle_history) < 10:
            return False
        
        recent = list(self.vehicle_history)[-5:]
        previous = list(self.vehicle_history)[-10:-5]
        
        recent_avg = sum(recent) / len(recent) if recent else 0
        previous_avg = sum(previous) / len(previous) if previous else 0
        
        # Surge if recent > previous * 1.5 and at least 2 vehicles
        return recent_avg > previous_avg * 1.5 and recent_avg > 2
    
    def _should_extend_green(self, lane_vehicles: List) -> bool:
        """Determine if green should be extended for burst traffic"""
        # Extend if queue is long
        if len(lane_vehicles) > self.config['extension_threshold']:
            return True
        
        # Extend if surge detected
        if self._has_traffic_surge():
            return True
        
        return False
    
    def _get_current_threshold(self, oncoming_count: int = 0) -> float:
        """Get current weighted threshold (adjusted for conditions)"""
        threshold = self.config['weighted_threshold']
        
        # Adjust for peak hours
        if self._is_peak_hour() and self.config['dynamic_thresholds']:
            threshold *= self.config['peak_hour_multiplier']
        
        # Adjust for heavy oncoming traffic
        if oncoming_count > 15:
            threshold *= self.config['oncoming_traffic_factor']
        
        return max(1.0, threshold)
    
    def _is_peak_hour(self) -> bool:
        """Check if current time is within peak hours"""
        current_hour = time.localtime().tm_hour
        return current_hour in self.peak_hours
    
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
        """Log event for audit trail"""
        self.event_log.append({
            'timestamp': time.time(),
            'event': event_type,
            'data': data
        })
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-1000:]
    
    def get_state(self) -> Dict:
        """Get current signal state"""
        current_time = time.time()
        
        return {
            'phase': self.current_phase.value,
            'phase_duration': current_time - self.phase_start_time,
            'blocking_vehicles': len(self.blocking_vehicles),
            'cooldown_remaining': max(0, self.cooldown_until - current_time),
            'is_peak_hour': self._is_peak_hour(),
            'risk_level': 'HIGH' if self.emergency_mode else 'MEDIUM',
            'pedestrian_mode_active': self.pedestrian_mode_triggered,
            'pedestrians_waiting': self.pedestrians_waiting_count,
            'emergency_mode': self.emergency_mode,
            'emergency_type': self.emergency_vehicle_type,
            'protected_triggers': self.protected_triggers,
            'pedestrian_triggers': self.pedestrian_triggers,
            'emergency_triggers': self.emergency_triggers,
            'spillover_triggers': self.spillover_triggers,
            'weighted_threshold': self.config['weighted_threshold'],
            'min_protected_duration': self.config['min_protected_duration'],
            'max_free_left_duration': self.config['max_free_left_duration']
        }
    
    def manual_protect(self) -> None:
        """Manual override to trigger protected left"""
        self.manual_override = True
        self.current_phase = SignalPhase.PROTECTED_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PROTECT', {})
        logger.info("🔒 Manual override - Protected left activated")
    
    def manual_emergency(self, vehicle_type: str = "Manual") -> None:
        """Manual emergency override"""
        self.emergency_mode = True
        self.emergency_start = time.time()
        self.emergency_vehicle_type = vehicle_type
        self.current_phase = SignalPhase.EMERGENCY_STOP
        self.phase_start_time = time.time()
        self._log_event('MANUAL_EMERGENCY', {'type': vehicle_type})
        logger.info(f"🚨 Manual emergency override - {vehicle_type}")
    
    def manual_pedestrian(self) -> None:
        """Manual pedestrian crossing override"""
        self.manual_override = True
        self.pedestrian_mode_triggered = True
        self.pedestrian_mode_start = time.time()
        self.current_phase = SignalPhase.PEDESTRIAN_CROSSING
        self.phase_start_time = time.time()
        self._log_event('MANUAL_PEDESTRIAN', {})
        logger.info("🚶 Manual override - Pedestrian crossing activated")
    
    def manual_reset(self) -> None:
        """Reset to automatic mode"""
        self.manual_override = False
        self.emergency_mode = False
        self.pedestrian_mode_triggered = False
        self.current_phase = SignalPhase.FREE_LEFT
        self.phase_start_time = time.time()
        self._log_event('MANUAL_RESET', {})
        logger.info("🔄 Manual reset - Automatic mode restored")
    
    def integrate_dataset(self, insights: Dict) -> None:
        """Integrate dataset insights to adjust thresholds"""
        risk = insights.get('risk_assessment', {})
        risk_level = risk.get('level', 'MEDIUM')
        violations = insights.get('violation_analysis', {})
        
        # Adjust based on risk level
        if risk_level == 'HIGH':
            self.config['weighted_threshold'] = 1.5
            self.config['blocking_duration_threshold'] = 3
            self.config['max_free_left_duration'] = 20
            self.config['cooldown_after_protected'] = 10
            logger.info("HIGH RISK - Aggressive thresholds applied")
        elif risk_level == 'MEDIUM':
            self.config['weighted_threshold'] = 2.5
            self.config['blocking_duration_threshold'] = 5
            self.config['max_free_left_duration'] = 30
            self.config['cooldown_after_protected'] = 15
            logger.info("MEDIUM RISK - Standard thresholds applied")
        else:  # LOW
            self.config['weighted_threshold'] = 3.5
            self.config['blocking_duration_threshold'] = 8
            self.config['max_free_left_duration'] = 45
            self.config['cooldown_after_protected'] = 20
            logger.info("LOW RISK - Relaxed thresholds applied")
        
        # Extract peak hours
        peak_hours = violations.get('peak_hours', {})
        if peak_hours:
            self.peak_hours = [int(h) for h in peak_hours.keys()]
        
        self._log_event('DATASET_INTEGRATED', {'risk_level': risk_level})
    
    def get_events(self, limit: int = 50) -> List[Dict]:
        """Get recent events"""
        return self.event_log[-limit:]
"""
Fatigue detection module that combines all signals.
Implements persistent alarm that ONLY stops when person returns to normal.
"""

import time
from typing import Tuple, Dict, Optional
from enum import Enum

from utils.config import DetectionConfig
from utils.logger import SystemLogger


class FatigueState(Enum):
    """Enumeration of possible fatigue states"""
    NORMAL = "NORMAL"
    FATIGUE = "FATIGUE"
    ALERT = "ALERT"


class FatigueDetector:
    """
    Main fatigue detection logic with PERSISTENT alarm.
    Alarm stays active until ALL signs of fatigue disappear.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        
        # Fatigue score (0-100)
        self.fatigue_score = 0.0
        
        # Component states (real-time)
        self.eyes_closed = False
        self.yawning = False
        self.head_dropping = False
        self.dangerous_posture = False
        self.posture_label = "NORMAL"
        
        # Component states (persistent)
        self.eyes_closed_persistent = False
        self.yawning_persistent = False
        self.head_dropping_persistent = False
        
        # Sleep detection
        self.sleep_frame_count = 0
        self.sleep_detected = False
        
        # State tracking
        self.current_state = FatigueState.NORMAL
        self.previous_state = FatigueState.NORMAL
        
        # PERSISTENT ALARM - Only stops when NORMAL state is confirmed
        self.alarm_active = False
        self.normal_frames_required = self.config.NORMAL_FRAMES_TO_CLEAR_ALARM
        self.normal_frame_counter = 0
        
        # Timing
        self.state_start_time = time.time()
        self.last_score_log = 0
        
        # Detection flags
        self.yawn_detected_flag = False
        self.eyes_closed_flag = False
        
    def update(self, eyes_closed: bool, yawning: bool, head_drop: bool,
               dangerous_posture: bool = False, posture_label: str = "NORMAL") -> Tuple[FatigueState, Dict]:
        """
        Update fatigue detection with new component states.
        
        Args:
            eyes_closed: Whether eyes are closed (real-time)
            yawning: Whether yawning is detected (real-time)
            head_drop: Whether head drop is detected (real-time)
            dangerous_posture: Whether another dangerous head posture is detected
            posture_label: Human-readable dangerous posture label
            
        Returns:
            Tuple of (current_state, metrics_dictionary)
        """
        previous_score = self.fatigue_score
        previous_dangerous_posture = self.dangerous_posture
        previous_posture_label = self.posture_label
        
        # Update real-time states
        self.eyes_closed = eyes_closed
        self.yawning = yawning
        self.head_dropping = head_drop
        self.dangerous_posture = dangerous_posture
        self.posture_label = posture_label if dangerous_posture else "NORMAL"
        
        # Update persistent flags for logging
        if eyes_closed:
            self.eyes_closed_flag = True
        if yawning:
            self.yawn_detected_flag = True
        
        # Sleep detection (prolonged eye closure)
        if eyes_closed:
            self.sleep_frame_count += 1
        else:
            self.sleep_frame_count = max(0, self.sleep_frame_count - 3)
        
        self.sleep_detected = self.sleep_frame_count >= self.config.SLEEP_CONSECUTIVE_FRAMES
        
        # Update fatigue score based on signals
        if self.sleep_detected:
            self.fatigue_score += self.config.SLEEP_SCORE_INCREMENT
            if previous_score < self.config.FATIGUE_SCORE_ALERT_THRESHOLD:
                self.logger.warning(f"Sleep detected - Fatigue score: {self.fatigue_score:.0f}")
        elif eyes_closed:
            self.fatigue_score += self.config.EYE_CLOSE_SCORE_INCREMENT
        elif yawning:
            self.fatigue_score += self.config.YAWN_SCORE_INCREMENT
            self.logger.info(f"Yawn detected - Fatigue score: {self.fatigue_score:.0f}")
        elif head_drop:
            self.fatigue_score += self.config.HEAD_DROP_SCORE_INCREMENT
        elif dangerous_posture:
            if self.posture_label == "CONDUCTEUR HORS CADRE":
                self.fatigue_score += self.config.NO_FACE_SCORE_INCREMENT
            else:
                self.fatigue_score += self.config.DANGEROUS_POSTURE_SCORE_INCREMENT
            if not previous_dangerous_posture or previous_posture_label != self.posture_label:
                self.logger.warning(f"Dangerous posture detected: {self.posture_label}")
        else:
            # Decay only if no fatigue signs
            self.fatigue_score = max(0, self.fatigue_score - self.config.FATIGUE_DECAY_RATE)
        
        # Clamp fatigue score
        self.fatigue_score = min(self.config.FATIGUE_SCORE_MAX, self.fatigue_score)
        
        # Determine state based on fatigue score
        self.previous_state = self.current_state
        
        if self.fatigue_score >= self.config.FATIGUE_SCORE_ALERT_THRESHOLD:
            self.current_state = FatigueState.ALERT
            # Alarm is triggered immediately
            self.alarm_active = True
            self.normal_frame_counter = 0
        elif self.fatigue_score >= self.config.FATIGUE_SCORE_WARNING_THRESHOLD:
            self.current_state = FatigueState.FATIGUE
            # Keep alarm if it was active, but allow recovery
            if not self.alarm_active:
                self.normal_frame_counter = 0
        else:
            self.current_state = FatigueState.NORMAL
            
            # PERSISTENT ALARM LOGIC:
            # Alarm continues until we have enough consecutive normal frames
            if self.alarm_active:
                self.normal_frame_counter += 1
                
                # Need sustained normal state to stop alarm
                if self.normal_frame_counter >= self.normal_frames_required:
                    self.alarm_active = False
                    self.normal_frame_counter = 0
                    self.logger.info("Alarm stopped - Driver returned to normal state")
                    # Reset detection flags
                    self.yawn_detected_flag = False
                    self.eyes_closed_flag = False
            else:
                self.normal_frame_counter = 0
        
        # Log state changes
        if self.current_state != self.previous_state:
            self.state_start_time = time.time()
            self.logger.info(f"State changed: {self.previous_state.value} -> {self.current_state.value}")
        
        # Log fatigue score periodically
        if time.time() - self.last_score_log > 5:
            alarm_status = "ACTIVE" if self.alarm_active else "INACTIVE"
            self.logger.debug(f"Fatigue: {self.fatigue_score:.0f}% | State: {self.current_state.value} | Alarm: {alarm_status}")
            self.last_score_log = time.time()
        
        # Prepare metrics
        metrics = self.get_metrics()
        
        return self.current_state, metrics
    
    def should_trigger_alarm(self) -> bool:
        """
        Determine if alarm should be triggered.
        Returns True if alarm is active (persistent).
        """
        return self.alarm_active
    
    def should_trigger_warning(self) -> bool:
        """
        Determine if warning beep should be triggered.
        Only for fatigue state without active alarm.
        """
        return self.current_state == FatigueState.FATIGUE and not self.alarm_active
    
    def get_metrics(self) -> Dict:
        """
        Get current fatigue metrics.
        
        Returns:
            Dictionary with all fatigue metrics
        """
        return {
            'fatigue_score': self.fatigue_score,
            'current_state': self.current_state.value,
            'eyes_closed': self.eyes_closed,
            'yawning': self.yawning,
            'head_dropping': self.head_dropping,
            'dangerous_posture': self.dangerous_posture,
            'posture_label': self.posture_label,
            'sleep_detected': self.sleep_detected,
            'sleep_frames': self.sleep_frame_count,
            'alarm_active': self.alarm_active,
            'normal_frames': self.normal_frame_counter,
            'state_duration': time.time() - self.state_start_time,
            'yawn_detected': self.yawn_detected_flag,
            'eyes_closed_detected': self.eyes_closed_flag
        }
    
    def reset(self):
        """Reset all fatigue detection variables"""
        self.fatigue_score = 0.0
        self.eyes_closed = False
        self.yawning = False
        self.head_dropping = False
        self.dangerous_posture = False
        self.posture_label = "NORMAL"
        self.eyes_closed_persistent = False
        self.yawning_persistent = False
        self.head_dropping_persistent = False
        self.sleep_frame_count = 0
        self.sleep_detected = False
        self.current_state = FatigueState.NORMAL
        self.previous_state = FatigueState.NORMAL
        self.alarm_active = False
        self.normal_frame_counter = 0
        self.state_start_time = time.time()
        self.last_score_log = 0
        self.yawn_detected_flag = False
        self.eyes_closed_flag = False

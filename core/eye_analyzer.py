"""
Eye analysis module for detecting eye closure using EAR (Eye Aspect Ratio).
Optimized for maximum precision with multi-level validation.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional

from utils.config import DetectionConfig
from utils.logger import SystemLogger


class EyeAnalyzer:
    """
    Analyzes eye states using Eye Aspect Ratio (EAR).
    Implements multi-level validation to eliminate false positives.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        
        # EAR calculation parameters
        self.smooth_ear_left = 0.35
        self.smooth_ear_right = 0.35
        self.smooth_factor = self.config.EAR_SMOOTHING_FACTOR
        
        # History for stability
        self.ear_history = deque(maxlen=self.config.EAR_HISTORY_SIZE)
        self.left_ear_history = deque(maxlen=self.config.EAR_HISTORY_SIZE)
        self.right_ear_history = deque(maxlen=self.config.EAR_HISTORY_SIZE)
        self.baseline_history = deque(maxlen=self.config.EYE_BASELINE_FRAMES)
        
        # Eye closure tracking with confirmation
        self.eye_close_frame_count = 0
        self.eye_close_confirmed = False
        self.eye_close_persistent = False
        
        # Statistics
        self.ear_min = 1.0
        self.ear_max = 0.0
        self.ear_avg = 0.35
        self.ear_std = 0.05
        
        # Validation flags
        self.last_valid_ear_left = 0.35
        self.last_valid_ear_right = 0.35
        self.consecutive_valid_left = 0
        self.consecutive_valid_right = 0
        self.open_eye_baseline = 0.35
        self.last_symmetry_error = 0.0
        
    def calculate_ear_precise(self, eye_points: List[Tuple[int, int]], side: str) -> float:
        """
        Calculate Eye Aspect Ratio with precise geometry validation.
        
        Args:
            eye_points: List of 6 points around the eye in correct order
            
        Returns:
            EAR value between 0 and 0.5 typically
        """
        last_valid_attr = f"last_valid_ear_{side}"
        consecutive_attr = f"consecutive_valid_{side}"
        last_valid_ear = getattr(self, last_valid_attr)
        consecutive_valid = getattr(self, consecutive_attr)

        if len(eye_points) < 6:
            return last_valid_ear
        
        try:
            # Convert to numpy arrays for efficient computation
            points = np.array(eye_points)
            
            # Calculate vertical distances (points 1-5 and 2-4)
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance (points 0-3)
            h = np.linalg.norm(points[0] - points[3])
            if h < self.config.EYE_MIN_VALID_WIDTH:
                setattr(self, consecutive_attr, 0)
                return last_valid_ear
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.35
            
            # Clamp to realistic range
            ear = max(0.15, min(0.50, ear))
            
            # Validate against sudden jumps (avoid false positives)
            if abs(ear - last_valid_ear) > 0.15:
                setattr(self, consecutive_attr, 0)
                return last_valid_ear

            consecutive_valid += 1
            setattr(self, consecutive_attr, consecutive_valid)
            if consecutive_valid >= 3:
                setattr(self, last_valid_attr, ear)

            return ear

        except Exception as e:
            self.logger.debug(f"EAR calculation error: {e}")
            return last_valid_ear
    
    def update(self, left_eye_points: List[Tuple[int, int]], 
               right_eye_points: List[Tuple[int, int]]) -> Tuple[float, bool]:
        """
        Update eye state with new landmarks.
        
        Args:
            left_eye_points: 6 points for left eye
            right_eye_points: 6 points for right eye
            
        Returns:
            Tuple of (average_ear, eyes_closed)
        """
        # Calculate EAR for both eyes
        ear_left = self.calculate_ear_precise(left_eye_points, "left") if left_eye_points else self.smooth_ear_left
        ear_right = self.calculate_ear_precise(right_eye_points, "right") if right_eye_points else self.smooth_ear_right
        
        # Apply exponential smoothing
        self.smooth_ear_left = (self.smooth_factor * ear_left + 
                                (1 - self.smooth_factor) * self.smooth_ear_left)
        self.smooth_ear_right = (self.smooth_factor * ear_right + 
                                 (1 - self.smooth_factor) * self.smooth_ear_right)
        
        avg_ear = (self.smooth_ear_left + self.smooth_ear_right) / 2.0
        self.last_symmetry_error = abs(self.smooth_ear_left - self.smooth_ear_right)
        
        # Update history
        self.ear_history.append(avg_ear)
        self.left_ear_history.append(self.smooth_ear_left)
        self.right_ear_history.append(self.smooth_ear_right)
        
        # Calculate statistics
        min_stats_frames = min(10, max(5, self.config.EAR_HISTORY_SIZE // 2))
        if len(self.ear_history) >= min_stats_frames:
            self.ear_avg = np.mean(self.ear_history)
            self.ear_std = np.std(self.ear_history)
            self.ear_min = min(self.ear_min, avg_ear)
            self.ear_max = max(self.ear_max, avg_ear)

        eyes_are_balanced = self.last_symmetry_error <= self.config.EYE_MAX_ASYMMETRY
        likely_open = (
            avg_ear >= self.config.EAR_THRESHOLD
            and eyes_are_balanced
            and len(left_eye_points) >= 6
            and len(right_eye_points) >= 6
        )
        if likely_open:
            self.baseline_history.append(avg_ear)
            self.open_eye_baseline = float(np.median(self.baseline_history))
        
        # Multi-level eye closure detection
        # Level 1: Current value below threshold
        adaptive_threshold = min(
            self.config.EAR_THRESHOLD,
            self.open_eye_baseline * self.config.EYE_CLOSED_BASELINE_RATIO,
        )
        below_threshold = avg_ear < adaptive_threshold
        both_eyes_low = (
            self.smooth_ear_left < adaptive_threshold
            and self.smooth_ear_right < adaptive_threshold
        )
        
        # Level 2: Average below threshold (smooths noise)
        avg_below = self.ear_avg < self.config.EAR_THRESHOLD
        
        # Level 3: Standard deviation low (stable closure)
        stable_closure = len(self.ear_history) >= min_stats_frames and self.ear_std < 0.03
        
        # Combined validation
        is_valid_closure = below_threshold and both_eyes_low and eyes_are_balanced and (avg_below or stable_closure)
        
        # Update frame counter with validation
        if is_valid_closure:
            self.eye_close_frame_count += 1
        else:
            self.eye_close_frame_count = max(0, self.eye_close_frame_count - 2)
        
        # Confirmation after consecutive frames
        self.eye_close_confirmed = self.eye_close_frame_count >= self.config.EAR_CONSECUTIVE_FRAMES
        
        # Persistent state (more stable for alarm)
        if self.eye_close_confirmed:
            self.eye_close_persistent = True
        elif self.eye_close_frame_count == 0:
            self.eye_close_persistent = False
        
        # Log significant changes
        if self.eye_close_confirmed and self.eye_close_frame_count % 10 == 0:
            self.logger.debug(f"Eyes closed confirmed - EAR: {avg_ear:.3f}, Frames: {self.eye_close_frame_count}")
        
        return avg_ear, self.eye_close_persistent
    
    def reset(self):
        """Reset all tracking variables"""
        self.eye_close_frame_count = 0
        self.eye_close_confirmed = False
        self.eye_close_persistent = False
        self.ear_history.clear()
        self.left_ear_history.clear()
        self.right_ear_history.clear()
        self.ear_min = 1.0
        self.ear_max = 0.0
        self.ear_avg = 0.35
        self.ear_std = 0.05
        self.baseline_history.clear()
        self.consecutive_valid_left = 0
        self.consecutive_valid_right = 0
        self.last_valid_ear_left = 0.35
        self.last_valid_ear_right = 0.35
        self.open_eye_baseline = 0.35
        self.last_symmetry_error = 0.0
    
    def get_statistics(self) -> dict:
        """
        Get eye statistics.
        
        Returns:
            Dictionary with eye statistics
        """
        return {
            'ear_current': self.ear_avg,
            'ear_min': self.ear_min,
            'ear_max': self.ear_max,
            'ear_std': self.ear_std,
            'eye_baseline': self.open_eye_baseline,
            'eye_asymmetry': self.last_symmetry_error,
            'eye_close_frames': self.eye_close_frame_count,
            'eye_close_confirmed': self.eye_close_confirmed,
            'eye_close_persistent': self.eye_close_persistent
        }

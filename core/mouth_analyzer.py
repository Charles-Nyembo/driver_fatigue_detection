"""
Mouth analysis module for detecting yawning.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional

from utils.config import DetectionConfig
from utils.logger import SystemLogger


class MouthAnalyzer:
    """
    Analyzes mouth states for yawn detection.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        
        # MAR calculation
        self.smooth_mar = 0.20
        self.smooth_factor = self.config.MAR_SMOOTHING_FACTOR
        
        # History
        self.mar_history = deque(maxlen=self.config.MAR_HISTORY_SIZE)
        self.mouth_height_history = deque(maxlen=15)
        self.mouth_width_history = deque(maxlen=15)
        
        # Baseline calibration
        self.mouth_baseline = None
        self.baseline_height = 0
        self.baseline_history = deque(maxlen=self.config.MOUTH_BASELINE_FRAMES)
        self.baseline_height_history = deque(maxlen=self.config.MOUTH_BASELINE_FRAMES)
        self.is_calibrated = False
        
        # Yawn tracking
        self.mouth_open_frame_count = 0
        self.mouth_open_confirmed = False
        self.yawn_frame_count = 0
        self.yawn_confirmed = False
        self.last_valid_mar = 0.20
        self.last_relative_opening = 1.0
        
    def calculate_mar(self, mouth_points: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate MAR and mouth height."""
        if len(mouth_points) < 20:
            return self.last_valid_mar, 0
        
        try:
            points = np.array(mouth_points)
            x_coords = points[:, 0]
            mouth_width = float(np.max(x_coords) - np.min(x_coords))
            center_x = float(np.median(x_coords))
            central_band = points[np.abs(x_coords - center_x) <= mouth_width * 0.35]
            if len(central_band) < 4:
                central_band = points

            top_y = float(np.mean(np.sort(central_band[:, 1])[:2]))
            bottom_y = float(np.mean(np.sort(central_band[:, 1])[-2:]))
            mouth_height = max(0.0, bottom_y - top_y)
            mar = mouth_height / mouth_width if mouth_width > 0 else 0.20
            mar = max(0.05, min(0.85, mar))

            self.last_valid_mar = mar
            self.mar_history.append(mar)
            self.mouth_height_history.append(mouth_height)
            self.mouth_width_history.append(mouth_width)
            
            return mar, mouth_height
            
        except Exception:
            return self.last_valid_mar, 0
    
    def calibrate_baseline(self, mar: float, height: float) -> None:
        """Calibrate mouth baseline."""
        if not self.is_calibrated and mar < 0.35 and height > 0:
            self.baseline_history.append(mar)
            self.baseline_height_history.append(height)
            
            if len(self.baseline_history) >= self.config.MOUTH_BASELINE_FRAMES:
                self.mouth_baseline = np.median(list(self.baseline_history))
                self.baseline_height = np.median(list(self.baseline_height_history))
                self.is_calibrated = True
                self.logger.info(f"Mouth calibrated: MAR={self.mouth_baseline:.3f}")
    
    def detect_yawn(self, mar: float, height: float) -> Tuple[bool, bool]:
        """Detect mouth opening and yawn with strict conditions."""
        if not self.is_calibrated:
            return False, False
        
        # Smooth MAR
        self.smooth_mar = (self.smooth_factor * mar + 
                          (1 - self.smooth_factor) * self.smooth_mar)
        
        # Calculate relative opening
        relative_opening = self.smooth_mar / max(self.mouth_baseline + 0.03, 1e-6)
        self.last_relative_opening = relative_opening
        
        # Calculate height increase
        height_increase = height / (self.baseline_height + 1.0)
        
        mouth_open = (
            self.smooth_mar > self.config.MAR_THRESHOLD
            or relative_opening > self.config.MOUTH_OPEN_MIN_RELATIVE
        )

        mar_high = self.smooth_mar > self.config.YAWN_MIN_MAR
        relative_high = relative_opening > self.config.YAWN_MIN_RELATIVE_OPENING
        height_sufficient = height_increase > self.config.YAWN_MIN_HEIGHT_MULTIPLIER

        # A yawn is a sustained, strong opening, not just an open mouth.
        is_yawn = mouth_open and mar_high and relative_high and height_sufficient

        return mouth_open, is_yawn
    
    def update(self, mouth_points: List[Tuple[int, int]]) -> Tuple[float, bool]:
        """Update mouth state."""
        if not mouth_points:
            return self.smooth_mar, False

        mar, height = self.calculate_mar(mouth_points)
        self.calibrate_baseline(mar, height)
        mouth_open, is_yawn = self.detect_yawn(mar, height)

        if mouth_open:
            self.mouth_open_frame_count += 1
        else:
            self.mouth_open_frame_count = max(0, self.mouth_open_frame_count - 1)

        self.mouth_open_confirmed = (
            self.mouth_open_frame_count >= self.config.MOUTH_OPEN_CONSECUTIVE_FRAMES
        )

        # Update yawn counter
        if is_yawn:
            self.yawn_frame_count += 1
        else:
            self.yawn_frame_count = max(0, self.yawn_frame_count - 2)
        
        self.yawn_confirmed = self.yawn_frame_count >= self.config.YAWN_CONSECUTIVE_FRAMES
        
        return self.smooth_mar, self.yawn_confirmed
    
    def reset(self):
        """Reset all variables."""
        self.smooth_mar = 0.20
        self.mouth_open_frame_count = 0
        self.mouth_open_confirmed = False
        self.yawn_frame_count = 0
        self.yawn_confirmed = False
        self.mar_history.clear()
        self.mouth_height_history.clear()
        self.mouth_width_history.clear()
        self.baseline_history.clear()
        self.baseline_height_history.clear()
        self.is_calibrated = False
        self.mouth_baseline = None
        self.baseline_height = 0
        self.last_valid_mar = 0.20
        self.last_relative_opening = 1.0

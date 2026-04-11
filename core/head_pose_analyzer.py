"""
Head pose analysis module for detecting dangerous driver head/face postures.
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional

from utils.config import DetectionConfig
from utils.logger import SystemLogger


class HeadPoseAnalyzer:
    """
    Analyzes head and face posture to detect dangerous driver positions.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        self.head_position_frames_required = max(1, int(self.config.CAMERA_FPS * self.config.HEAD_POSITION_HOLD_SECONDS))
        
        # Head drop tracking
        self.head_drop_frame_count = 0
        self.head_drop_confirmed = False
        self.head_drop_persistent = False
        self.dangerous_posture_frame_count = 0
        self.dangerous_posture_persistent = False
        self.posture_label = "NORMAL"
        
        # Position tracking
        self.prev_nose_y = None
        self.prev_chin_y = None
        self.prev_nose_x = None
        self.prev_chin_x = None
        self.nose_chin_distances = deque(maxlen=self.config.HEAD_DROP_HISTORY_SIZE)
        self.baseline_distance_history = deque(maxlen=self.config.HEAD_BASELINE_FRAMES)
        
        # Head tilt history
        self.tilt_history = deque(maxlen=self.config.HEAD_DROP_HISTORY_SIZE)
        
        # Statistics
        self.vertical_movement = 0.0
        self.max_head_drop = 0.0
        self.baseline_nose_y = None
        self.baseline_chin_distance = None
        
    def update(
        self,
        nose_position: Optional[Tuple[int, int]],
        chin_position: Optional[Tuple[int, int]],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        left_eye_points: Optional[list] = None,
        right_eye_points: Optional[list] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[bool, bool, float, str]:
        """
        Update head pose state with new positions.
        
        Args:
            nose_position: (x, y) coordinates of nose tip
            chin_position: (x, y) coordinates of chin
            
        Returns:
            Tuple of (head_drop_detected, dangerous_posture_detected, tilt_ratio, posture_label)
        """
        if nose_position is None or chin_position is None:
            self.head_drop_frame_count = max(0, self.head_drop_frame_count - 1)
            self.head_drop_persistent = self.head_drop_frame_count >= self.head_position_frames_required
            self.dangerous_posture_frame_count = max(0, self.dangerous_posture_frame_count - 1)
            self.dangerous_posture_persistent = (
                self.dangerous_posture_frame_count >= self.head_position_frames_required
            )
            if not self.dangerous_posture_persistent and self.dangerous_posture_frame_count == 0:
                self.posture_label = "NORMAL"
            return False, self.dangerous_posture_persistent, 0.0, self.posture_label
        
        nose_y = nose_position[1]
        chin_y = chin_position[1]
        
        # Calculate nose-chin distance
        nose_chin_dist = chin_y - nose_y
        if nose_chin_dist <= 0:
            self.head_drop_frame_count = max(0, self.head_drop_frame_count - 1)
            self.dangerous_posture_frame_count = max(0, self.dangerous_posture_frame_count - 1)
            if self.dangerous_posture_frame_count == 0:
                self.dangerous_posture_persistent = False
                self.posture_label = "NORMAL"
            return self.head_drop_persistent, self.dangerous_posture_persistent, 0.0, self.posture_label
        self.nose_chin_distances.append(nose_chin_dist)
        
        # Calculate tilt ratio (horizontal offset)
        tilt_ratio = (nose_position[0] - chin_position[0]) / max(abs(nose_chin_dist), 1)
        horizontal_angle = float(np.degrees(np.arctan2(abs(nose_position[0] - chin_position[0]), max(abs(nose_chin_dist), 1))))
        
        stable_pose = (
            abs(tilt_ratio) < 0.25
            and self.head_drop_frame_count == 0
            and (
                self.baseline_chin_distance is None
                or nose_chin_dist >= self.baseline_chin_distance * 0.95
            )
        )
        if stable_pose:
            self.baseline_distance_history.append(nose_chin_dist)
            self.baseline_chin_distance = float(np.median(self.baseline_distance_history))
            baseline_nose_samples = [nose_y]
            if self.baseline_nose_y is not None:
                baseline_nose_samples.append(self.baseline_nose_y)
            self.baseline_nose_y = float(np.mean(baseline_nose_samples))

        distance_ratio = 1.0
        nose_drop_ratio = 0.0
        if self.baseline_chin_distance and self.baseline_chin_distance > 0:
            distance_ratio = nose_chin_dist / self.baseline_chin_distance
            if self.baseline_nose_y is not None:
                nose_drop_ratio = (nose_y - self.baseline_nose_y) / self.baseline_chin_distance

        self.vertical_movement = nose_drop_ratio

        drop_detected = (
            self.baseline_chin_distance is not None
            and distance_ratio <= self.config.HEAD_DROP_MIN_DISTANCE_RATIO
            and nose_drop_ratio >= self.config.HEAD_DROP_MIN_NOSE_MOVEMENT_RATIO
        )

        if drop_detected:
            self.head_drop_frame_count += 1
        else:
            self.head_drop_frame_count = max(0, self.head_drop_frame_count - 2)
        
        # Update previous positions
        self.prev_nose_y = nose_y
        self.prev_chin_y = chin_y
        self.prev_nose_x = nose_position[0]
        self.prev_chin_x = chin_position[0]
        
        # Track maximum head drop
        if self.head_drop_frame_count > 0:
            self.max_head_drop = max(self.max_head_drop, self.head_drop_frame_count)
        
        # Confirm head drop after sustained duration
        self.head_drop_confirmed = (
            self.head_drop_frame_count >= self.config.HEAD_DROP_CONSECUTIVE_FRAMES
        )
        
        # Persistent state
        self.head_drop_persistent = self.head_drop_confirmed
        
        # Add to tilt history
        self.tilt_history.append(tilt_ratio)
        avg_tilt = np.mean(self.tilt_history) if self.tilt_history else 0.0

        dangerous_posture = False
        posture_label = "NORMAL"

        if horizontal_angle >= self.config.HEAD_TILT_DANGER_ANGLE_DEGREES:
            dangerous_posture = True
            posture_label = "TETE PENCHEE A DROITE" if avg_tilt > 0 else "TETE PENCHEE A GAUCHE"
        elif (
            self.baseline_chin_distance
            and nose_drop_ratio <= self.config.HEAD_RAISE_MAX_NOSE_MOVEMENT_RATIO
            and distance_ratio >= self.config.HEAD_RAISE_MAX_DISTANCE_RATIO
        ):
            dangerous_posture = True
            posture_label = "TETE TROP RELEVEE"
        elif face_bbox is not None and frame_size is not None:
            frame_width, frame_height = frame_size
            x_min, y_min, x_max, y_max = face_bbox
            face_width = max(1, x_max - x_min)
            face_height = max(1, y_max - y_min)
            face_area_ratio = (face_width * face_height) / max(frame_width * frame_height, 1)
            nose_relative_x = (nose_position[0] - x_min) / face_width
            nose_relative_y = (nose_position[1] - y_min) / face_height

            if (
                horizontal_angle >= self.config.HEAD_STRONG_TURN_MIN_ANGLE_DEGREES
                and nose_relative_x <= self.config.HEAD_STRONG_TURN_NOSE_EDGE_RATIO
            ):
                dangerous_posture = True
                posture_label = "TETE TOURNEE COMPLETEMENT A DROITE"
            elif (
                horizontal_angle >= self.config.HEAD_STRONG_TURN_MIN_ANGLE_DEGREES
                and nose_relative_x >= (1.0 - self.config.HEAD_STRONG_TURN_NOSE_EDGE_RATIO)
            ):
                dangerous_posture = True
                posture_label = "TETE TOURNEE COMPLETEMENT A GAUCHE"
            elif (
                self.baseline_chin_distance
                and distance_ratio <= self.config.HEAD_DOWN_TOP_VISIBLE_DISTANCE_RATIO
                and nose_relative_y >= self.config.HEAD_DOWN_TOP_VISIBLE_NOSE_RATIO
            ):
                dangerous_posture = True
                posture_label = "TETE TROP BAISSEE"
            elif face_area_ratio >= self.config.FACE_TOO_LARGE_AREA_RATIO:
                dangerous_posture = True
                posture_label = "CONDUCTEUR TROP PROCHE"
            elif left_eye_points and right_eye_points and len(left_eye_points) >= 4 and len(right_eye_points) >= 4:
                left_eye_width = max(abs(left_eye_points[3][0] - left_eye_points[0][0]), 1)
                right_eye_width = max(abs(right_eye_points[3][0] - right_eye_points[0][0]), 1)
                eye_ratio = max(left_eye_width, right_eye_width) / max(min(left_eye_width, right_eye_width), 1)
                if eye_ratio >= self.config.FACE_TURN_EYE_RATIO:
                    dangerous_posture = True
                    posture_label = (
                        "VISAGE TOURNE A GAUCHE"
                        if right_eye_width > left_eye_width
                        else "VISAGE TOURNE A DROITE"
                    )

        if dangerous_posture:
            self.dangerous_posture_frame_count += 1
            self.posture_label = posture_label
        else:
            self.dangerous_posture_frame_count = max(0, self.dangerous_posture_frame_count - 2)
            if self.dangerous_posture_frame_count == 0:
                self.posture_label = "NORMAL"

        self.dangerous_posture_persistent = (
            self.dangerous_posture_frame_count >= self.head_position_frames_required
        )

        if not self.dangerous_posture_persistent and self.dangerous_posture_frame_count == 0:
            self.posture_label = "NORMAL"

        return self.head_drop_persistent, self.dangerous_posture_persistent, avg_tilt, self.posture_label
    
    def reset(self):
        """Reset all tracking variables"""
        self.head_drop_frame_count = 0
        self.head_drop_confirmed = False
        self.head_drop_persistent = False
        self.dangerous_posture_frame_count = 0
        self.dangerous_posture_persistent = False
        self.posture_label = "NORMAL"
        self.nose_chin_distances.clear()
        self.baseline_distance_history.clear()
        self.tilt_history.clear()
        self.prev_nose_y = None
        self.prev_chin_y = None
        self.prev_nose_x = None
        self.prev_chin_x = None
        self.vertical_movement = 0.0
        self.max_head_drop = 0.0
        self.baseline_nose_y = None
        self.baseline_chin_distance = None
    
    def get_statistics(self) -> dict:
        """
        Get head pose statistics.
        
        Returns:
            Dictionary with head pose statistics
        """
        return {
            'head_drop_frames': self.head_drop_frame_count,
            'head_drop_confirmed': self.head_drop_confirmed,
            'head_drop_persistent': self.head_drop_persistent,
            'dangerous_posture_frames': self.dangerous_posture_frame_count,
            'dangerous_posture_persistent': self.dangerous_posture_persistent,
            'posture_label': self.posture_label,
            'vertical_movement': self.vertical_movement,
            'max_head_drop': self.max_head_drop
        }

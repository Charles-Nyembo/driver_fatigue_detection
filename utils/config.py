"""
Configuration file for driver fatigue detection system.
Contains all thresholds, parameters, and system settings.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionConfig:
    """Centralized configuration for detection parameters."""

    # Eye detection parameters
    EAR_THRESHOLD: float = 0.22
    EAR_CONSECUTIVE_FRAMES: int = 12
    EAR_SMOOTHING_FACTOR: float = 0.25
    EAR_HISTORY_SIZE: int = 20
    EYE_BASELINE_FRAMES: int = 20
    EYE_MIN_VALID_WIDTH: float = 8.0
    EYE_MAX_ASYMMETRY: float = 0.08
    EYE_CLOSED_BASELINE_RATIO: float = 0.72

    # Mouth detection parameters
    MAR_THRESHOLD: float = 0.48
    YAWN_CONSECUTIVE_FRAMES: int = 10
    MAR_SMOOTHING_FACTOR: float = 0.25
    MAR_HISTORY_SIZE: int = 20
    MOUTH_BASELINE_FRAMES: int = 45
    MOUTH_OPEN_CONSECUTIVE_FRAMES: int = 4
    MOUTH_OPEN_MIN_RELATIVE: float = 1.45

    # Yawn detection thresholds
    YAWN_MIN_MAR: float = 0.48
    YAWN_MIN_HEIGHT_MULTIPLIER: float = 2.8
    YAWN_MIN_RELATIVE_OPENING: float = 2.0

    # Head pose parameters
    HEAD_DROP_THRESHOLD: float = 0.12
    HEAD_DROP_CONSECUTIVE_FRAMES: int = 12
    HEAD_DROP_HISTORY_SIZE: int = 15
    HEAD_BASELINE_FRAMES: int = 20
    HEAD_POSITION_HOLD_SECONDS: float = 3.0
    HEAD_DROP_MIN_DISTANCE_RATIO: float = 0.68
    HEAD_DROP_MIN_NOSE_MOVEMENT_RATIO: float = 0.34
    HEAD_TILT_DANGER_ANGLE_DEGREES: float = 80.0
    HEAD_RAISE_MAX_NOSE_MOVEMENT_RATIO: float = -0.22
    HEAD_RAISE_MAX_DISTANCE_RATIO: float = 1.22
    HEAD_STRONG_TURN_NOSE_EDGE_RATIO: float = 0.30
    HEAD_STRONG_TURN_MIN_ANGLE_DEGREES: float = 18.0
    HEAD_DOWN_TOP_VISIBLE_NOSE_RATIO: float = 0.60
    HEAD_DOWN_TOP_VISIBLE_DISTANCE_RATIO: float = 0.78
    FACE_OFF_CENTER_RATIO: float = 0.22
    FACE_TOO_SMALL_AREA_RATIO: float = 0.06
    FACE_TOO_LARGE_AREA_RATIO: float = 0.40
    FACE_TURN_EYE_RATIO: float = 2.2
    NO_FACE_HOLD_SECONDS: float = 3.0
    PRESENCE_HOLD_SECONDS: float = 2.0
    PRESENCE_DISTRACTION_LOW_CENTER_RATIO: float = 0.62
    PRESENCE_DISTRACTION_SIDE_CENTER_RATIO: float = 0.22
    PRESENCE_DISTRACTION_MIN_AREA_RATIO: float = 0.08
    PRESENCE_PROFILE_MAX_WIDTH_HEIGHT_RATIO: float = 0.72
    PRESENCE_SLEEP_LOW_CENTER_RATIO: float = 0.58
    PRESENCE_SLEEP_MIN_AREA_RATIO: float = 0.12
    PRESENCE_SLEEP_CENTER_TOLERANCE_RATIO: float = 0.18
    PRESENCE_SLEEP_MIN_WIDTH_HEIGHT_RATIO: float = 0.82

    # Sleep detection parameters
    SLEEP_CONSECUTIVE_FRAMES: int = 40
    SLEEP_SCORE_INCREMENT: float = 6.0
    EYE_CLOSE_SCORE_INCREMENT: float = 3.0
    YAWN_SCORE_INCREMENT: float = 2.5
    HEAD_DROP_SCORE_INCREMENT: float = 1.5
    DANGEROUS_POSTURE_SCORE_INCREMENT: float = 2.0
    NO_FACE_SCORE_INCREMENT: float = 2.5

    # Fatigue scoring
    FATIGUE_SCORE_MAX: float = 100.0
    FATIGUE_SCORE_ALERT_THRESHOLD: float = 45.0
    FATIGUE_SCORE_WARNING_THRESHOLD: float = 20.0
    FATIGUE_DECAY_RATE: float = 0.8
    NORMAL_FRAMES_TO_CLEAR_ALARM: int = 45

    # Camera settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30
    CAMERA_INDEX: int = 0

    # Face detection confidence
    FACE_DETECTION_CONFIDENCE: float = 0.85
    FACE_TRACKING_CONFIDENCE: float = 0.85

    # Alarm settings
    ALARM_BEEP_FREQUENCY: int = 1900
    ALARM_WARNING_FREQUENCY: int = 880
    ALARM_BEEP_DURATION: float = 0.12
    ALARM_WARNING_INTERVAL: float = 3.0

    # Visualization
    FONT_SCALE: float = 0.45
    FONT_THICKNESS: int = 1
    TITLE_FONT_SCALE: float = 0.7
    TITLE_FONT_THICKNESS: int = 2

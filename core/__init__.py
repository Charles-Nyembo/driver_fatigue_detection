"""
Core modules for driver fatigue detection.
"""

__all__ = [
    'FaceDetector',
    'EyeAnalyzer',
    'MouthAnalyzer',
    'HeadPoseAnalyzer',
    'FatigueDetector',
    'FatigueState'
]


def __getattr__(name):
    if name == "FaceDetector":
        from core.face_detector import FaceDetector
        return FaceDetector
    if name == "EyeAnalyzer":
        from core.eye_analyzer import EyeAnalyzer
        return EyeAnalyzer
    if name == "MouthAnalyzer":
        from core.mouth_analyzer import MouthAnalyzer
        return MouthAnalyzer
    if name == "HeadPoseAnalyzer":
        from core.head_pose_analyzer import HeadPoseAnalyzer
        return HeadPoseAnalyzer
    if name in {"FatigueDetector", "FatigueState"}:
        from core.fatigue_detector import FatigueDetector, FatigueState
        return {"FatigueDetector": FatigueDetector, "FatigueState": FatigueState}[name]
    raise AttributeError(f"module 'core' has no attribute {name!r}")

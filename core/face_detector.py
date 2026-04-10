"""
Face detection and landmark extraction using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

from utils.config import DetectionConfig
from utils.logger import SystemLogger

def _resolve_face_mesh_module():
    """Resolve the FaceMesh module across MediaPipe package layouts."""
    try:
        from mediapipe.python.solutions import face_mesh as resolved_face_mesh
        return resolved_face_mesh
    except (ImportError, AttributeError):
        pass

    solutions_module = getattr(mp, "solutions", None)
    if solutions_module is not None and hasattr(solutions_module, "face_mesh"):
        return solutions_module.face_mesh

    raise ImportError(
        "MediaPipe FaceMesh could not be imported. "
        "Activate the project's virtual environment and reinstall mediapipe if needed."
    )


def _resolve_face_detection_module():
    """Resolve the FaceDetection module across MediaPipe package layouts."""
    try:
        from mediapipe.python.solutions import face_detection as resolved_face_detection
        return resolved_face_detection
    except (ImportError, AttributeError):
        pass

    solutions_module = getattr(mp, "solutions", None)
    if solutions_module is not None and hasattr(solutions_module, "face_detection"):
        return solutions_module.face_detection

    raise ImportError(
        "MediaPipe FaceDetection could not be imported. "
        "Activate the project's virtual environment and reinstall mediapipe if needed."
    )


mp_face_mesh = _resolve_face_mesh_module()
mp_face_detection = _resolve_face_detection_module()

class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe FaceMesh.
    Provides high-precision facial landmarks for eye and mouth analysis.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        
        # Initialize MediaPipe FaceMesh
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=max(0.5, self.config.FACE_DETECTION_CONFIDENCE - 0.2),
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.config.FACE_TRACKING_CONFIDENCE
        )
        
        # Landmark indices
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 185]
        self.NOSE_INDICES = [1, 4]  # Nose tip and base
        self.CHIN_INDEX = 152
        
        self.face_landmarks = None
        self.presence_bbox = None
        self.frame_width = 0
        self.frame_height = 0
        
        self.logger.info("FaceDetector initialized successfully")
        
    def detect_presence(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect whether a driver is present in front of the camera.

        Returns:
            Tuple of (presence_detected, presence_bbox)
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        if not results.detections:
            self.presence_bbox = None
            return False, None

        detection = max(
            results.detections,
            key=lambda item: item.score[0] if item.score else 0.0,
        )
        rel_bbox = detection.location_data.relative_bounding_box
        x_min = max(0, int(rel_bbox.xmin * self.frame_width))
        y_min = max(0, int(rel_bbox.ymin * self.frame_height))
        width = max(1, int(rel_bbox.width * self.frame_width))
        height = max(1, int(rel_bbox.height * self.frame_height))
        x_max = min(self.frame_width, x_min + width)
        y_max = min(self.frame_height, y_min + height)
        self.presence_bbox = (x_min, y_min, x_max, y_max)
        return True, self.presence_bbox

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[object], bool]:
        """
        Process a frame and extract face landmarks.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (landmarks, face_detected)
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            self.face_landmarks = results.multi_face_landmarks[0]
            return self.face_landmarks, True
        
        self.face_landmarks = None
        return None, False
    
    def get_eye_landmarks(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Extract left and right eye landmarks.
        
        Returns:
            Tuple of (left_eye_points, right_eye_points)
        """
        if self.face_landmarks is None:
            return [], []
        
        left_eye = []
        right_eye = []
        
        try:
            for idx in self.LEFT_EYE_INDICES:
                point = self.face_landmarks.landmark[idx]
                left_eye.append((
                    int(point.x * self.frame_width),
                    int(point.y * self.frame_height)
                ))
            
            for idx in self.RIGHT_EYE_INDICES:
                point = self.face_landmarks.landmark[idx]
                right_eye.append((
                    int(point.x * self.frame_width),
                    int(point.y * self.frame_height)
                ))
        except Exception as e:
            self.logger.debug(f"Error extracting eye landmarks: {e}")
            return [], []
        
        return left_eye, right_eye
    
    def get_mouth_landmarks(self) -> List[Tuple[int, int]]:
        """
        Extract mouth landmarks.
        
        Returns:
            List of mouth landmark points
        """
        if self.face_landmarks is None:
            return []
        
        mouth = []
        try:
            for idx in self.MOUTH_INDICES:
                point = self.face_landmarks.landmark[idx]
                mouth.append((
                    int(point.x * self.frame_width),
                    int(point.y * self.frame_height)
                ))
        except Exception as e:
            self.logger.debug(f"Error extracting mouth landmarks: {e}")
            return []
        
        return mouth
    
    def get_nose_position(self) -> Optional[Tuple[int, int]]:
        """
        Extract nose tip position.
        
        Returns:
            (x, y) coordinates of nose tip
        """
        if self.face_landmarks is None:
            return None
        
        try:
            point = self.face_landmarks.landmark[self.NOSE_INDICES[0]]
            return (int(point.x * self.frame_width), int(point.y * self.frame_height))
        except Exception:
            return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self.face_detection.close()
        except Exception as exc:
            self.logger.debug(f"Face detection close error: {exc}")
        try:
            self.face_mesh.close()
        except Exception as exc:
            self.logger.debug(f"Face mesh close error: {exc}")
    
    def get_chin_position(self) -> Optional[Tuple[int, int]]:
        """
        Extract chin position.
        
        Returns:
            (x, y) coordinates of chin
        """
        if self.face_landmarks is None:
            return None
        
        try:
            point = self.face_landmarks.landmark[self.CHIN_INDEX]
            return (int(point.x * self.frame_width), int(point.y * self.frame_height))
        except Exception:
            return None
    
    def get_face_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of the face.
        
        Returns:
            (x_min, y_min, x_max, y_max) or None
        """
        if self.face_landmarks is None:
            return None
        
        try:
            x_coords = [lm.x for lm in self.face_landmarks.landmark]
            y_coords = [lm.y for lm in self.face_landmarks.landmark]
            
            x_min = int(min(x_coords) * self.frame_width)
            x_max = int(max(x_coords) * self.frame_width)
            y_min = int(min(y_coords) * self.frame_height)
            y_max = int(max(y_coords) * self.frame_height)
            
            return (x_min, y_min, x_max, y_max)
        except Exception:
            return None

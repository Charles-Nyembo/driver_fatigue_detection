"""
Driver Fatigue Detection System - Main Entry Point
Professional implementation combining eye, mouth, and head pose analysis.
"""

import cv2
import numpy as np
import time
import sys
import signal
from typing import Optional

from utils.config import DetectionConfig
from utils.logger import SystemLogger
from utils.alarm import PersistentAlarm

from core.face_detector import FaceDetector
from core.eye_analyzer import EyeAnalyzer
from core.mouth_analyzer import MouthAnalyzer
from core.head_pose_analyzer import HeadPoseAnalyzer
from core.fatigue_detector import FatigueDetector, FatigueState

from visualization.overlay import VisualizationOverlay


class DriverFatigueSystem:
    """
    Main driver fatigue detection system.
    Integrates all components and manages the main loop.
    """
    
    WINDOW_NAME = "Driver Fatigue Detection System"

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the fatigue detection system.
        
        Args:
            config: System configuration
        """
        self.config = config or DetectionConfig()
        self.logger = SystemLogger()
        self.running = True
        self.simulation_started = False
        self.driving_active = False
        self.mouse_callback_registered = False

        # Initialize components
        self.logger.info("Initializing system components...")
        self.face_detector = FaceDetector(self.config, self.logger)
        self.eye_analyzer = EyeAnalyzer(self.config, self.logger)
        self.mouth_analyzer = MouthAnalyzer(self.config, self.logger)
        self.head_pose_analyzer = HeadPoseAnalyzer(self.config, self.logger)
        self.fatigue_detector = FatigueDetector(self.config, self.logger)
        self.alarm = PersistentAlarm(self.config, self.logger)
        self.visualization = VisualizationOverlay(self.config)
        
        self.cap = None
        
        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.no_face_frame_count = 0
        self.no_face_frames_required = max(1, int(self.config.CAMERA_FPS * self.config.NO_FACE_HOLD_SECONDS))
        self.last_presence_time = 0.0
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logger.info("System initialized successfully")
        
    def _init_camera(self) -> None:
        """Initialize and configure camera."""
        if self.cap is not None and self.cap.isOpened():
            return
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.logger.info(f"Camera initialized: {self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}")

    def _reset_detection_state(self) -> None:
        """Reset all analysis modules when a new driving session starts or stops."""
        self.eye_analyzer.reset()
        self.mouth_analyzer.reset()
        self.head_pose_analyzer.reset()
        self.fatigue_detector.reset()
        self.no_face_frame_count = 0
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.last_presence_time = 0.0

    def _presence_is_recent(self) -> bool:
        """Keep presence active briefly when tracking is temporarily lost."""
        if self.last_presence_time <= 0:
            return False
        return (time.time() - self.last_presence_time) <= self.config.PRESENCE_HOLD_SECONDS

    def _start_driving(self) -> None:
        """Start vehicle simulation and enable face analysis."""
        if self.driving_active:
            return
        if not self.simulation_started:
            self.logger.info("Start the simulation before starting the vehicle")
            return
        self._reset_detection_state()
        self._init_camera()
        self.driving_active = True
        self.logger.info("Driving mode started - face analysis enabled")

    def _start_simulation(self) -> None:
        """Launch the simulation without starting vehicle analysis."""
        if self.simulation_started:
            return
        self.simulation_started = True
        self.driving_active = False
        self._reset_detection_state()
        if self.alarm.is_active():
            self.alarm.stop_alarm()
        self.logger.info("Simulation launched - waiting for vehicle start")

    def _stop_driving(self) -> None:
        """Stop vehicle simulation and disable face analysis."""
        if not self.simulation_started and not self.driving_active:
            return
        self.driving_active = False
        if self.alarm.is_active():
            self.alarm.stop_alarm()
        self._reset_detection_state()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.logger.info("Driving mode stopped - face analysis disabled")

    def _register_mouse_callback(self) -> None:
        """Attach a mouse callback once the window exists."""
        if self.mouse_callback_registered:
            return
        cv2.setMouseCallback(self.WINDOW_NAME, self._handle_mouse_event)
        self.mouse_callback_registered = True

    def _step_regions_for_current_view(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Return current clickable step regions based on the active layout."""
        origin_x = self.config.CAMERA_WIDTH if self.driving_active else 0
        return self.visualization.get_step_regions(origin_x=origin_x, origin_y=0)

    def _handle_mouse_event(self, event, x, y, flags, param) -> None:
        """Handle clicks on the step cards."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for index, ((x1, y1), (x2, y2)) in enumerate(self._step_regions_for_current_view(), start=1):
            if x1 <= x <= x2 and y1 <= y <= y2:
                if index == 1:
                    self._start_simulation()
                elif index == 2:
                    try:
                        self._start_driving()
                    except Exception as exc:
                        self.logger.error(f"Unable to start driving mode: {exc}")
                elif index == 5:
                    self._stop_driving()
                break

    def _build_idle_frame(self) -> np.ndarray:
        """Create the home simulation interface shown before driving starts."""
        frame = np.zeros((self.config.CAMERA_HEIGHT, 1000, 3), dtype=np.uint8)
        self.visualization.draw_dashboard(
            frame,
            simulation_started=self.simulation_started,
            driving_active=self.driving_active,
            camera_ready=False,
            monochrome=True,
        )
        return frame
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle keyboard interrupts."""
        self.logger.info("Received interrupt signal")
        self.running = False

    def _get_status_text_override(
        self,
        eyes_closed: bool,
        yawning: bool,
        head_drop: bool,
        dangerous_posture: bool,
        posture_label: str,
        sleep_detected: bool,
    ) -> Optional[str]:
        """Return a custom top-bar label for distraction-related driving behavior."""
        sleep_posture_labels = {
            "TETE TROP BAISSEE",
            "CONDUCTEUR ENDORMI SUR LE VOLANT",
        }
        distraction_labels = {
            "TETE PENCHEE A DROITE",
            "TETE PENCHEE A GAUCHE",
            "TETE TOURNEE COMPLETEMENT A DROITE",
            "TETE TOURNEE COMPLETEMENT A GAUCHE",
            "VISAGE TOURNE A GAUCHE",
            "VISAGE TOURNE A DROITE",
        }

        if dangerous_posture and posture_label in sleep_posture_labels and not sleep_detected and not eyes_closed:
            return "SOMNOLENCE SUSPECTEE"

        is_distraction = head_drop or (dangerous_posture and posture_label in distraction_labels)
        if is_distraction and not sleep_detected and not eyes_closed and not yawning:
            return "CONDUCTEUR NON CONCENTRE AU VOLANT"
        return None

    def _infer_presence_only_sleep_posture(
        self,
        presence_bbox: Optional[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
    ) -> Optional[str]:
        """Infer likely sleep posture from a coarse presence box when face mesh is unavailable."""
        if presence_bbox is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = presence_bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x_ratio = ((x1 + x2) / 2.0) / max(frame_width, 1)
        center_y_ratio = ((y1 + y2) / 2.0) / max(frame_height, 1)
        box_area_ratio = (box_width * box_height) / max(frame_width * frame_height, 1)
        width_height_ratio = box_width / box_height

        is_centered = abs(center_x_ratio - 0.5) <= self.config.PRESENCE_SLEEP_CENTER_TOLERANCE_RATIO
        if (
            is_centered
            and center_y_ratio >= self.config.PRESENCE_SLEEP_LOW_CENTER_RATIO
            and box_area_ratio >= self.config.PRESENCE_SLEEP_MIN_AREA_RATIO
            and width_height_ratio >= self.config.PRESENCE_SLEEP_MIN_WIDTH_HEIGHT_RATIO
        ):
            return "CONDUCTEUR ENDORMI SUR LE VOLANT"
        return None

    def _infer_presence_only_distraction(
        self,
        presence_bbox: Optional[tuple[int, int, int, int]],
        frame_shape: tuple[int, int, int],
    ) -> Optional[str]:
        """Infer obvious distraction postures from the coarse presence box alone."""
        if presence_bbox is None:
            return None

        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = presence_bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        center_x_ratio = ((x1 + x2) / 2.0) / max(frame_width, 1)
        center_y_ratio = ((y1 + y2) / 2.0) / max(frame_height, 1)
        box_area_ratio = (box_width * box_height) / max(frame_width * frame_height, 1)
        width_height_ratio = box_width / box_height

        if (
            center_y_ratio >= self.config.PRESENCE_DISTRACTION_LOW_CENTER_RATIO
            and box_area_ratio >= self.config.PRESENCE_DISTRACTION_MIN_AREA_RATIO
        ):
            return "CONDUCTEUR AFFAISSE SUR LE VOLANT"
        if (
            center_x_ratio <= self.config.PRESENCE_DISTRACTION_SIDE_CENTER_RATIO
            and width_height_ratio <= self.config.PRESENCE_PROFILE_MAX_WIDTH_HEIGHT_RATIO
        ):
            return "CONDUCTEUR TOURNE COMPLETEMENT A DROITE"
        if (
            center_x_ratio >= (1.0 - self.config.PRESENCE_DISTRACTION_SIDE_CENTER_RATIO)
            and width_height_ratio <= self.config.PRESENCE_PROFILE_MAX_WIDTH_HEIGHT_RATIO
        ):
            return "CONDUCTEUR TOURNE COMPLETEMENT A GAUCHE"
        return None
    
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame through all detection pipelines.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with visualization
        """
        presence_detected, presence_bbox = self.face_detector.detect_presence(frame)

        if not presence_detected:
            self.no_face_frame_count += 1
            if self.alarm.is_active():
                self.alarm.stop_alarm()
            if not self.driving_active:
                cv2.putText(frame, "AUCUNE PRESENCE CONDUCTEUR DETECTEE", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2)
                cv2.putText(frame, "Analyse presence et tete 3D en pause", (30, 82),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
            return frame

        self.no_face_frame_count = 0
        self.last_presence_time = time.time()
        if presence_bbox is not None:
            x1, y1, x2, y2 = presence_bbox
        cv2.putText(frame, "PRESENCE CONDUCTEUR DETECTEE", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 130), 2)

        # Face mesh / head 3D detection
        landmarks, face_detected = self.face_detector.process_frame(frame)
        
        if not face_detected:
            inferred_sleep = self._infer_presence_only_sleep_posture(presence_bbox, frame.shape)
            inferred_distraction = self._infer_presence_only_distraction(presence_bbox, frame.shape)
            if inferred_sleep is not None:
                self.alarm.start_alarm()
            elif self.alarm.is_active():
                self.alarm.stop_alarm()
            if not self.driving_active:
                if inferred_sleep is not None:
                    self.visualization.draw_status_bar(
                        frame,
                        FatigueState.ALERT,
                        100.0,
                        status_text_override="SOMNOLENCE SUSPECTEE",
                    )
                    self.visualization.draw_detection_indicators(
                        frame,
                        False,
                        False,
                        False,
                        True,
                        True,
                        inferred_sleep,
                    )
                    self.visualization.draw_visual_alert(frame, FatigueState.ALERT)
                elif inferred_distraction is not None:
                    self.visualization.draw_status_bar(
                        frame,
                        FatigueState.FATIGUE,
                        0.0,
                        status_text_override="CONDUCTEUR NON CONCENTRE AU VOLANT",
                    )
                    self.visualization.draw_detection_indicators(
                        frame,
                        False,
                        False,
                        False,
                        False,
                        True,
                        inferred_distraction,
                    )
                cv2.putText(frame, "SUIVI TETE 3D INDISPONIBLE", (30, 82),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 190, 255), 2)
                detail_text = (
                    "Posture de sommeil detectee meme sans tete 3D complete"
                    if inferred_sleep is not None
                    else "Posture extreme detectee meme sans tete 3D complete"
                    if inferred_distraction is not None
                    else "Le conducteur est present mais la tete n est pas assez visible"
                )
                cv2.putText(frame, detail_text, (30, 114),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2)
            return frame

        cv2.putText(frame, "ANALYSE TETE CONDUCTEUR 3D ACTIVE", (30, 82),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2)
        
        # Extract landmarks
        left_eye, right_eye = self.face_detector.get_eye_landmarks()
        mouth = self.face_detector.get_mouth_landmarks()
        nose = self.face_detector.get_nose_position()
        chin = self.face_detector.get_chin_position()
        face_bbox = self.face_detector.get_face_bbox()
        
        # Eye analysis
        ear, eyes_closed = self.eye_analyzer.update(left_eye, right_eye)
        
        # Mouth analysis
        mar, yawning = self.mouth_analyzer.update(mouth)
        
        # Head pose analysis
        head_drop, dangerous_posture, tilt, posture_label = self.head_pose_analyzer.update(
            nose,
            chin,
            face_bbox=face_bbox,
            left_eye_points=left_eye,
            right_eye_points=right_eye,
            frame_size=(frame.shape[1], frame.shape[0]),
        )
        
        # Fatigue detection
        state, metrics = self.fatigue_detector.update(
            eyes_closed, yawning, head_drop, dangerous_posture, posture_label
        )
        
        # Alarm management
        if self.fatigue_detector.should_trigger_alarm():
            self.alarm.start_alarm()
        elif not self.fatigue_detector.should_trigger_alarm():
            self.alarm.stop_alarm()
        
        # Play warning if needed
        if self.fatigue_detector.should_trigger_warning():
            self.alarm.play_warning()
        
        # Visualization
        status_text_override = self._get_status_text_override(
            eyes_closed,
            yawning,
            head_drop,
            metrics['dangerous_posture'],
            metrics['posture_label'],
            metrics['sleep_detected'],
        )

        # Determine whether a detection event is active (to toggle bbox/marker color)
        detection_active = bool(
            eyes_closed
            or yawning
            or head_drop
            or metrics.get('sleep_detected')
            or metrics.get('dangerous_posture')
            or state == FatigueState.ALERT
            or self.fatigue_detector.should_trigger_alarm()
        )

        # Draw/redraw face bounding box in green (ok) or red (detection)
        bbox = face_bbox or presence_bbox
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            box_color = (0, 0, 255) if detection_active else (50, 220, 130)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 3)

        # When driving, draw all face mesh landmarks as points
        if self.driving_active and self.face_detector.face_landmarks is not None:
            pts = [
                (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                for lm in self.face_detector.face_landmarks.landmark
            ]
            # keep landmark points green regardless of detection state
            dot_color = (88, 224, 166)
            self.visualization.draw_full_landmarks(frame, pts, color=dot_color)

        # Always display core detection labels and fatigue progress directly
        # on the camera feed so the surveillance view shows critical info.
        self.visualization.draw_detection_indicators(
            frame,
            eyes_closed,
            yawning,
            head_drop,
            metrics['sleep_detected'],
            metrics['dangerous_posture'],
            metrics['posture_label'],
        )

        # Draw the fatigue progress bar on the image itself.
        self.visualization.draw_fatigue_bar(frame, metrics['fatigue_score'], state)

        # Visual alert (flashing overlay) if in ALERT state
        self.visualization.draw_visual_alert(frame, state)

        # If there's an override status message (e.g., "CONDUCTEUR NON CONCENTRE..."),
        # show it as an unobtrusive banner on the camera frame.
        if status_text_override:
            # place near top-left under any window chrome
            text_color = self.visualization.COLORS.get('alert', (30, 90, 200))
            bg_color = (24, 30, 36)
            self.visualization.draw_text_with_background(frame, status_text_override, (20, 56), text_color, 0.7, 2, bg_color)

        # Show a concise alarm badge if active
        self.visualization.draw_alarm_status(frame, self.alarm.is_active())
        
        return frame
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def run(self) -> None:
        """
        Main execution loop.
        """
        self.logger.info("Starting main detection loop")
        self.logger.info("Click step 1 to launch simulation, step 2 to start driving, step 5 to stop, Q to quit")

        try:
            cv2.namedWindow(self.WINDOW_NAME)
            self._register_mouse_callback()
            while self.running:
                if self.driving_active:
                    if self.cap is None or not self.cap.isOpened():
                        try:
                            self._init_camera()
                        except Exception as exc:
                            self.logger.error(f"Camera start error: {exc}")
                            self._stop_driving()
                            continue

                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.error("Failed to read frame from camera")
                        self._stop_driving()
                        continue

                    frame = cv2.flip(frame, 1)

                    try:
                        processed_frame = self._process_frame(frame)
                    except Exception as exc:
                        self.logger.error(f"Frame processing error: {exc}")
                        processed_frame = frame
                        cv2.putText(
                            processed_frame,
                            "PROCESSING ERROR - CHECK LOGS",
                            (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                    self._update_fps()
                    self.visualization.draw_fps(processed_frame, self.fps)
                    output_frame = self.visualization.compose_driving_view(
                        processed_frame,
                        simulation_started=self.simulation_started,
                        driving_active=True,
                        camera_ready=True,
                    )
                else:
                    output_frame = self._build_idle_frame()

                cv2.imshow(self.WINDOW_NAME, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('1'),):
                    self._start_simulation()
                elif key in (ord('2'),):
                    try:
                        self._start_driving()
                    except Exception as exc:
                        self.logger.error(f"Unable to start driving mode: {exc}")
                elif key in (ord('5'), ord('s'), ord('S')):
                    self._stop_driving()
                elif key == ord('q') or key == ord('Q'):
                    self.running = False
                    break
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Shutting down system...")
        
        # Stop alarm
        if self.alarm.is_active():
            self.alarm.stop_alarm()

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        self.face_detector.close()
        
        # Close windows
        cv2.destroyAllWindows()
        
        self.logger.info("System shutdown complete")


def main():
    """Entry point."""
    try:
        config = DetectionConfig()
        system = DriverFatigueSystem(config)
        system.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

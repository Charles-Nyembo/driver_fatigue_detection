import logging
import unittest

from core.eye_analyzer import EyeAnalyzer
from core.fatigue_detector import FatigueDetector, FatigueState
from core.head_pose_analyzer import HeadPoseAnalyzer
from core.mouth_analyzer import MouthAnalyzer
from utils.config import DetectionConfig
from utils.logger import SystemLogger


def make_eye_points(width=20, open_ratio=0.35, x_offset=0, y_offset=0):
    half_height = width * open_ratio / 2
    return [
        (x_offset, y_offset),
        (x_offset + 5, int(y_offset - half_height)),
        (x_offset + 15, int(y_offset - half_height)),
        (x_offset + width, y_offset),
        (x_offset + 15, int(y_offset + half_height)),
        (x_offset + 5, int(y_offset + half_height)),
    ]


def make_mouth_points(width=40, open_height=8, x_offset=0, y_offset=0):
    left_x = x_offset
    right_x = x_offset + width
    center_x = x_offset + width // 2
    top_y = y_offset - open_height // 2
    bottom_y = y_offset + open_height // 2
    return [
        (left_x, y_offset),
        (x_offset + 4, top_y + 1),
        (x_offset + 8, top_y),
        (x_offset + 12, top_y + 1),
        (x_offset + 16, top_y + 2),
        (center_x, top_y),
        (x_offset + 24, top_y + 2),
        (x_offset + 28, top_y + 1),
        (x_offset + 32, top_y),
        (x_offset + 36, top_y + 1),
        (right_x, y_offset),
        (x_offset + 36, bottom_y - 1),
        (x_offset + 32, bottom_y),
        (x_offset + 28, bottom_y - 1),
        (x_offset + 24, bottom_y - 2),
        (center_x, bottom_y),
        (x_offset + 16, bottom_y - 2),
        (x_offset + 12, bottom_y - 1),
        (x_offset + 8, bottom_y),
        (x_offset + 4, bottom_y - 1),
    ]


class TestDetectionConfig(unittest.TestCase):
    def test_instances_are_independent(self):
        first = DetectionConfig()
        second = DetectionConfig()

        first.EAR_THRESHOLD = 0.1

        self.assertEqual(second.EAR_THRESHOLD, 0.22)


class TestLogger(unittest.TestCase):
    def test_logger_does_not_duplicate_handlers(self):
        logger = logging.getLogger(SystemLogger.LOGGER_NAME)
        for handler in list(logger.handlers):
            handler.close()
        logger.handlers.clear()

        first = SystemLogger()
        handler_count = len(first.logger.handlers)

        second = SystemLogger()

        self.assertEqual(len(second.logger.handlers), handler_count)
        self.assertGreaterEqual(handler_count, 1)


class TestFatigueDetector(unittest.TestCase):
    def test_alarm_is_persistent_until_normal_frames_complete(self):
        config = DetectionConfig(
            FATIGUE_SCORE_ALERT_THRESHOLD=10,
            FATIGUE_SCORE_WARNING_THRESHOLD=5,
            EYE_CLOSE_SCORE_INCREMENT=10,
            FATIGUE_DECAY_RATE=6,
            NORMAL_FRAMES_TO_CLEAR_ALARM=3,
        )
        detector = FatigueDetector(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        state, metrics = detector.update(True, False, False)
        self.assertEqual(state, FatigueState.ALERT)
        self.assertTrue(metrics["alarm_active"])

        for expected_counter in (1, 2, 3):
            state, metrics = detector.update(False, False, False)
            self.assertEqual(state, FatigueState.NORMAL)
            if expected_counter < 3:
                self.assertTrue(metrics["alarm_active"])
                self.assertEqual(metrics["normal_frames"], expected_counter)
            else:
                self.assertFalse(metrics["alarm_active"])
                self.assertEqual(metrics["normal_frames"], 0)

    def test_reset_clears_runtime_flags(self):
        detector = FatigueDetector(logger=SystemLogger(log_level=logging.CRITICAL))
        detector.update(True, True, True)
        detector.reset()

        metrics = detector.get_metrics()
        self.assertEqual(metrics["current_state"], FatigueState.NORMAL.value)
        self.assertFalse(metrics["eyes_closed"])
        self.assertFalse(metrics["yawning"])
        self.assertFalse(metrics["head_dropping"])
        self.assertFalse(metrics["alarm_active"])


class TestAnalyzers(unittest.TestCase):
    def test_eye_closure_requires_both_eyes_and_stability(self):
        config = DetectionConfig(
            EAR_CONSECUTIVE_FRAMES=3,
            EAR_HISTORY_SIZE=6,
            EYE_BASELINE_FRAMES=4,
            EAR_SMOOTHING_FACTOR=0.6,
        )
        analyzer = EyeAnalyzer(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        open_eye = make_eye_points(open_ratio=0.35)
        closed_eye = make_eye_points(open_ratio=0.08)

        for _ in range(6):
            analyzer.update(open_eye, open_eye)

        for _ in range(4):
            _, detected = analyzer.update(closed_eye, open_eye)
            self.assertFalse(detected)

        for _ in range(6):
            _, detected = analyzer.update(closed_eye, closed_eye)

        self.assertTrue(detected)
        self.assertGreater(analyzer.open_eye_baseline, config.EAR_THRESHOLD)

    def test_mouth_open_does_not_immediately_mean_yawn(self):
        config = DetectionConfig(
            MOUTH_BASELINE_FRAMES=4,
            YAWN_CONSECUTIVE_FRAMES=3,
            MOUTH_OPEN_CONSECUTIVE_FRAMES=2,
            MAR_SMOOTHING_FACTOR=0.7,
        )
        analyzer = MouthAnalyzer(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        neutral_mouth = make_mouth_points(open_height=8)
        open_mouth = make_mouth_points(open_height=22)
        yawn_mouth = make_mouth_points(open_height=34)

        for _ in range(4):
            analyzer.update(neutral_mouth)

        for _ in range(2):
            _, yawning = analyzer.update(open_mouth)
            self.assertFalse(yawning)

        self.assertTrue(analyzer.mouth_open_confirmed)

        for _ in range(3):
            _, yawning = analyzer.update(yawn_mouth)

        self.assertTrue(yawning)
        self.assertGreater(analyzer.last_relative_opening, config.YAWN_MIN_RELATIVE_OPENING)

    def test_head_drop_requires_sustained_pose_change(self):
        config = DetectionConfig(
            HEAD_DROP_CONSECUTIVE_FRAMES=3,
            HEAD_BASELINE_FRAMES=4,
        )
        analyzer = HeadPoseAnalyzer(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        for _ in range(5):
            detected, _, _, _ = analyzer.update((50, 50), (50, 110))
            self.assertFalse(detected)

        for nose_y in (71, 72):
            detected, _, _, _ = analyzer.update((50, nose_y), (50, 110))
            self.assertFalse(detected)

        detected, _, _, _ = analyzer.update((50, 73), (50, 110))
        self.assertTrue(detected)
        self.assertGreater(analyzer.vertical_movement, config.HEAD_DROP_MIN_NOSE_MOVEMENT_RATIO)

    def test_mouth_reset_clears_calibration_state(self):
        analyzer = MouthAnalyzer(logger=SystemLogger(log_level=logging.CRITICAL))
        analyzer.is_calibrated = True
        analyzer.mouth_baseline = 0.3
        analyzer.baseline_height = 10
        analyzer.baseline_history.extend([0.2, 0.25])
        analyzer.baseline_height_history.extend([5, 6])
        analyzer.mar_history.extend([0.2, 0.3])
        analyzer.mouth_height_history.extend([8])
        analyzer.mouth_width_history.extend([12])
        analyzer.reset()

        self.assertFalse(analyzer.is_calibrated)
        self.assertIsNone(analyzer.mouth_baseline)
        self.assertEqual(analyzer.baseline_height, 0)
        self.assertEqual(len(analyzer.baseline_history), 0)
        self.assertEqual(len(analyzer.baseline_height_history), 0)
        self.assertEqual(len(analyzer.mar_history), 0)
        self.assertEqual(len(analyzer.mouth_height_history), 0)
        self.assertEqual(len(analyzer.mouth_width_history), 0)

    def test_eye_reset_clears_per_eye_validation_state(self):
        analyzer = EyeAnalyzer(logger=SystemLogger(log_level=logging.CRITICAL))
        analyzer.consecutive_valid_left = 4
        analyzer.consecutive_valid_right = 2
        analyzer.last_valid_ear_left = 0.2
        analyzer.last_valid_ear_right = 0.25
        analyzer.eye_close_frame_count = 3
        analyzer.reset()

        self.assertEqual(analyzer.consecutive_valid_left, 0)
        self.assertEqual(analyzer.consecutive_valid_right, 0)
        self.assertEqual(analyzer.last_valid_ear_left, 0.35)
        self.assertEqual(analyzer.last_valid_ear_right, 0.35)
        self.assertEqual(analyzer.eye_close_frame_count, 0)

    def test_head_pose_handles_missing_points_without_crashing(self):
        analyzer = HeadPoseAnalyzer(logger=SystemLogger(log_level=logging.CRITICAL))
        detected, _, tilt, _ = analyzer.update(None, None)

        self.assertFalse(detected)
        self.assertEqual(tilt, 0.0)

    def test_head_pose_flags_complete_side_turn(self):
        config = DetectionConfig(
            CAMERA_FPS=1,
            HEAD_POSITION_HOLD_SECONDS=1.0,
            HEAD_STRONG_TURN_MIN_ANGLE_DEGREES=10.0,
            HEAD_STRONG_TURN_NOSE_EDGE_RATIO=0.30,
        )
        analyzer = HeadPoseAnalyzer(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        _, dangerous, _, label = analyzer.update(
            (20, 55),
            (45, 110),
            face_bbox=(0, 0, 100, 120),
            left_eye_points=make_eye_points(),
            right_eye_points=make_eye_points(),
            frame_size=(200, 200),
        )

        self.assertTrue(dangerous)
        self.assertEqual(label, "TETE TOURNEE COMPLETEMENT A DROITE")

    def test_head_pose_flags_top_of_head_visible(self):
        config = DetectionConfig(
            CAMERA_FPS=1,
            HEAD_POSITION_HOLD_SECONDS=1.0,
            HEAD_BASELINE_FRAMES=3,
            HEAD_DOWN_TOP_VISIBLE_DISTANCE_RATIO=0.85,
            HEAD_DOWN_TOP_VISIBLE_NOSE_RATIO=0.60,
        )
        analyzer = HeadPoseAnalyzer(config=config, logger=SystemLogger(log_level=logging.CRITICAL))

        for _ in range(3):
            analyzer.update(
                (50, 45),
                (50, 110),
                face_bbox=(0, 0, 100, 120),
                left_eye_points=make_eye_points(),
                right_eye_points=make_eye_points(),
                frame_size=(200, 200),
            )

        _, dangerous, _, label = analyzer.update(
            (50, 76),
            (50, 120),
            face_bbox=(0, 0, 100, 120),
            left_eye_points=make_eye_points(),
            right_eye_points=make_eye_points(),
            frame_size=(200, 200),
        )

        self.assertTrue(dangerous)
        self.assertEqual(label, "TETE TROP BAISSEE")


if __name__ == "__main__":
    unittest.main()

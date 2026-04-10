"""
Visualization overlay for driver fatigue detection system.
Renders dashboard cards, status banners, and real-time metrics.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple

from core.fatigue_detector import FatigueState
from utils.config import DetectionConfig


class VisualizationOverlay:
    """
    Handles all visualization elements including status text,
    progress bars, and visual alerts.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.step_regions = [
            ((28, 196), (332, 250)),
            ((28, 262), (332, 316)),
            ((28, 328), (332, 382)),
            ((28, 394), (332, 448)),
            ((28, 460), (332, 514)),
        ]
        self.dashboard_width = 420

        # Modern color palette (BGR)
        self.COLORS = {
            "normal": (54, 215, 170),    # soft teal
            "fatigue": (40, 140, 200),    # muted blue
            "alert": (30, 120, 200),      # vivid alert
            "warning": (36, 140, 200),
            "text": (240, 245, 250),
            "text_dim": (160, 172, 180),
            "panel": (22, 28, 36),
            "panel_alt": (28, 34, 44),
            "background": (10, 16, 24),
            "border": (60, 74, 96),
            "accent": (10, 150, 200),
            "shadow": (0, 0, 0),
            "camera_label": (210, 225, 235),
        }

        # Typography / sizing tuning
        self.font_large = cv2.FONT_HERSHEY_DUPLEX
        self.font_medium = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN
        self.base_padding = 12
        # target design resolution: ensure text fits in this window
        self.target_resolution = (1920, 1200)

        self.STATE_TEXT = {
            FatigueState.NORMAL: "VIGILANCE STABLE",
            FatigueState.FATIGUE: "FATIGUE LEGERE",
            FatigueState.ALERT: "ALERTE SOMNOLENCE",
        }

    def _draw_gradient_background(
        self,
        frame: np.ndarray,
        top_color: Tuple[int, int, int],
        bottom_color: Tuple[int, int, int],
    ) -> None:
        """Fill the frame with a soft vertical gradient."""
        height, width = frame.shape[:2]
        overlay = np.zeros_like(frame)
        for row in range(height):
            blend = row / max(height - 1, 1)
            color = tuple(
                int(top_color[index] * (1.0 - blend) + bottom_color[index] * blend)
                for index in range(3)
            )
            cv2.line(overlay, (0, row), (width, row), color, 1)
        cv2.addWeighted(overlay, 1.0, frame, 0.0, 0, frame)

    def _draw_panel(
        self,
        frame: np.ndarray,
        top_left: Tuple[int, int],
        bottom_right: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.92,
        border_color: Optional[Tuple[int, int, int]] = None,
        border_thickness: int = 1,
    ) -> None:
        """Draw a soft card panel with a subtle shadow."""
        x1, y1 = top_left
        x2, y2 = bottom_right

        shadow = frame.copy()
        cv2.rectangle(shadow, (x1 + 5, y1 + 5), (x2 + 5, y2 + 5), self.COLORS["shadow"], -1)
        cv2.addWeighted(shadow, 0.18, frame, 0.82, 0, frame)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        border = border_color or self.COLORS["border"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, border_thickness)

    def _draw_badge(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        scale: float = 0.55,
    ) -> Tuple[int, int]:
        """Draw a compact label badge and return its size."""
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            1,
        )
        x, y = origin
        padding_x = 12
        padding_y = 8
        self._draw_panel(
            frame,
            (x, y),
            (x + text_width + padding_x * 2, y + text_height + padding_y * 2),
            bg_color,
            alpha=0.96,
            border_color=bg_color,
        )
        cv2.putText(
            frame,
            text,
            (x + padding_x, y + text_height + padding_y - 2),
            self.font_large,
            scale,
            text_color,
            1,
            cv2.LINE_AA,
        )
        return text_width + padding_x * 2, text_height + padding_y * 2 + baseline

    def _draw_progress_line(
        self,
        frame: np.ndarray,
        top_left: Tuple[int, int],
        width: int,
        value: float,
        color: Tuple[int, int, int],
        background_color: Tuple[int, int, int] = (58, 67, 78),
    ) -> None:
        """Draw a thin progress line used in cards."""
        x, y = top_left
        value = max(0.0, min(1.0, value))
        cv2.rectangle(frame, (x, y), (x + width, y + 8), background_color, -1)
        fill_width = max(0, int(width * value))
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + 8), color, -1)

    def _draw_metric_card(
        self,
        frame: np.ndarray,
        title: str,
        value_text: str,
        subtitle: str,
        top_left: Tuple[int, int],
        size: Tuple[int, int],
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Draw a compact metric card on top of the camera frame."""
        x, y = top_left
        w, h = size
        self._draw_panel(frame, (x, y), (x + w, y + h), self.COLORS["panel"], alpha=0.92)
        cv2.putText(frame, title, (x + 14, y + 26), self.font_medium, 0.45, self.COLORS["text_dim"], 1, cv2.LINE_AA)
        cv2.putText(frame, value_text, (x + 14, y + int(h * 0.45)), self.font_large, 0.85, accent_color, 2, cv2.LINE_AA)
        cv2.putText(frame, subtitle, (x + 14, y + h - 14), self.font_small, 0.45, self.COLORS["text_dim"], 1, cv2.LINE_AA)

    def _state_color(self, state: FatigueState) -> Tuple[int, int, int]:
        """Return the main accent color for the state."""
        if state == FatigueState.ALERT:
            return self.COLORS["alert"]
        if state == FatigueState.FATIGUE:
            return self.COLORS["fatigue"]
        return self.COLORS["normal"]

    def _draw_icon_power(self, frame: np.ndarray, center: Tuple[int, int], radius: int, active: bool) -> None:
        color = self.COLORS["normal"] if active else (112, 122, 136)
        cv2.circle(frame, center, radius, color, 3)
        cv2.line(frame, (center[0], center[1] - radius - 4), (center[0], center[1] + 3), color, 3)

    def _draw_icon_camera(self, frame: np.ndarray, top_left: Tuple[int, int], active: bool) -> None:
        color = self.COLORS["normal"] if active else (112, 122, 136)
        x, y = top_left
        cv2.rectangle(frame, (x, y + 8), (x + 48, y + 36), color, 2)
        cv2.circle(frame, (x + 24, y + 22), 8, color, 2)
        pts = np.array([[x + 48, y + 16], [x + 62, y + 10], [x + 62, y + 34], [x + 48, y + 28]], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 2)

    def _draw_icon_car(self, frame: np.ndarray, top_left: Tuple[int, int], active: bool) -> None:
        color = self.COLORS["normal"] if active else (112, 122, 136)
        x, y = top_left
        body = np.array(
            [
                [x + 10, y + 28],
                [x + 18, y + 14],
                [x + 48, y + 14],
                [x + 58, y + 28],
                [x + 62, y + 28],
                [x + 62, y + 40],
                [x + 6, y + 40],
                [x + 6, y + 28],
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [body], True, color, 2)
        cv2.circle(frame, (x + 18, y + 42), 6, color, 2)
        cv2.circle(frame, (x + 50, y + 42), 6, color, 2)

    def get_step_regions(self, origin_x: int = 0, origin_y: int = 0) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Return clickable step rectangles adjusted to the given origin."""
        return [
            ((x1 + origin_x, y1 + origin_y), (x2 + origin_x, y2 + origin_y))
            for (x1, y1), (x2, y2) in self.step_regions
        ]

    def draw_dashboard(self, frame: np.ndarray, simulation_started: bool, driving_active: bool, camera_ready: bool, monochrome: bool = False) -> None:
        """Draw the simulation-style dashboard and step cards.

        If `monochrome` is True, render a simplified black-and-white
        interface intended for the initial idle screen. Text is wrapped
        so it remains inside the UI bounds.
        """
        h, w = frame.shape[:2]

        if monochrome:
            # Clear to black
            frame[:] = (0, 0, 0)

            header_h = 72
            # header area (black background with white badge)
            cv2.rectangle(frame, (0, 0), (w, header_h), (0, 0, 0), -1)
            self._draw_badge(frame, "POSTE DE CONTROLE", (18, 12), (255, 255, 255), (0, 0, 0), 0.5)
            self._draw_wrapped_text(frame, "INTERFACE DE SURVEILLANCE", (28, 48), self._max_text_width(frame, 28, 28), font=self.font_medium, scale=0.9, color=(245, 245, 245), thickness=1)

            # Main panel with subtle white border
            margin = 20
            panel_tl = (margin, header_h + 8)
            panel_br = (w - margin, h - margin)
            self._draw_panel(frame, panel_tl, panel_br, (20, 20, 20), alpha=0.98, border_color=(220, 220, 220))

            # Large central status card
            card_x = panel_tl[0] + 24
            card_y = panel_tl[1] + 24
            card_w = min(800, panel_br[0] - card_x - 24)
            card_h = 140
            self._draw_panel(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (28, 28, 28), alpha=0.98, border_color=(200, 200, 200))
            status = "SYSTEME EN ATTENTE" if not simulation_started else "SIMULATION PRETE"
            self._draw_wrapped_text(frame, f"STATUT: {status}", (card_x + 14, card_y + 34), card_w - 28, font=self.font_medium, scale=0.85, color=(245, 245, 245), thickness=1)
            self._draw_wrapped_text(frame, "Appuyez sur 1 pour lancer la simulation. 2 pour demarrer la conduite.", (card_x + 14, card_y + 74), card_w - 28, font=self.font_medium, scale=0.6, color=(200, 200, 200), thickness=1)

            # Small metrics column to the right inside panel
            small_w = 220
            small_h = 72
            sx = card_x + card_w + 18
            sy = card_y
            metrics = [("CAMERAS", "Non connectee"), ("ALARMES", "0"), ("PRESENCE", "Aucune")]
            for i, (t, v) in enumerate(metrics):
                cy = sy + i * (small_h + 12)
                if sx + small_w + 16 > panel_br[0]:
                    sx = card_x
                    cy = card_y + card_h + 18 + i * (small_h + 12)
                self._draw_panel(frame, (sx, cy), (sx + small_w, cy + small_h), (24, 24, 24), alpha=0.98, border_color=(200, 200, 200))
                self._draw_wrapped_text(frame, t, (sx + 12, cy + 22), small_w - 24, font=self.font_small, scale=0.6, color=(210, 210, 210), thickness=1)
                self._draw_wrapped_text(frame, v, (sx + 12, cy + 50), small_w - 24, font=self.font_medium, scale=0.72, color=(245, 245, 245), thickness=1)

            # Footer note inside panel, wrapped
            foot_y = panel_br[1] - 42
            self._draw_wrapped_text(frame, "Interface monochrome — textes confinés au cadre.", (card_x + 12, foot_y), panel_br[0] - card_x - 36, font=self.font_small, scale=0.5, color=(180, 180, 180), thickness=1)
            return

        if driving_active:
            status_text = "VEHICULE EN MARCHE"
            status_color = self.COLORS["normal"]
            status_fill = (26, 55, 44)
        elif simulation_started:
            status_text = "SIMULATION PRETE"
            status_color = self.COLORS["fatigue"]
            status_fill = (36, 53, 70)
        else:
            status_text = "SYSTEME EN ATTENTE"
            status_color = (185, 192, 200)
            status_fill = (43, 47, 54)

        self._draw_panel(frame, (28, 132), (w - 28, 176), status_fill, alpha=0.95, border_color=status_color)
        cv2.putText(
            frame,
            status_text,
            (44, 160),
            cv2.FONT_HERSHEY_DUPLEX,
            0.72,
            status_color,
            2,
            cv2.LINE_AA,
        )

        self._draw_panel(frame, (20, 188), (340, h - 20), self.COLORS["panel"], alpha=0.94)
        cv2.putText(frame, "PARCOURS", (38, 220), cv2.FONT_HERSHEY_DUPLEX, 0.74, self.COLORS["text"], 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Chaque carte peut etre activee au clic.",
            (38, 244),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self.COLORS["text_dim"],
            1,
            cv2.LINE_AA,
        )

        steps = [
            ("01", "Lancer la simulation", "Preparation de l interface", simulation_started, self.COLORS["fatigue"]),
            ("02", "Demarrer le vehicule", "Activation camera et analyse", driving_active, self.COLORS["normal"]),
            ("03", "Presence conducteur", "Verification de presence", driving_active and camera_ready, self.COLORS["accent"]),
            ("04", "Suivi tete 3D", "Lecture posture et orientation", driving_active and camera_ready, self.COLORS["fatigue"]),
            ("05", "Arreter le vehicule", "Retour au mode attente", simulation_started and not driving_active, self.COLORS["warning"]),
        ]

        for index, (step_no, title, subtitle, done, accent) in enumerate(steps):
            (x1, y1), (x2, y2) = self.step_regions[index]
            bg = self.COLORS["panel_alt"] if done else (26, 33, 45)
            border = accent if done else (56, 67, 82)
            self._draw_panel(frame, (x1, y1), (x2, y2), bg, alpha=0.96, border_color=border)
            cv2.circle(frame, (x1 + 28, y1 + 27), 16, accent if done else (90, 100, 116), -1)
            cv2.putText(frame, step_no, (x1 + 15, y1 + 33), cv2.FONT_HERSHEY_DUPLEX, 0.48, (18, 22, 28), 1, cv2.LINE_AA)
            cv2.putText(frame, title, (x1 + 56, y1 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.56, self.COLORS["text"], 1, cv2.LINE_AA)
            cv2.putText(frame, subtitle, (x1 + 56, y1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLORS["text_dim"], 1, cv2.LINE_AA)

        self._draw_panel(frame, (352, 188), (w - 20, 340), self.COLORS["panel"], alpha=0.94)
        cv2.putText(frame, "COMMANDES", (372, 220), cv2.FONT_HERSHEY_DUPLEX, 0.74, self.COLORS["text"], 2, cv2.LINE_AA)
        commands = [
            "Clic sur 01 pour lancer l interface",
            "Clic sur 02 pour ouvrir la surveillance",
            "Clic sur 05 pour couper la conduite",
            "Raccourcis clavier : 1, 2, 5, Q",
        ]
        for idx, text in enumerate(commands):
            y = 252 + idx * 26
            cv2.circle(frame, (378, y - 4), 4, self.COLORS["accent"], -1)
            self._draw_wrapped_text(frame, text, (392, y), self._max_text_width(frame, 392, 28), font=self.font_medium, scale=0.49, color=(228, 234, 240), thickness=1, line_spacing=4)

        self._draw_panel(frame, (352, 356), (w - 20, h - 20), self.COLORS["panel"], alpha=0.94)
        cv2.putText(frame, "ETAT DU SYSTEME", (372, 388), cv2.FONT_HERSHEY_DUPLEX, 0.74, self.COLORS["text"], 2, cv2.LINE_AA)

        self._draw_icon_power(frame, (404, 444), 22, simulation_started)
        self._draw_icon_car(frame, (488, 416), driving_active)
        self._draw_icon_camera(frame, (588, 420), camera_ready and driving_active)

        labels = [
            ("Simulation", 366, 492, simulation_started),
            ("Vehicule", 470, 492, driving_active),
            ("Camera IA", 565, 492, camera_ready and driving_active),
        ]
        for text, x, y, active in labels:
            color = self.COLORS["normal"] if active else self.COLORS["text_dim"]
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

        readiness = 1.0 if driving_active and camera_ready else 0.55 if simulation_started else 0.18
        readiness_text = (
            "Surveillance active : presence, fatigue et posture en direct"
            if driving_active and camera_ready
            else "Systeme arme, en attente d activation conduite"
            if simulation_started
            else "Systeme inactif tant que la simulation n est pas lancee"
        )
        readiness_color = self.COLORS["normal"] if driving_active and camera_ready else self.COLORS["fatigue"]
        cv2.putText(frame, "NIVEAU DE PREPARATION", (372, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.COLORS["text_dim"], 1, cv2.LINE_AA)
        self._draw_progress_line(frame, (372, 554), w - 412, readiness, readiness_color)
        self._draw_wrapped_text(frame, readiness_text, (372, 584), self._max_text_width(frame, 372, 28), font=self.font_medium, scale=0.47, color=(228, 234, 240), thickness=1, line_spacing=6)

    def compose_driving_view(self, camera_frame: np.ndarray, simulation_started: bool, driving_active: bool, camera_ready: bool) -> np.ndarray:
        """Combine live camera feed with the dashboard side panel."""
        # Instead of composing a separate side-panel canvas, draw the
        # secondary dashboard elements directly onto the camera frame so
        # the interface is overlaid on the image itself.
        h, w = camera_frame.shape[:2]
        # Add a small badge identifying the camera and window size info.
        self._draw_badge(camera_frame, "CAMERA CONDUCTEUR", (18, h - 52), (23, 33, 48), self.COLORS["camera_label"], 0.52)
        self.draw_window_size(camera_frame)
        return camera_frame

    def draw_window_size(self, frame: np.ndarray) -> None:
        """Draw the current canvas/window size as a small badge."""
        h, w = frame.shape[:2]
        text = f"{w}x{h}"
        # position to the left of the FPS badge
        x = max(18, frame.shape[1] - 240)
        y = 14
        self._draw_badge(frame, text, (x, y), (24, 33, 46), self.COLORS["text"], 0.46)

    def draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
        bg_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """Draw text with semi-transparent background."""
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        x, y = position
        padding_x = 10
        padding_y = 7
        self._draw_panel(
            frame,
            (x - padding_x, y - text_height - padding_y),
            (x + text_width + padding_x, y + baseline + padding_y - 2),
            bg_color or self.COLORS["panel"],
            alpha=0.88,
            border_color=bg_color or self.COLORS["border"],
        )
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_wrapped_text(
        self,
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        max_width: int,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        scale: float = 0.5,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        line_spacing: int = 6,
    ) -> None:
        """Draw text wrapped to fit within max_width. Long words are truncated with ellipsis.

        origin is the (x, y) for the first baseline.
        """
        words = text.split()
        lines: list[str] = []
        cur_line = ""
        (space_w, _), _ = cv2.getTextSize(" ", font, scale, thickness)

        def _fits(s: str) -> bool:
            (w, _), _ = cv2.getTextSize(s, font, scale, thickness)
            return w <= max_width

        def _truncate(s: str) -> str:
            if _fits(s):
                return s
            # truncate with ellipsis
            for l in range(len(s), 0, -1):
                candidate = s[:l] + "..."
                if _fits(candidate):
                    return candidate
            return "..."

        for w in words:
            if cur_line:
                test = cur_line + " " + w
            else:
                test = w

            if _fits(test):
                cur_line = test
            else:
                if not cur_line:
                    # single too-long word
                    lines.append(_truncate(w))
                    cur_line = ""
                else:
                    lines.append(cur_line)
                    cur_line = w if _fits(w) else _truncate(w)

        if cur_line:
            lines.append(cur_line)

        # draw lines
        x, y = origin
        (line_h, _), _ = cv2.getTextSize("Ay", font, scale, thickness)
        for i, line in enumerate(lines):
            y_i = y + i * (line_h + line_spacing)
            cv2.putText(frame, line, (x, y_i), font, scale, color, thickness, cv2.LINE_AA)

    def _max_text_width(self, frame: np.ndarray, left_margin: int = 28, right_margin: int = 28) -> int:
        """Return a capped maximum width for text blocks so that all text fits within target resolution.

        Uses the smaller of the current frame width and the target design width.
        """
        frame_w = frame.shape[1]
        target_w = min(frame_w, self.target_resolution[0])
        return max(120, target_w - (left_margin + right_margin))

    def draw_status_bar(self, frame: np.ndarray, state: FatigueState, fatigue_score: float, status_text_override: Optional[str] = None) -> None:
        """Draw top status banner with current vigilance state."""
        h, w = frame.shape[:2]
        del h
        color = self._state_color(state)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 116), (8, 12, 18), -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        state_text = status_text_override or self.STATE_TEXT.get(state, "VIGILANCE STABLE")
        self._draw_badge(frame, "SURVEILLANCE ACTIVE", (18, 14), (26, 40, 58), self.COLORS["text_dim"], 0.45)
        cv2.putText(frame, state_text, (20, 64), cv2.FONT_HERSHEY_DUPLEX, 0.95, color, 2, cv2.LINE_AA)
        info_line = "Evaluation continue des yeux, de la bouche et de la posture."
        self._draw_wrapped_text(frame, info_line, (20, 92), self._max_text_width(frame, 20, 28), font=self.font_medium, scale=0.5, color=self.COLORS["text_dim"], thickness=1)
        score_label = f"SCORE {fatigue_score:.0f}%"
        self.draw_text_with_background(frame, score_label, (w - 160, 58), color, 0.62, 2, (25, 35, 48))

    def draw_metrics(self, frame: np.ndarray, ear: float, mar: float, eye_frames: int, yawn_frames: int) -> None:
        """Draw real-time metrics as cards."""
        self._draw_metric_card(
            frame,
            "EAR",
            f"{ear:.3f}",
            f"Seuil {self.config.EAR_THRESHOLD:.2f}",
            (18, 124),
            (122, 74),
            self.COLORS["normal"] if ear >= self.config.EAR_THRESHOLD else self.COLORS["alert"],
        )
        self._draw_metric_card(
            frame,
            "MAR",
            f"{mar:.3f}",
            f"Seuil {self.config.MAR_THRESHOLD:.2f}",
            (150, 124),
            (122, 74),
            self.COLORS["warning"] if mar >= self.config.MAR_THRESHOLD else self.COLORS["fatigue"],
        )
        self._draw_metric_card(
            frame,
            "YEUX FERMES",
            str(eye_frames),
            "Frames consecutives",
            (282, 124),
            (156, 74),
            self.COLORS["alert"] if eye_frames > 0 else self.COLORS["text"],
        )
        self._draw_metric_card(
            frame,
            "BAILLEMENT",
            str(yawn_frames),
            "Compteur actif",
            (448, 124),
            (156, 74),
            self.COLORS["warning"] if yawn_frames > 0 else self.COLORS["text"],
        )

    def draw_fatigue_bar(self, frame: np.ndarray, fatigue_score: float, state: FatigueState) -> None:
        """Draw fatigue progress bar."""
        h, w = frame.shape[:2]
        color = self._state_color(state)
        bar_width = max(140, w - 40)
        container_y = h - 70
        self._draw_panel(frame, (18, container_y - 22), (w - 18, h - 18), self.COLORS["panel"], alpha=0.88)
        cv2.putText(frame, "NIVEAU DE FATIGUE", (32, container_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.COLORS["text"], 1, cv2.LINE_AA)
        self._draw_progress_line(frame, (32, container_y + 16), bar_width - 52, fatigue_score / 100.0, color, background_color=(55, 62, 72))
        cv2.putText(frame, f"{fatigue_score:.0f} POURCENT", (w - 170, container_y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 2, cv2.LINE_AA)

    def draw_alarm_status(self, frame: np.ndarray, alarm_active: bool) -> None:
        """Draw alarm status indicator."""
        if alarm_active:
            h, w = frame.shape[:2]
            badge_w, _ = self._draw_badge(frame, "ALARME ACTIVE", (w - 190, h - 122), (48, 36, 58), self.COLORS["alert"], 0.5)
            msg = "Attente d un retour normal stable"
            self._draw_wrapped_text(frame, msg, (w - 190, h - 92), self._max_text_width(frame, 20, 28), font=self.font_medium, scale=0.42, color=self.COLORS["text_dim"], thickness=1)
            del badge_w

    def draw_detection_indicators(
        self,
        frame: np.ndarray,
        eyes_closed: bool,
        yawning: bool,
        head_drop: bool,
        sleep: bool,
        dangerous_posture: bool = False,
        posture_label: str = "NORMAL",
    ) -> None:
        """Draw a single prominent live detection badge."""
        label = None
        color = None
        if sleep:
            label = "SOMMEIL DETECTE"
            color = self.COLORS["alert"]
        elif eyes_closed:
            label = "YEUX FERMES"
            color = self.COLORS["alert"]
        elif yawning:
            label = "BAILLEMENT"
            color = self.COLORS["warning"]
        elif head_drop:
            label = "REGARD VERS LE BAS"
            color = self.COLORS["warning"]
        elif dangerous_posture:
            label = posture_label
            color = self.COLORS["warning"]

        if label is None:
            return

        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.56, 1)
        x = frame.shape[1] - text_width - 44
        self._draw_badge(frame, label, (x, 124), (27, 37, 50), color, 0.56)

    def draw_visual_alert(self, frame: np.ndarray, state: FatigueState) -> None:
        """Draw visual alert effect."""
        if state == FatigueState.ALERT and int(time.time() * 4) % 2:
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.COLORS["alert"], 18)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter."""
        self._draw_badge(frame, f"FPS {fps:.1f}", (frame.shape[1] - 116, 14), (24, 33, 46), self.COLORS["text"], 0.46)

    def draw_instructions(self, frame: np.ndarray) -> None:
        """Draw instruction text."""
        self._draw_badge(frame, "Q POUR QUITTER", (frame.shape[1] - 176, 48), (24, 33, 46), self.COLORS["text_dim"], 0.43)

    def draw_landmarks(self, frame: np.ndarray, left_eye: list, right_eye: list, mouth: list) -> None:
        """Draw facial landmarks for visualization."""
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 1, (88, 224, 166), -1)
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (255, 198, 92), -1)
        for point in mouth:
            cv2.circle(frame, point, 2, (95, 170, 255), -1)

    def draw_full_landmarks(self, frame: np.ndarray, points: list[tuple[int, int]], color: Tuple[int, int, int] = (200, 240, 200)) -> None:
        """Draw all face mesh landmarks as small dots.

        `points` is a list of (x, y) coordinates in image space.
        """
        if not points:
            return
        for (x, y) in points:
            # draw a subtle outer dot and a brighter inner dot for visibility
            cv2.circle(frame, (x, y), 2, (40, 40, 40), -1)
            cv2.circle(frame, (x, y), 1, color, -1)

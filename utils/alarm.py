"""
Persistent alarm system for driver fatigue detection.
Provides continuous audio alerts until condition is resolved.
"""

import threading
import time
import numpy as np
import pygame
from typing import Optional

from utils.config import DetectionConfig
from utils.logger import SystemLogger


class PersistentAlarm:
    """
    Persistent audio alarm that continues until manually stopped.
    Implements anti-spam and different alarm types.
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None, logger: Optional[SystemLogger] = None):
        self.config = config or DetectionConfig()
        self.logger = logger or SystemLogger()
        
        # Initialize pygame mixer
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2)
            self.initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize pygame mixer: {e}")
            self.initialized = False
        
        # Alarm state
        self.alarm_active = False
        self.alarm_thread = None
        self.stop_alarm_flag = False
        
        # Warning cooldown
        self.last_warning_time = 0
        
    def _generate_beep(self, frequency: int, duration: float, volume: float = 0.9) -> Optional[pygame.mixer.Sound]:
        """
        Generate a beep sound.
        
        Args:
            frequency: Sound frequency in Hz
            duration: Duration in seconds
            volume: Volume level (0-1)
            
        Returns:
            Pygame sound object or None
        """
        if not self.initialized:
            return None
        
        try:
            sample_rate = 44100
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples)
            wave = volume * np.sin(2 * np.pi * frequency * t)
            wave_int16 = (wave * 32767).astype(np.int16)
            wave_stereo = np.column_stack((wave_int16, wave_int16))
            return pygame.sndarray.make_sound(wave_stereo)
        except Exception as e:
            self.logger.error(f"Failed to generate beep: {e}")
            return None
    
    def _play_loop(self):
        """Internal loop for continuous alarm"""
        beep = self._generate_beep(
            self.config.ALARM_BEEP_FREQUENCY,
            self.config.ALARM_BEEP_DURATION
        )
        
        while self.alarm_active and not self.stop_alarm_flag:
            try:
                # Play series of 3 beeps continuously
                for _ in range(3):
                    if not self.alarm_active:
                        break
                    if beep:
                        beep.play()
                    time.sleep(self.config.ALARM_BEEP_DURATION)
                    time.sleep(0.03)
                time.sleep(0.25)
            except Exception as e:
                self.logger.error(f"Alarm playback error: {e}")
    
    def start_alarm(self):
        """Start persistent alarm"""
        if not self.alarm_active and self.initialized:
            self.alarm_active = True
            self.stop_alarm_flag = False
            self.alarm_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.alarm_thread.start()
            self.logger.info("Persistent alarm activated")
    
    def stop_alarm(self):
        """Stop persistent alarm"""
        if self.alarm_active:
            self.alarm_active = False
            self.stop_alarm_flag = True
            if self.initialized:
                pygame.mixer.stop()
            if self.alarm_thread and self.alarm_thread.is_alive():
                self.alarm_thread.join(timeout=0.5)
            self.logger.info("Persistent alarm deactivated")
    
    def play_warning(self):
        """Play single warning beep with cooldown"""
        if not self.initialized:
            return
        
        current_time = time.time()
        if current_time - self.last_warning_time < self.config.ALARM_WARNING_INTERVAL:
            return
        
        self.last_warning_time = current_time
        
        def play():
            beep = self._generate_beep(
                self.config.ALARM_WARNING_FREQUENCY,
                0.2,
                0.5
            )
            if beep:
                beep.play()
                time.sleep(0.2)
        
        threading.Thread(target=play, daemon=True).start()
    
    def is_active(self) -> bool:
        """Check if alarm is currently active"""
        return self.alarm_active

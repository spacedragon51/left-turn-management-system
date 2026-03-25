"""
Lane Manager - Handles lane region configuration and management
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import json
import os


class LaneManager:
    """
    Manage lane regions for free-left turn detection
    Supports saving/loading configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize lane manager
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "lane_config.json"
        self.lane_region = None
        self.crossing_region = None
        self._load_config()
    
    def set_lane_region(self, points: List[Tuple[int, int]]) -> None:
        """
        Set free-left lane region
        
        Args:
            points: List of (x, y) polygon points
        """
        self.lane_region = points
        self._save_config()
    
    def set_crossing_region(self, points: List[Tuple[int, int]]) -> None:
        """
        Set pedestrian crossing region
        
        Args:
            points: List of (x, y) polygon points
        """
        self.crossing_region = points
        self._save_config()
    
    def get_lane_region(self) -> Optional[List[Tuple[int, int]]]:
        """Get free-left lane region"""
        return self.lane_region
    
    def get_crossing_region(self) -> Optional[List[Tuple[int, int]]]:
        """Get pedestrian crossing region"""
        return self.crossing_region
    
    def is_in_lane(self, point: Tuple[int, int]) -> bool:
        """Check if point is in free-left lane"""
        if not self.lane_region:
            return False
        return cv2.pointPolygonTest(
            np.array(self.lane_region, dtype=np.int32),
            point,
            False
        ) >= 0
    
    def is_at_crossing(self, point: Tuple[int, int]) -> bool:
        """Check if point is at pedestrian crossing"""
        if not self.crossing_region:
            return False
        return cv2.pointPolygonTest(
            np.array(self.crossing_region, dtype=np.int32),
            point,
            False
        ) >= 0
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        config = {
            'lane_region': self.lane_region,
            'crossing_region': self.crossing_region
        }
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.lane_region = config.get('lane_region')
                    self.crossing_region = config.get('crossing_region')
            except Exception as e:
                print(f"Could not load config: {e}")
    
    def get_default_lane_region(self, width: int = 1280, height: int = 720) -> List[Tuple[int, int]]:
        """Get default lane region for given dimensions"""
        return [
            (width // 10, height - 150),
            (width // 3, height - 150),
            (width // 3 + 30, height - 120),
            (width // 10, height - 120)
        ]
    
    def get_default_crossing_region(self, width: int = 1280, height: int = 720) -> List[Tuple[int, int]]:
        """Get default crossing region for given dimensions"""
        return [
            (width // 3, height // 2 - 50),
            (2 * width // 3, height // 2 - 50),
            (2 * width // 3, height // 2 + 50),
            (width // 3, height // 2 + 50)
        ]
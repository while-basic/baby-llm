#----------------------------------------------------------------------------
#File:       logger.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Development logger for the neural child system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Development logger for the neural child system.

Extracted from neural-child-init/logger.py
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class DevelopmentLogger:
    """Logger for development-related events and interactions."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize the development logger.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        self.development_log = self.log_dir / "development.log"
        self.interaction_log = self.log_dir / "interactions.json"
        self.error_log = self.log_dir / "errors.log"
        self.vision_log = self.log_dir / "vision.log"
        self.training_log = self.log_dir / "training.json"

        # Set up Python logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def log_development(self, data: Dict[str, Any]) -> None:
        """Log development-related events.

        Args:
            data: Development data to log
        """
        self._write_log(self.development_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'development',
            'data': data
        })

    def log_interaction(
        self,
        interaction_data: Dict[str, Any],
        action: int,
        reward: float,
        state: Any
    ) -> None:
        """Log interaction data.

        Args:
            interaction_data: Interaction data dictionary
            action: Action taken
            reward: Reward received
            state: Current state
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'interaction_data': interaction_data,
            'action': int(action),
            'reward': float(reward),
            'state': str(state)
        }

        # Append to log file
        try:
            with open(self.interaction_log, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Error writing interaction log: {e}")

    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error events.

        Args:
            error: Error message
            context: Optional context information
        """
        self._write_log(self.error_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'error': str(error),
            'context': context or {}
        })

    def log_vision(self, data: Dict[str, Any]) -> None:
        """Log vision processing events.

        Args:
            data: Vision processing data
        """
        self._write_log(self.vision_log, {
            'timestamp': datetime.now().isoformat(),
            'type': 'vision',
            'data': data
        })

    def log_training(self, metrics: Dict[str, float]) -> None:
        """Log training metrics.

        Args:
            metrics: Training metrics dictionary
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'metrics': metrics
        }

        # Append to log file
        try:
            with open(self.training_log, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Error writing training log: {e}")

    def _write_log(self, log_file: Path, data: Dict[str, Any]) -> None:
        """Write log entry to file.

        Args:
            log_file: Path to log file
            data: Data to write
        """
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                json.dump(data, f)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Error writing to log {log_file}: {e}")

    def get_development_history(
        self,
        start_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get development history from logs.

        Args:
            start_time: Optional start time to filter logs

        Returns:
            List of development log entries
        """
        history = []
        try:
            if not self.development_log.exists():
                return history

            with open(self.development_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if start_time:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            if entry_time >= start_time:
                                history.append(entry)
                        else:
                            history.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            self.logger.error(f"Error reading development history: {e}")

        return history


"""Tests for configuration management.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
import os
import yaml
import json
import tempfile
from pydantic import ValidationError

from neuralchild.config import (
    Config,
    ServerConfig,
    ModelConfig,
    VisualizationConfig,
    MindConfig,
    LoggingConfig,
    DevelopmentConfig,
    load_config,
    get_config
)


class TestServerConfig:
    """Test suite for ServerConfig."""

    def test_server_config_defaults(self):
        """Test that ServerConfig has correct default values."""
        config = ServerConfig()
        assert config.llm_server_url == "http://localhost:1234/v1/chat/completions"
        assert config.embedding_server_url == "http://localhost:1234/v1/embeddings"
        assert config.obsidian_api_url is None

    def test_server_config_custom_values(self):
        """Test ServerConfig with custom values."""
        config = ServerConfig(
            llm_server_url="http://custom:8080/chat",
            embedding_server_url="http://custom:8080/embed",
            obsidian_api_url="http://obsidian:3000"
        )
        assert config.llm_server_url == "http://custom:8080/chat"
        assert config.embedding_server_url == "http://custom:8080/embed"
        assert config.obsidian_api_url == "http://obsidian:3000"


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_model_config_defaults(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.temperature == 0.7
        assert config.max_tokens == -1

    def test_model_config_temperature_validation(self):
        """Test that temperature is validated correctly."""
        # Valid temperature
        config = ModelConfig(temperature=1.0)
        assert config.temperature == 1.0

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            ModelConfig(temperature=2.5)

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            ModelConfig(temperature=-0.5)

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            llm_model="gpt-4",
            embedding_model="custom-embed",
            temperature=0.9,
            max_tokens=2000
        )
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.9
        assert config.max_tokens == 2000


class TestVisualizationConfig:
    """Test suite for VisualizationConfig."""

    def test_visualization_config_defaults(self):
        """Test that VisualizationConfig has correct default values."""
        config = VisualizationConfig()
        assert config.enabled is True
        assert config.update_interval == 1.0
        assert config.obsidian_integration is False
        assert config.graph_enabled is True

    def test_visualization_config_colors(self):
        """Test color configuration."""
        config = VisualizationConfig()
        assert "background" in config.colors
        assert "text" in config.colors
        assert config.colors["background"] == "#f0f0f0"

    def test_visualization_config_network_display(self):
        """Test network display configuration."""
        config = VisualizationConfig()
        assert config.network_display["consciousness"] is True
        assert config.network_display["emotions"] is True
        assert config.network_display["perception"] is True
        assert config.network_display["thoughts"] is True


class TestMindConfig:
    """Test suite for MindConfig."""

    def test_mind_config_defaults(self):
        """Test that MindConfig has correct default values."""
        config = MindConfig()
        assert config.learning_rate == 0.001
        assert config.step_interval == 0.1
        assert config.development_acceleration == 1.0
        assert config.starting_stage == "INFANT"

    def test_mind_config_validation(self):
        """Test MindConfig field validation."""
        # Valid learning rate
        config = MindConfig(learning_rate=0.01)
        assert config.learning_rate == 0.01

        # Invalid learning rate (too high)
        with pytest.raises(ValidationError):
            MindConfig(learning_rate=0.5)

        # Invalid learning rate (too low)
        with pytest.raises(ValidationError):
            MindConfig(learning_rate=0.00001)

    def test_mind_config_networks(self):
        """Test network configuration."""
        config = MindConfig()
        assert "consciousness" in config.networks
        assert "emotions" in config.networks
        assert config.networks["consciousness"]["hidden_dim"] == 128

    def test_mind_config_features(self):
        """Test features configuration."""
        config = MindConfig()
        assert config.features_enabled["memory_consolidation"] is True
        assert config.features_enabled["emotional_development"] is True


class TestLoggingConfig:
    """Test suite for LoggingConfig."""

    def test_logging_config_defaults(self):
        """Test that LoggingConfig has correct default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.file_logging is True
        assert config.console_logging is True
        assert config.log_file == "neuralchild.log"

    def test_logging_config_level_validation(self):
        """Test log level validation."""
        # Valid log level
        config = LoggingConfig(level="DEBUG")
        assert config.level == "DEBUG"

        # Invalid log level should be corrected to INFO
        config = LoggingConfig(level="INVALID")
        assert config.level == "INFO"


class TestDevelopmentConfig:
    """Test suite for DevelopmentConfig."""

    def test_development_config_defaults(self):
        """Test that DevelopmentConfig has correct default values."""
        config = DevelopmentConfig()
        assert config.debug_mode is False
        assert config.simulate_llm is False
        assert config.profile_performance is False
        assert config.crash_on_error is False
        assert config.record_metrics is True

    def test_development_config_experimental_features(self):
        """Test experimental features."""
        config = DevelopmentConfig(
            experimental_features={"new_feature": True}
        )
        assert config.experimental_features["new_feature"] is True


class TestConfig:
    """Test suite for main Config class."""

    def test_config_initialization(self):
        """Test that Config initializes with all sub-configs."""
        config = Config()
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.visualization, VisualizationConfig)
        assert isinstance(config.mind, MindConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.development, DevelopmentConfig)

    def test_config_to_yaml(self, temp_dir: str):
        """Test saving configuration to YAML."""
        config = Config()
        config_path = os.path.join(temp_dir, "test_config.yaml")

        # Save to YAML
        config.to_yaml(config_path)

        # Verify file exists
        assert os.path.exists(config_path)

        # Verify content
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        assert "server" in data
        assert "model" in data
        assert "mind" in data

    def test_config_from_yaml(self, config_file: str):
        """Test loading configuration from YAML."""
        loaded_config = Config.from_yaml(config_file)
        assert isinstance(loaded_config, Config)
        assert loaded_config.development.simulate_llm is True

    def test_config_from_yaml_nonexistent_file(self):
        """Test loading from non-existent file returns defaults."""
        config = Config.from_yaml("/nonexistent/path/config.yaml")
        assert isinstance(config, Config)
        assert config.development.simulate_llm is False  # Default value

    def test_config_from_yaml_empty_file(self, temp_dir: str):
        """Test loading from empty file returns defaults."""
        empty_file = os.path.join(temp_dir, "empty.yaml")
        with open(empty_file, 'w') as f:
            f.write("")

        config = Config.from_yaml(empty_file)
        assert isinstance(config, Config)

    def test_config_to_json(self, temp_dir: str):
        """Test saving configuration to JSON."""
        config = Config()
        json_path = os.path.join(temp_dir, "test_config.json")

        # Save to JSON
        config.to_json(json_path)

        # Verify file exists
        assert os.path.exists(json_path)

        # Verify content
        with open(json_path, 'r') as f:
            data = json.load(f)
        assert "server" in data
        assert "model" in data
        assert "mind" in data

    def test_config_nested_values(self):
        """Test accessing nested configuration values."""
        config = Config()
        # Test nested access
        assert config.server.llm_server_url is not None
        assert config.model.temperature > 0
        assert config.mind.learning_rate > 0

    def test_config_modification(self):
        """Test modifying configuration values."""
        config = Config()
        original_temp = config.model.temperature

        # Modify value
        config.model.temperature = 0.9
        assert config.model.temperature == 0.9
        assert config.model.temperature != original_temp

    def test_load_config_function(self, config_file: str):
        """Test the load_config function."""
        config = load_config(config_file)
        assert isinstance(config, Config)
        assert config.development.simulate_llm is True

    def test_get_config_function(self):
        """Test the get_config function."""
        config = get_config()
        assert isinstance(config, Config)


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_config_roundtrip_yaml(self, temp_dir: str):
        """Test saving and loading configuration maintains values."""
        # Create config with custom values
        original_config = Config()
        original_config.model.temperature = 0.85
        original_config.mind.learning_rate = 0.005
        original_config.development.debug_mode = True

        # Save to YAML
        config_path = os.path.join(temp_dir, "roundtrip.yaml")
        original_config.to_yaml(config_path)

        # Load from YAML
        loaded_config = Config.from_yaml(config_path)

        # Verify values match
        assert loaded_config.model.temperature == 0.85
        assert loaded_config.mind.learning_rate == 0.005
        assert loaded_config.development.debug_mode is True

    def test_config_partial_update(self, temp_dir: str):
        """Test loading partial configuration file."""
        # Create partial config file
        partial_config = {
            "model": {
                "temperature": 0.95
            },
            "development": {
                "debug_mode": True
            }
        }

        config_path = os.path.join(temp_dir, "partial.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(partial_config, f)

        # Load config
        config = Config.from_yaml(config_path)

        # Verify updated values
        assert config.model.temperature == 0.95
        assert config.development.debug_mode is True

        # Verify defaults are still present
        assert config.server.llm_server_url is not None
        assert config.mind.learning_rate == 0.001

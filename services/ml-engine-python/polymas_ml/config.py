import structlog
from pathlib import Path

logger = structlog.get_logger()

CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
MODEL_DIR = Path(__file__).parent / "config"

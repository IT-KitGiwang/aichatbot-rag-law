# src/utils/logger.py
"""
Module logging cho toàn bộ hệ thống.
Ghi log ra console và file.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.utils.config import BASE_DIR


def setup_logger(
    name: str = "chatbot_rag",
    level: str = "INFO",
    log_dir: str = None
) -> logging.Logger:
    """
    Thiết lập logger với output ra console và file.
    
    Args:
        name: Tên logger
        level: Mức log (DEBUG, INFO, WARNING, ERROR)
        log_dir: Thư mục lưu file log
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Tránh duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is None:
        log_dir = BASE_DIR / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(
        log_dir / f"{today}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Logger mặc định cho toàn dự án
logger = setup_logger()

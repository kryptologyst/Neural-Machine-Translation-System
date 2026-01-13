"""
Enhanced Logging Configuration for Neural Machine Translation System

This module provides comprehensive logging setup with file rotation,
structured logging, and performance monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime
from functools import wraps


class TranslationLogger:
    """Enhanced logger for translation operations."""
    
    def __init__(self, name: str = "translation"):
        """
        Initialize the translation logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "translation.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def log_translation_start(self, text: str, language_pair: str, **kwargs):
        """Log translation start."""
        self.logger.info(
            f"Translation started - Pair: {language_pair}, "
            f"Length: {len(text.split())} words, "
            f"Params: {kwargs}"
        )
    
    def log_translation_end(self, result: str, duration: float, **kwargs):
        """Log translation completion."""
        self.logger.info(
            f"Translation completed - Duration: {duration:.3f}s, "
            f"Result length: {len(result.split())} words"
        )
    
    def log_evaluation(self, metrics: Dict[str, Any], sample_count: int):
        """Log evaluation results."""
        self.logger.info(
            f"Evaluation completed - Samples: {sample_count}, "
            f"Metrics: {json.dumps(metrics, indent=2)}"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        self.logger.error(
            f"Error in {context}: {type(error).__name__}: {str(error)}",
            exc_info=True
        )


class PerformanceLogger:
    """Logger for performance monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        """
        Initialize performance logger.
        
        Args:
            logger_name: Logger name
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            self._setup_handler()
    
    def _setup_handler(self):
        """Set up performance logging handler."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.logger.info(
            f"PERF - {operation} - Duration: {duration:.3f}s - "
            f"Metrics: {json.dumps(metrics)}"
        )


def log_execution_time(operation_name: str):
    """Decorator to log execution time of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                perf_logger = PerformanceLogger()
                perf_logger.log_performance(
                    operation_name,
                    duration,
                    function=func.__name__,
                    success=True
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                perf_logger = PerformanceLogger()
                perf_logger.log_performance(
                    operation_name,
                    duration,
                    function=func.__name__,
                    success=False,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Set up application-wide logging.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        console_output: Whether to output to console
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class StructuredLogger:
    """Logger for structured data output."""
    
    def __init__(self, name: str = "structured"):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            self._setup_handler()
    
    def _setup_handler(self):
        """Set up structured logging handler."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.handlers.RotatingFileHandler(
            log_dir / "structured.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        handler.setLevel(logging.INFO)
        
        # Custom formatter for JSON output
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)
                
                return json.dumps(log_entry)
        
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured event."""
        extra_data = {"event_type": event_type, **data}
        self.logger.info(f"Event: {event_type}", extra={"extra_data": extra_data})


# Global logger instances
_translation_logger: Optional[TranslationLogger] = None
_performance_logger: Optional[PerformanceLogger] = None
_structured_logger: Optional[StructuredLogger] = None


def get_translation_logger() -> TranslationLogger:
    """Get the global translation logger."""
    global _translation_logger
    if _translation_logger is None:
        _translation_logger = TranslationLogger()
    return _translation_logger


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger


if __name__ == "__main__":
    # Test logging setup
    setup_logging(level="DEBUG")
    
    # Test different loggers
    translation_logger = get_translation_logger()
    performance_logger = get_performance_logger()
    structured_logger = get_structured_logger()
    
    # Test logging
    translation_logger.log_translation_start(
        "Hello world", "en-fr", max_length=512
    )
    
    performance_logger.log_performance(
        "test_operation", 1.234, samples=100, accuracy=0.95
    )
    
    structured_logger.log_event(
        "translation_completed",
        {"language_pair": "en-fr", "duration": 1.234, "success": True}
    )
    
    print("Logging test completed!")

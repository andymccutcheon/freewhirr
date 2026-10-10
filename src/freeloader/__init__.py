"""freeloader — use every free LLM before you pay a cent."""

from freeloader.config import FreeloaderConfig, load_config
from freeloader.errors import (
    BudgetExceeded,
    ConfigError,
    FailoverError,
    FreeloaderError,
    FreeModelsExhausted,
    OpenRouterError,
    PaidFallbackExhausted,
    QualityFailed,
)
from freeloader.quality import CallableCheck, JsonSchemaCheck, MinLengthCheck, QualityCheck
from freeloader.router import Freeloader
from freeloader.stats import StatsReport, collect_stats
from freeloader.types import ChatResult, Complexity, ExhaustionPolicy, Usage

__version__ = "0.1.0"

__all__ = [
    "BudgetExceeded",
    "CallableCheck",
    "ChatResult",
    "Complexity",
    "ConfigError",
    "ExhaustionPolicy",
    "FailoverError",
    "FreeModelsExhausted",
    "Freeloader",
    "FreeloaderConfig",
    "FreeloaderError",
    "JsonSchemaCheck",
    "MinLengthCheck",
    "OpenRouterError",
    "PaidFallbackExhausted",
    "QualityCheck",
    "QualityFailed",
    "StatsReport",
    "Usage",
    "collect_stats",
    "load_config",
    "__version__",
]

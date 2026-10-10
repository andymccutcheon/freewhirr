"""freewhirr — spend nothing until you choose to."""

from freewhirr._version import __version__
from freewhirr.config import FreewhirrConfig, load_config
from freewhirr.errors import (
    BudgetExceeded,
    ConfigError,
    FailoverError,
    FreeModelsExhausted,
    FreewhirrError,
    OpenRouterError,
    PaidFallbackExhausted,
    QualityFailed,
)
from freewhirr.quality import (
    CallableCheck,
    JsonSchemaCheck,
    MinLengthCheck,
    QualityCheck,
    ToolCallCheck,
)
from freewhirr.router import Freewhirr
from freewhirr.stats import StatsReport, collect_stats
from freewhirr.types import ChatResult, Complexity, ExhaustionPolicy, Usage

__all__ = [
    "BudgetExceeded",
    "CallableCheck",
    "ChatResult",
    "Complexity",
    "ConfigError",
    "ExhaustionPolicy",
    "FailoverError",
    "FreeModelsExhausted",
    "Freewhirr",
    "FreewhirrConfig",
    "FreewhirrError",
    "JsonSchemaCheck",
    "MinLengthCheck",
    "OpenRouterError",
    "PaidFallbackExhausted",
    "QualityCheck",
    "QualityFailed",
    "StatsReport",
    "ToolCallCheck",
    "Usage",
    "collect_stats",
    "load_config",
    "__version__",
]

from __future__ import annotations

from importlib.metadata import version

from freewhirr import __version__
from freewhirr._version import __version__ as source_version


def test_version_is_single_source() -> None:
    assert __version__ == source_version
    assert version("freewhirr") == source_version
    assert source_version == "0.1.0"

"""Pixeltable session storage for the OpenAI Agents SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from openai_agents_pixeltable.session import PixeltableSession

__all__ = ["PixeltableSession"]

try:
    __version__ = _pkg_version("openai-agents-pixeltable")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0.0.0"

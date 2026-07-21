"""Deployment artifact exporters."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .onnx import OnnxExportResult, export_onnx

__all__ = ["OnnxExportResult", "export_onnx"]


def __getattr__(name: str):
    if name in __all__:
        from . import onnx

        return getattr(onnx, name)
    raise AttributeError(name)

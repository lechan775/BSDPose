from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import nn

from .cbam import CBAM, SEAttention


AttentionVariant = Literal["cbam", "se"]
DEFAULT_PAN_OUTPUT_LAYERS = (15, 18, 21)


@dataclass(frozen=True)
class AttentionInjection:
    layer_index: int
    channels: int
    variant: str


class AttentionWrappedModule(nn.Module):
    """Wrap an Ultralytics layer and apply attention to its output."""

    def __init__(self, module: nn.Module, attention: nn.Module, variant: str) -> None:
        super().__init__()
        self.module = module
        self.attention = attention
        self.attention_variant = variant

        # Ultralytics BaseModel uses these attributes during graph execution.
        for attr in ("i", "f", "type"):
            if hasattr(module, attr):
                setattr(self, attr, getattr(module, attr))
        self.np = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.module(x))


def _infer_out_channels(module: nn.Module) -> int:
    candidates = [
        ("conv", "out_channels"),
        ("cv2", "conv", "out_channels"),
        ("cv3", "conv", "out_channels"),
        ("m", "0", "cv2", "conv", "out_channels"),
    ]
    for path in candidates:
        obj: object = module
        try:
            for name in path:
                obj = obj[int(name)] if isinstance(obj, nn.Sequential) and name.isdigit() else getattr(obj, name)
            if isinstance(obj, int):
                return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise ValueError(f"Could not infer output channels for layer {module!r}")


def _get_ultralytics_layers(yolo_or_model: object) -> nn.ModuleList | nn.Sequential:
    model = getattr(yolo_or_model, "model", yolo_or_model)
    if hasattr(model, "model") and isinstance(model.model, (nn.ModuleList, nn.Sequential)):
        return model.model
    raise TypeError("Expected an Ultralytics YOLO object or a model with model.model ModuleList")


def inject_pan_attention(
    yolo_or_model: object,
    variant: AttentionVariant = "cbam",
    target_layers: Iterable[int] = DEFAULT_PAN_OUTPUT_LAYERS,
    reduction_ratio: int = 16,
) -> list[AttentionInjection]:
    """Insert CBAM or SE blocks after YOLOv8-Pose PAN P3/P4/P5 output layers.

    The default target layer indices match Ultralytics yolov8n-pose:
    P3=15, P4=18, P5=21, feeding the Pose head at layer 22.
    """

    layers = _get_ultralytics_layers(yolo_or_model)
    injections: list[AttentionInjection] = []
    for idx in target_layers:
        if idx < 0 or idx >= len(layers):
            raise IndexError(f"target layer {idx} is outside model layer range 0..{len(layers) - 1}")

        layer = layers[idx]
        if isinstance(layer, AttentionWrappedModule):
            if layer.attention_variant == variant:
                injections.append(
                    AttentionInjection(idx, _infer_out_channels(layer.module), layer.attention_variant)
                )
                continue
            layer = layer.module

        channels = _infer_out_channels(layer)
        attention: nn.Module
        if variant == "cbam":
            attention = CBAM(channels, reduction_ratio=reduction_ratio)
        elif variant == "se":
            attention = SEAttention(channels, reduction_ratio=reduction_ratio)
        else:
            raise ValueError("variant must be 'cbam' or 'se'")

        layers[idx] = AttentionWrappedModule(layer, attention, variant)
        injections.append(AttentionInjection(idx, channels, variant))
    return injections


def build_yolov8_pose_with_attention(
    weights_or_yaml: str | Path = "yolov8n-pose.pt",
    variant: AttentionVariant = "cbam",
    target_layers: Iterable[int] = DEFAULT_PAN_OUTPUT_LAYERS,
    reduction_ratio: int = 16,
):
    """Load an Ultralytics YOLOv8-Pose model and insert PAN attention modules."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLOv8-Pose training") from exc

    model = YOLO(str(weights_or_yaml))
    injections = inject_pan_attention(model, variant=variant, target_layers=target_layers, reduction_ratio=reduction_ratio)
    model.attention_injections = injections
    return model
